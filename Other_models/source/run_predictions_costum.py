from utils import gdal_utils
from tqdm import tqdm
import time
# import utils.image_processing_utils as image_processing_utils
from utils.model_utils import load_model, convert_training_images_to_numpy_arrays, fake_colors, sliding_window_ensemble
import data_processing as d_p
import glob
import os
import sys
import numpy as np
import gdal
from scipy import stats
import yaml
from datetime import date
import tensorflow as tf
from utils.gdal_utils import read_tiff_file
from utils.image_processing_utils import median_filter
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# first party imports
"""
This script should be used to make predictions on a set of big images (6000x8000 pixels). Image of other sizes should
also work but this has not been tested. A folder of images must be provided as well as a trained model.
The predictions will be written to a output folder. The output images will be in
the big image format as geo referenced tiff files.
"""


class PipelineTrackerClass:
    """
    To keep track of status of pipeline, keep track of time, operation being done, etc.
    """

    def __init__(self):
        self.start_time = time.time()
        self.time_passed = 0
        self.time_passed_operation = 0
        self.operation_name = None

    def get_operation_name(self, operation_name_val, print_flage=True):
        self.operation_name = operation_name_val
        if print_flage:
            print(f'Operation {self.operation_name} begins')

    def format_seconds(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def report_total_time(self):
        # calculate the time passed
        self.time_passed_operation = (
            time.time() - self.start_time) - self.time_passed
        self.time_passed = time.time() - self.start_time

        print(f'Total time so far: {self.format_seconds(self.time_passed)}'
              f'Operation: {self.operation_name},\tTime spent: {self.format_seconds(self.time_passed_operation)}')


def run_ensemble(model_path_list, input_folder, output_folder, intensity_correction=0.0, add_filter_channel=False):
    if not os.path.exists(input_folder):
        print(f'the input path {input_folder} not exist')

    model_list = []
    for model_path in model_path_list:
        model = load_model(model_path)
        model_list.append(model)

    big_image_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    # make directory for output
    os.makedirs(output_folder, exist_ok=True)

    # just to have time of each experiment

    for idx, big_image_path in enumerate(big_image_paths):
        print(f'predicting image {idx}')
        start_time_tmp = time.time()

        images = d_p.divide_image(big_image_path, big_image_path, image_size=512, do_crop=False,
                                  do_overlap=False)
        # Make predictions
        for image in tqdm(images):
            data = convert_training_images_to_numpy_arrays([image])[
                0]
            data += intensity_correction / (2 ** 8 - 1)
            data = fake_colors(data)

            prediction_list = []
            for model in model_list:
                # do the ensemble here
                prediction = model.predict(data)
                prediction = np.argmax(prediction, axis=-1)
                prediction = np.squeeze(prediction)
                prediction_list.append(prediction)

            # get the mode of all predictions
            all_predictions = np.stack(prediction_list)
            mode_all_predictions, count = stats.mode(all_predictions, axis=0)
            image.labels = mode_all_predictions

        big_image_ds = gdal.Open(big_image_path)
        geo_transform = big_image_ds.GetGeoTransform()
        projection = big_image_ds.GetProjection()
        big_image_shape = (big_image_ds.RasterYSize, big_image_ds.RasterXSize)
        big_image_ds = None  # Close the image the gdal way

        big_image_array = d_p.reassemble_big_image(images, small_image_size=512,
                                                   big_image_shape=big_image_shape)
        big_image = d_p.TrainingImage(big_image_array, big_image_array, geo_transform,
                                      projection=projection, name=os.path.split(big_image_path)[-1])
        big_image.write_labels_to_raster(
            os.path.join(output_folder, big_image.name))

        print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_tmp)))


def run_sliding_window(model_path, input_folder, output_folder, intensity_correction=0.0, add_filter_channel=False):
    """
    Function which uses ensemble of 4 different sliding window
    :param model_path: path to the segmentation model
    :param input_folder: path to input directory
    :param output_folder: path to output directory
    :param intensity_correction: is not being used here
    :return: write the file to output path and returns nothing
    """
    # get the value inside the model path list
    model_path = model_path[0]
    os.makedirs(output_folder, exist_ok=True)
    input_folder = glob.glob(os.path.join(input_folder, "*.tif"))

    for idx, big_image_path in enumerate(input_folder):
        print(f'working on image {idx}')
        l_image_matrix, geo_transform, projection = gdal_utils.get_numpy_array_large_image(
            big_image_path, normalize=True, numpy_array_only=False)
        l_image_matrix += intensity_correction / (2 ** 8 - 1)
        prediction_ensemble = sliding_window_ensemble(
            l_image_matrix, model_path)
        l_image_label = prediction_ensemble
        print(f'label shape is: {l_image_label.shape}')
        big_image = d_p.TrainingImage(l_image_matrix, l_image_label, geo_transform,
                                      projection=projection, name=os.path.split(big_image_path)[-1])
        big_image.write_labels_to_raster(
            os.path.join(output_folder, big_image.name))


def run(model_path, input_folder, output_folder, intensity_correction=0.0,
        median_after_segmentation=False, add_filter_channel=False,
        have_lr_scheduler=False):
    """
    add_filter_channel: determines if filters will be added as extera channels
    """
    if not os.path.exists(input_folder):
        print(f'the input path {input_folder} not exist')
    # get the value inside the model path list
    model_path = model_path[0]
    model = load_model(model_path, have_lr_scheduler=have_lr_scheduler)

    big_image_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    # make directory for output
    os.makedirs(output_folder, exist_ok=True)

    for idx, big_image_path in enumerate(big_image_paths):
        print(f'predicting image:{idx} name:{os.path.split(big_image_path)[-1]}')
        start_time_tmp = time.time()

        images = d_p.divide_image(big_image_path, big_image_path, image_size=512, do_crop=False,
                                  do_overlap=False)
        # Make predictions
        for image in images:
            data = convert_training_images_to_numpy_arrays([image],
                                                           add_filter_channel=add_filter_channel)[
                0]
            data += intensity_correction / (2 ** 8 - 1)
            if not add_filter_channel and data.shape[-1] != 3:
                # print('add filter channel')
                data = fake_colors(data)
            prediction = model.predict(data)
            prediction = np.argmax(prediction, axis=-1)
            prediction = np.squeeze(prediction)
            image.labels = prediction

        big_image_ds = gdal.Open(big_image_path)
        geo_transform = big_image_ds.GetGeoTransform()
        projection = big_image_ds.GetProjection()
        big_image_shape = (big_image_ds.RasterYSize, big_image_ds.RasterXSize)
        big_image_ds = None  # Close the image the gdal way

        big_image_array = d_p.reassemble_big_image(images, small_image_size=512,
                                                   big_image_shape=big_image_shape,
                                                   unknown_zero=True,)
        # median filter performing
        if median_after_segmentation:
            big_image_array = median_filter(
                image=big_image_array, kernel_size=7)

        big_image = d_p.TrainingImage(big_image_array, big_image_array, geo_transform,
                                      projection=projection, name=os.path.split(big_image_path)[-1])
        big_image.write_labels_to_raster(
            os.path.join(output_folder, big_image.name))
        print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_tmp)))


def median_filter_directory(input_folder, output_folder, kernel_size, new_name_extra=''):
    """
    loads tif files and performs median filter and save them with original_namenew_name_extra.tif
    """
    if not os.path.exists(input_folder):
        print(f'the input path {input_folder} not exist')

    big_image_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    # make directory for output
    os.makedirs(output_folder, exist_ok=True)

    for idx, big_image_path in enumerate(big_image_paths):
        print(f'filtering image {idx}')
        start_time_tmp = time.time()
        big_image_name = os.path.split(big_image_path)[-1].replace('.tif', '')
        big_image_new_name = f'{big_image_name}{new_name_extra}.tif'
        big_image_array, geo_transform, projection = read_tiff_file(large_image_path=big_image_path,
                                                                    normalize=False, zeropadsize=None,
                                                                    numpy_array_only=False, grayscale_only=False)
        big_image_array = median_filter(
            image=big_image_array, kernel_size=kernel_size)

        big_image = d_p.TrainingImage(big_image_array, big_image_array, geo_transform,
                                      projection=projection, name=big_image_new_name)

        big_image.write_labels_to_raster(
            os.path.join(output_folder, big_image.name))

        print(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_tmp)))


# run dictionary
run_dict = {'single_model': run, 'ensemble_model': run_ensemble,
            'sliding_window_model': run_sliding_window}


if __name__ == '__main__':
    print('can you hear me?????????????????????????/')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-conf', type=str,
                        default='./configs/prediction_config.yaml')
    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # check if the model path is a list
    assert type(
        config['model_path']) is list, f'model_path should be list but it is {config["model_path"].type}'

    # Write config to the output directory
    conf_save_file = os.path.join(config['output_path'], 'config')
    if os.path.exists(os.path.join(conf_save_file, 'config.yaml')):
        answer = input(
            f'{os.path.split(conf_save_file)[-1]} already exist do you want to delete it (y/n)?')
        if answer == 'y':
            with open(os.path.join(conf_save_file, f'config_{date.today()}_{time.time()}.yaml'), 'w') as f:
                _ = yaml.dump(config, f)
        else:
            print('config is not saved')
    else:
        os.makedirs(conf_save_file, exist_ok=True)
        with open(os.path.join(conf_save_file, 'config.yaml'), 'w') as f:
            _ = yaml.dump(config, f)

    # Define a pipeline controller
    pipelineTracker = PipelineTrackerClass()

    print(f"prediction using {config['experiment']['exp_type']}")
    pipelineTracker.get_operation_name('segmentation')

    # Predict and write to file
    if config['experiment']['segmentation']:
        # run(config['model_path'], config['input_path'], config['output_path'],
        #     config['experiment']['intensity_correction'])
        print(run_dict[config['experiment']['exp_type']])
        run_dict[config['experiment']['exp_type']](config['model_path'], config['input_path'], config['output_path'],
                                                   config['experiment']['intensity_correction'],
                                                   add_filter_channel=config['experiment']['add_filter_channel'],
                                                   median_after_segmentation=config['experiment']['median_after_segmentation'],
                                                   have_lr_scheduler=config['experiment']['have_lr_scheduler'])
    # check the time passed
    pipelineTracker.report_total_time()

    if config['experiment']['median_kernel_size']:
        pipelineTracker.get_operation_name('performing median filter')
        median_filter_directory(
            input_folder=config['output_path'],
            output_folder=config['output_path'],
            kernel_size=int(config['experiment']['median_kernel_size']), new_name_extra='')

    # check and merge
    if config['experiment']['merging']:
        pipelineTracker.get_operation_name('merging tif files')
        gdal_utils.merge_all_rasters_directory(
            config['output_path'], config['experiment']['output_name'], config['output_path'])
        # check the time passed
        pipelineTracker.report_total_time()

    # check and clip the rasters based on boundary shapefile
    if config['experiment']['clip']:
        pipelineTracker.get_operation_name('cliping the river boundary')
        rasin = os.path.join(
            config['output_path'], f"{config['experiment']['output_name']}.tif")
        rasout = os.path.join(
            config['output_path'], f"{config['experiment']['output_name']}_clipped.tif")
        gdal_utils.clip_raster_using_polygon(rasin=rasin,
                                             shpin=config['boundary_shape_path'],
                                             rasout=rasout)

    # check if polygon should be created
    if config['experiment']['polygonize']:
        pipelineTracker.get_operation_name('create polygons')
        gdal_utils.convert_tif_to_shape_directory(config['output_path'],
                                                  f"{config['experiment']['output_name']}.tif")
        # check the time passed
        pipelineTracker.report_total_time()
