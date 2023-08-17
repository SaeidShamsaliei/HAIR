from pyexpat import model
import tensorflow as tf
from osgeo import osr
from osgeo import ogr
import gdal
from cv2 import medianBlur
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import time
import imageio
import glob
import os
import sys
import numpy as np
from utils import image_processing_utils, gdal_utils, model_utils
import data_processing as d_p
from utils.miou_utils import get_miou, get_cof_arr
from pip import main
import pandas as pd
from tabnanny import verbose
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def predict_model_one_image(model_fn, big_image_path_fn, big_image_shape,
                            image_size=512,
                            add_filter_channel=False,
                            intensity_correction=0.0,
                            median_after_segmentation=True
                            ):

    big_image_name = os.path.split(big_image_path_fn)[-1]

    images = d_p.divide_image(big_image_path_fn, big_image_path_fn,
                              image_size=image_size, do_crop=False,
                              do_overlap=False)
    # Make predictions
    for image in images:
        data = model_utils.convert_training_images_to_numpy_arrays([image],
                                                                   add_filter_channel=add_filter_channel)[
            0]
        data += intensity_correction / (2 ** 8 - 1)
        if not add_filter_channel and data.shape[-1] != 3:
            # print('add filter channel')
            data = model_utils.fake_colors(data)
        prediction = model_fn.predict(data)
        prediction = np.argmax(prediction, axis=-1)
        prediction = np.squeeze(prediction)
        image.labels = prediction

    big_image_array = d_p.reassemble_big_image(images, small_image_size=image_size,
                                               big_image_shape=big_image_shape)

    if median_after_segmentation:
        big_image_array = image_processing_utils.median_filter(
            image=big_image_array, kernel_size=7)

    return big_image_array


def calculate_test_miou_magnet_conf(predict_path, dataset_path,
                                    number_classes=6, median_after_segmentation=True,
                                    verbose=0):
    # first predicts each image and adds the miou to the list, then averages the means
    # loops through all the images

    miou_list = []
    conf_list = []
    miou_arr_list = []

    # load images
    img_paths = glob.glob(os.path.join(dataset_path, 'image', '*.tif'))
    lbl_paths = glob.glob(os.path.join(dataset_path, 'label', '*.tif'))
    pred_paths = glob.glob(os.path.join(predict_path, '*.png'))

    img_paths.sort()
    lbl_paths.sort()
    pred_paths.sort()

    print(pred_paths)

    for idx, img in enumerate(img_paths):

        print(f'working on image {os.path.split(img_paths[idx])[-1]}')
        assert os.path.split(
            img_paths[idx])[-1] == os.path.split(lbl_paths[idx])[-1]

        assert os.path.split(
            pred_paths[idx])[-1].replace('png', 'tif') == os.path.split(lbl_paths[idx])[-1]

        test_label_path = lbl_paths[idx]
        test_predict_path = pred_paths[idx]

        # get the ground truth
        label_matrix = gdal_utils.read_tiff_file(large_image_path=test_label_path,
                                                 normalize=False,
                                                 zeropadsize=None,
                                                 numpy_array_only=True,
                                                 grayscale_only=False)

        prediction_matrix = image_processing_utils.read_png_file(
            image_path=test_predict_path,)

        # get the miou of single image
        if label_matrix.ndim > 2:
            label_matrix = label_matrix[:, :, 0]
        # get the miou of single image
        if prediction_matrix.ndim > 2:
            prediction_matrix = prediction_matrix[:, :, 0]

        # get conf matrix
        conf, miou_arr = get_cof_arr(true_label=label_matrix,
                                     predict_label=prediction_matrix,
                                     number_classes=number_classes)

        miou = get_miou(true_label=label_matrix,
                        predict_label=prediction_matrix,
                        number_classes=number_classes)

        if verbose == 1:
            print(f'miou is {miou}')

        miou_list.append(miou)
        miou_arr_list.append(miou_arr)
        conf_list.append(conf)

    # get the miou arr for each test set
    miou_arr_np = np.stack(miou_arr_list)
    miou_arr_avg = miou_arr_np.mean(axis=0)

    # get the confusion matrix
    conf_mat_np = np.stack(conf_list).sum(axis=0)
    conf_mat_df = pd.DataFrame(conf_mat_np)

    conf_mat_df = conf_mat_df.rename(columns={0: 'U', 1: 'G', 2: 'V', 3: 'F', 4: 'A', 5: 'W'},
                                     index={0: 'U', 1: 'G', 2: 'V', 3: 'F', 4: 'A', 5: 'W'})
    conf_mat_df = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(4)

    return np.stack(miou_list).mean(), miou_list, miou_arr_avg, conf_mat_df


def calculate_test_miou_predict(model_path, dataset_path,
                                image_size=512, number_classes=6,
                                median_after_segmentation=True, verbose=0):
    # first predicts each image and adds the miou to the list, then averages the means
    # loops through all the images

    have_lr_scheduler = True
    miou_list = []

    # load the model
    model = model_utils.load_model(
        model_path, have_lr_scheduler=have_lr_scheduler)

    # load images
    img_paths = glob.glob(os.path.join(dataset_path, 'image', '*.tif'))
    lbl_paths = glob.glob(os.path.join(dataset_path, 'label', '*.tif'))

    for idx, img in enumerate(img_paths):

        print(f'working on image {os.path.split(img_paths[idx])[-1]}')
        assert os.path.split(
            img_paths[idx])[-1] == os.path.split(lbl_paths[idx])[-1]

        test_image_path = img_paths[idx]
        test_label_path = lbl_paths[idx]

        # get the ground truth
        label_matrix = gdal_utils.read_tiff_file(large_image_path=test_label_path,
                                                 normalize=False,
                                                 zeropadsize=None,
                                                 numpy_array_only=True,
                                                 grayscale_only=False)

        # get the prediction
        prediction_matrix = predict_model_one_image(model_fn=model,
                                                    big_image_path_fn=test_image_path,
                                                    image_size=512,
                                                    big_image_shape=(
                                                        label_matrix.shape[0], label_matrix.shape[1]),
                                                    median_after_segmentation=True,)

        # get the miou of single image
        if label_matrix.ndim > 2:
            label_matrix = label_matrix[:, :, 0]
        # get the miou of single image
        if prediction_matrix.ndim > 2:
            prediction_matrix = prediction_matrix[:, :, 0]

        miou = get_miou(true_label=label_matrix,
                        predict_label=prediction_matrix,
                        number_classes=number_classes)
        if verbose == 1:
            print(f'miou is {miou}')

        miou_list.append(miou)

    return np.stack(miou_list).mean(), miou_list


if __name__ == '__main__':

    base_prediction_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/magnet_predictions'
    base_dataset_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/dataset_division/all_test_tif'
    base_dest_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/results/magnet'
    test_type_list = ['IOD', 'OOD']
    training_type_list = ['normal', 'pretrain']

    seed_name_list = ['seed4', 'seed5', 'seed6', 'seed7', 'seed8']

    for test_type in test_type_list:
        for training_type in training_type_list:
            for seed_name in seed_name_list:
                print(f'working on {test_type}, {training_type}, {seed_name}')
                prediction_path = os.path.join(
                    base_prediction_path, test_type, training_type, seed_name)
                testset_path = os.path.join(base_dataset_path, test_type)
                model_name = f'{training_type}_{seed_name}'
                dest_path = os.path.join(
                    base_dest_path, test_type, training_type, model_name)

                # prediction_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/dataset_division/magent_prediction97/TestSet'
                # testset_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/dataset_division/all_test_tif/TestSet/tif'
                # model_name = 'magent_prediction97'
                # dest_path = f'/home/saeid/phd/segmentation/dataset/neurips dataset/results/nips_5times/TestSet/magnet/{model_name}'
                """
                testing the models five times.
                """

                no_loop = 1

                n_classes = 6
                miou_list = []
                verbose = 1

                os.makedirs(dest_path, exist_ok=True)

                for rep_no in range(0, no_loop):

                    miou_final, miou_list, miou_arr, conf_df = calculate_test_miou_magnet_conf(
                        predict_path=prediction_path,
                        dataset_path=testset_path,
                        number_classes=6, median_after_segmentation=True,
                        verbose=1)

                    # write the files to csv and txt

                    csv_name = f'{model_name}_{rep_no}_list.csv'
                    csv_confmatrix = f'{model_name}_{rep_no}_confmat.csv'
                    txt_name = f'{model_name}_{rep_no}_overall.txt'
                    txt_arr_name = f'{model_name}_{rep_no}_arr.txt'

                    f = open(os.path.join(dest_path, txt_name), "w")
                    f.write(str(miou_final))
                    f.close()

                    f = open(os.path.join(dest_path, txt_arr_name), "w")
                    f.write(str(miou_arr))
                    f.close()

                    conf_df.to_csv(os.path.join(
                        dest_path, csv_confmatrix), index=False)

                    df = pd.DataFrame(miou_list)
                    df.to_csv(os.path.join(dest_path, csv_name), index=False)

                    print(f'repeat {rep_no} done')
