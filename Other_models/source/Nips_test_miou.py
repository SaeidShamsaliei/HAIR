import pandas as pd
from tabnanny import verbose
from PIL import Image
from pip import main
from utils.miou_utils import get_miou, get_cof_arr
import data_processing as d_p

from utils import image_processing_utils, gdal_utils, model_utils
import numpy as np
import sys
import os
import glob
import imageio
import time
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from cv2 import medianBlur
import gdal
from osgeo import ogr
from osgeo import osr
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def predict_model_one_image(model_fn, big_image_path_fn, big_image_shape,
                            image_size=512,
                            add_filter_channel=False,
                            intensity_correction=0.0,
                            median_after_segmentation=False
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
                                               big_image_shape=big_image_shape,
                                               unknown_zero=True)

    if median_after_segmentation:
        big_image_array = image_processing_utils.median_filter(
            image=big_image_array, kernel_size=7)

    return big_image_array


def calculate_test_miou_predict_conf(model_path, dataset_path,
                                     image_size=512, number_classes=6,
                                     median_after_segmentation=True, verbose=0,
                                     swin_transformer_flag=False, return_names=False):
    # first predicts each image and adds the miou to the list, then averages the means
    # loops through all the images

    have_lr_scheduler = True
    miou_list = []
    conf_list = []
    miou_arr_list = []

    if return_names:
        name_miou_list = []  # a list of tuples (img_name, miou)

    # load the model
    model = model_utils.load_model(
        model_path, have_lr_scheduler=have_lr_scheduler,
        swin_transformer_flag=swin_transformer_flag)

    # load images
    img_paths = glob.glob(os.path.join(dataset_path, 'image', '*.tif'))
    lbl_paths = glob.glob(os.path.join(dataset_path, 'label', '*.tif'))

    for idx, img in enumerate(img_paths):

        image_name = os.path.split(img_paths[idx])[-1]
        print(f'working on image {image_name}')
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
                                                    median_after_segmentation=median_after_segmentation,)

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
        if return_names:
            name_miou_list.append((image_name, miou))

    # get the miou arr for each test set
    miou_arr_avg = np.stack(miou_arr_list).mean(axis=0)

    # get the confusion matrix
    conf_mat_np = np.stack(conf_list).sum(axis=0)
    conf_mat_df = pd.DataFrame(conf_mat_np)

    conf_mat_df = conf_mat_df.rename(columns={0: 'U', 1: 'G', 2: 'V', 3: 'F', 4: 'A', 5: 'W'},
                                     index={0: 'U', 1: 'G', 2: 'V', 3: 'F', 4: 'A', 5: 'W'})
    conf_mat_df = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(4)

    if return_names:
        return np.stack(miou_list).mean(), miou_list, miou_arr_avg, conf_mat_df, name_miou_list
    else:
        return np.stack(miou_list).mean(), miou_list, miou_arr_avg, conf_mat_df, []


if __name__ == '__main__':

    """
    testing the models five times.
    """

    # number of times the evaluation takes place
    no_loop = 1

    # list of paths to the model
    model_list = [
        # '/home/saeid/phd/segmentation/dataset/neurips dataset/exp/all_exp_5_times/swin/exp_no51/2022-09-13_11:56:41.309159_swin_transformer_freeze_0/checkpoints/cp.ckpt',
        # '/home/saeid/phd/segmentation/dataset/neurips dataset/exp/all_exp_5_times/hrnet/2022-09-14_00:45:06.560321_hrnet_freeze_0/model.hdf5',
    ]

    # Check if swin transformer are True
    swin_transformer_flag = False

    # Get list of rivers name to calaculate the MIoU of these rivers separately
    gaula_1998_test_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/dataset_division/all_test_tif/gaula_1998/image'

    orkla_1963_test_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/dataset_division/all_test_tif/orkla_1962/image'

    gaula_1998_test_list = [os.path.split(tmp_path)[1] for tmp_path in
                            glob.glob(
                                os.path.join(gaula_1998_test_path, '*.tif'))]

    orkla_1963_test_list = [os.path.split(tmp_path)[1] for tmp_path in
                            glob.glob(
                                os.path.join(orkla_1963_test_path, '*.tif'))]

    # a list of tuples of (river_name, list_of_image_names)
    individual_test_sets = [
        ('gaula_1998', gaula_1998_test_list),
        ('orkla_1962', orkla_1963_test_list),
    ]

    # model_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/exp/exp_no994/2022-08-21_09:46:48.634903_hrnet_freeze_0/model_swa.hdf5'

    for model_path in model_list:
        print('model is:')
        print(model_path)

        if swin_transformer_flag:
            model_name = os.path.split(os.path.split(
                os.path.split(model_path)[0])[0])[1]
        else:
            model_name = os.path.split(os.path.split(model_path)[0])[1]

            # chack if the name of the function had any swintransformer and the flag is off
            if model_name.find('swin') != -1:
                raise ValueError(
                    f'When model is swin transformer, the swin_transformer_flag should be True while it is {swin_transformer_flag}')

        testset_path = '/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/dataset_division/all_test_tif/OOD'
        dest_path = f'/home/saeid/phd/segmentation/dataset/neurips dataset/2023 submission/results/swin/OOD/{model_name}'

        os.makedirs(dest_path, exist_ok=True)

        return_names = False
        size = 512
        n_classes = 6
        miou_list = []
        verbose = 1
        median_after_segmentation = False

        if return_names and not ('OOD' in testset_path):
            raise ValueError('return_names is set to True for non OOD')

        for rep_no in range(0, no_loop):

            print(
                f'__________________________Strart of the test rep {rep_no}________________________________')

            rep_no = rep_no
            miou_final, miou_list, miou_arr, conf_df, name_miou_list = calculate_test_miou_predict_conf(
                model_path=model_path,
                dataset_path=testset_path,
                image_size=size,
                number_classes=n_classes,
                median_after_segmentation=median_after_segmentation,
                swin_transformer_flag=swin_transformer_flag,
                verbose=verbose,
                return_names=return_names)

            # write the files to csv and txt

            model_name = os.path.split(os.path.split(model_path)[0])[1]
            csv_name = f'{model_name}_{rep_no}_list.csv'
            csv_confmatrix = f'{model_name}_{rep_no}_confmat.csv'
            txt_name = f'{model_name}_{rep_no}_overall.txt'
            txt_arr_name = f'{model_name}_{rep_no}_arr.txt'

            os.path.join(dest_path, csv_name)

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

            print(f'The overall MIoU of set is: {miou_final}')

            # get individual's miou
            if return_names:
                for (test_river_name, test_river_image_name_list) in individual_test_sets:
                    print(f'The MIoU of river {test_river_name}')
                    individual_miou_list = []

                    # we need to get all the mious which are in the test_river_image_name_list
                    for (individual_test_pred_name, individual_test_pred_miou) in name_miou_list:
                        if individual_test_pred_name in test_river_image_name_list:
                            individual_miou_list.append(
                                individual_test_pred_miou)

                    # make the mean
                    if not len(individual_miou_list):
                        print(f'There is no picture of {test_river_name}')
                    else:
                        individual_miou_np = np.array(individual_miou_list)
                        individual_miou_mean = np.mean(individual_miou_np)
                        print(
                            f'The MIoU of river {test_river_name} is: {individual_miou_mean}')

                        # write the miou to a file
                        txt_individual_name = f'{model_name}_{rep_no}_overall_{test_river_name}.txt'
                        f = open(os.path.join(
                            dest_path, txt_individual_name), "w")
                        f.write(str(individual_miou_mean))
                        f.close()

            print(f'End')

            print(
                f'__________________________repeat {rep_no} done__________________________')
