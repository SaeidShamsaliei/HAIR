from tkinter.messagebox import NO
import numpy as np
import sys
import os
import glob
import imageio
import pandas as pd
import time
from scipy import ndimage, signal
from cv2 import medianBlur

from PIL import Image
from utils import image_processing_utils, gdal_utils, model_utils
from data_processing import TrainingImage

# try with color image instead of label


def get_augment_rotate_double_colored_deepglobe(image_matrix, label_matrix_changed, label_matrix_changed_2d, small_image_size, selection_lim, selected_class,
                                      angle, large_image_file_name, labeled_percentage=0.1,
                                      overlap_percentage=0.5, method='skimage', rotation_noise=0, unknown_zero_flag=True, verbose=1, large_image=True, most_distant=False):
    """
    make the image to have same shape as deepglobe
    rotates image_matrix and label_matrix_changed based on angle and samples small_image_sizes which has the selected_class in them
    Note: label_matrix_changed_2d shouild be provided to the function

    param selected_lim: some lower bound to the number of sampling
    param selected_class: should be 3d [R,G,B]
    TODO we can sample multiple classes here just to have a better distribution
    """

    original_image_x_shape = image_matrix.shape[0]
    original_image_y_shape = image_matrix.shape[1]

    start = time.time()
    float_epsilon = np.finfo(float).eps

    # selection_class should be a vector of color
    assert selected_class.ndim == 1 and selected_class.shape[0] == 3
    selected_class_2d = image_processing_utils.single_rgb_to_label(
        selected_class, unknown_zero_flag=unknown_zero_flag, most_distant=most_distant)

    # check if the picture has more than 100 pixel of selected calss (for example gravel)
    # get the 2d image
    # color_matrix = np.array([[0, 0, 0], [255, 255, 0], [0, 128, 0], [
    #                         128, 128, 0], [128, 0, 128], [0, 0, 255]], dtype=np.uint8)

    # test the distance issue
    color_matrix = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
        0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)
    color_matrix_float = color_matrix / 255

    arr_tmp = np.zeros((1, 6))
    unique_tmp, count_tmp = np.unique(
        label_matrix_changed_2d, return_counts=True)
    arr_tmp[0, unique_tmp] = count_tmp

    # csv list consists of [large image name, angle, xcenter, ycenter]
    csv_list = []
    image_list = []
    label_list = []
    global_counter = 0
    not_useful_count = 0
    constant_selection_lim = selection_lim

    angle += rotation_noise
    # set the angle
    if method == 'scipy':
        tmp_large_image, tmp_large_label = image_processing_utils.rotate_scipy(
            angle, image_matrix, label_matrix_changed)
    elif method == 'skimage':
        tmp_large_image, tmp_large_label = image_processing_utils.rotate_skimage(
            angle, image_matrix, label_matrix_changed)
    else:
        raise ('method is not defined')

    if verbose == 1:
        end = time.time()
        print(f'\t rotated the image the time takes {end - start}')

    # apply the correction of filter
    tmp_large_label = tmp_large_label / 255
    if large_image:
        tmp_large_label = image_processing_utils.color_nearest_neighbors_large(
            img_matrix=tmp_large_label, clr_matrix=color_matrix_float)
    else:
        tmp_large_label = image_processing_utils.color_nearest_neighbors(
            img_matrix=tmp_large_label, clr_matrix=color_matrix_float)
    tmp_large_label = tmp_large_label * 255
    end = time.time()
    print(f'\t color neighbors the time takes {end - start}')

    # make 2d of label
    tmp_large_label_2d = image_processing_utils.replace_rgb_to_class(tmp_large_label,
                                                                     unknown_zero_flag=unknown_zero_flag, most_distant=most_distant)
    # 2 times the median filter
    tmp_large_label_2d = image_processing_utils.median_filter(
        image=tmp_large_label_2d, kernel_size=7)
    tmp_large_label_2d = image_processing_utils.median_filter(
        image=tmp_large_label_2d, kernel_size=7)

    if verbose == 1:
        end = time.time()
        print(f'\t refined the image the time takes {end - start}')

    # make a mask of large image
    # mask_large controls the unknown class
    masked_large = tmp_large_label_2d > 0
    # overlap_large controls the overlap of sampling
    overlap_large = tmp_large_label_2d > -1

    # x, y = np.where(tmp_large_label_2d == selected_class_2d)
    
    x = np.arange(small_image_size, original_image_x_shape-small_image_size)
    y = np.arange(small_image_size, original_image_y_shape-small_image_size)

    # we either get constant_selection_lim images or all the possible small images in the large image
    small_image_area = small_image_size * small_image_size
    selection_lim = max(
        int(x.shape[0]/small_image_area), constant_selection_lim)
    # in case that we dont have any pixels
    if x.shape[0] < 1:
        selection_lim = 0
    if verbose == 1:
        print(
            f'\t x shape is {x.shape[0]} so the number of selection will be {selection_lim}')

    for select_no in range(selection_lim):
        # select one pixel to be the center of the frame
        # in order to avoid forever loops, we set a counter
        while_counter_lim = x.shape[0]
        while_counter = 0
        center_selection_flag = False
        # select the center of the image
        while True:
            # check the lim of the while
            if while_counter > while_counter_lim:
                break
            while_counter += 1

            pixel_idx = np.random.randint(0, min(x.shape[0], y.shape[0]))
            if (small_image_size / 2) < x[pixel_idx] < tmp_large_label.shape[0] - (
                    small_image_size / 2) \
                    and (small_image_size / 2) < y[pixel_idx] < tmp_large_label.shape[1] - (
                    small_image_size / 2):
                center_selection_flag = True
                break
        # if the pixel selected is not in the center
        if not center_selection_flag:
            break
        # center pixel
        ctr_x = x[pixel_idx]
        ctr_y = y[pixel_idx]
        # range of frame work
        x_min = int(ctr_x - (small_image_size / 2))
        x_max = int(ctr_x + (small_image_size / 2))
        y_min = int(ctr_y - (small_image_size / 2))
        y_max = int(ctr_y + (small_image_size / 2))

        # check if more than 50 percent of the labels are masked as True
        label_true_percent = (masked_large[x_min:x_max, y_min:y_max].sum()) / ((
            masked_large[x_min:x_max, y_min:y_max].size) + float_epsilon)
        overlap_true_percent = (overlap_large[x_min:x_max, y_min:y_max].sum()) / ((
            overlap_large[x_min:x_max, y_min:y_max].size) + float_epsilon)
        # get the image

        # For EX2 deepglobe we have 2d labels
        tmp_small_label = tmp_large_label_2d[x_min:x_max, y_min:y_max]
        tmp_small_image = tmp_large_image[x_min:x_max, y_min:y_max]

        # if the frame is on the right spot
        if (label_true_percent > labeled_percentage) and (overlap_true_percent > overlap_percentage):
            image_list.append(tmp_small_image)
            label_list.append(tmp_small_label)
            csv_list.append(
                [global_counter, large_image_file_name, angle, ctr_x, ctr_y])
            global_counter += 1
            # change the overlap mask to indicate that this pixels are already taken
            overlap_large[x_min:x_max, y_min:y_max] = False
        # if frame is not on the right spot
        else:
            not_useful_count += 1
    print(f'\t there were {not_useful_count} not useful images here')
    print(f'\t there were {len(image_list)} useful images here')

    if len(image_list) == 0:
        return False, False, False, False, False
    return image_list, label_list, csv_list, masked_large, overlap_large




def get_augment_rotate_double_colored(image_matrix, label_matrix_changed, label_matrix_changed_2d, small_image_size, selection_lim, selected_class,
                                      angle, large_image_file_name, labeled_percentage=0.1,
                                      overlap_percentage=0.5, method='skimage', rotation_noise=0, unknown_zero_flag=True, verbose=1, large_image=True, most_distant=False):
    """
    rotates image_matrix and label_matrix_changed based on angle and samples small_image_sizes which has the selected_class in them
    Note: label_matrix_changed_2d shouild be provided to the function

    param selected_lim: some lower bound to the number of sampling
    param selected_class: should be 3d [R,G,B]
    TODO we can sample multiple classes here just to have a better distribution
    """
    start = time.time()
    float_epsilon = np.finfo(float).eps

    # selection_class should be a vector of color
    assert selected_class.ndim == 1 and selected_class.shape[0] == 3
    selected_class_2d = image_processing_utils.single_rgb_to_label(
        selected_class, unknown_zero_flag=unknown_zero_flag, most_distant=most_distant)

    # check if the picture has more than 100 pixel of selected calss (for example gravel)
    # get the 2d image
    # color_matrix = np.array([[0, 0, 0], [255, 255, 0], [0, 128, 0], [
    #                         128, 128, 0], [128, 0, 128], [0, 0, 255]], dtype=np.uint8)

    # test the distance issue
    color_matrix = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
        0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)
    color_matrix_float = color_matrix / 255

    arr_tmp = np.zeros((1, 6))
    unique_tmp, count_tmp = np.unique(
        label_matrix_changed_2d, return_counts=True)
    arr_tmp[0, unique_tmp] = count_tmp
    if arr_tmp[0, selected_class_2d] < 1000:
        if verbose == 1:
            print('\t less than 1000 pixel of selected class')
            end = time.time()
            print(f'\t the time takes {end - start}')
        return False, False, False, False, False

    # csv list consists of [large image name, angle, xcenter, ycenter]
    csv_list = []
    image_list = []
    label_list = []
    global_counter = 0
    not_useful_count = 0
    constant_selection_lim = selection_lim

    angle += rotation_noise
    # set the angle
    if method == 'scipy':
        tmp_large_image, tmp_large_label = image_processing_utils.rotate_scipy(
            angle, image_matrix, label_matrix_changed)
    elif method == 'skimage':
        tmp_large_image, tmp_large_label = image_processing_utils.rotate_skimage(
            angle, image_matrix, label_matrix_changed)
    else:
        raise ('method is not defined')

    if verbose == 1:
        end = time.time()
        print(f'\t rotated the image the time takes {end - start}')

    # apply the correction of filter
    tmp_large_label = tmp_large_label / 255

    # print(f'\n first: {tmp_large_label.shape}     last: {color_matrix_float.shape}')

    if large_image:
        tmp_large_label = image_processing_utils.color_nearest_neighbors_large(
            img_matrix=tmp_large_label, clr_matrix=color_matrix_float)
    else:
        tmp_large_label = image_processing_utils.color_nearest_neighbors(
            img_matrix=tmp_large_label, clr_matrix=color_matrix_float)
    tmp_large_label = tmp_large_label * 255
    end = time.time()
    print(f'\t color neighbors the time takes {end - start}')

    # make 2d of label
    tmp_large_label_2d = image_processing_utils.replace_rgb_to_class(tmp_large_label,
                                                                     unknown_zero_flag=unknown_zero_flag, most_distant=most_distant)
    # 2 times the median filter
    tmp_large_label_2d = image_processing_utils.median_filter(
        image=tmp_large_label_2d, kernel_size=7)
    tmp_large_label_2d = image_processing_utils.median_filter(
        image=tmp_large_label_2d, kernel_size=7)

    if verbose == 1:
        end = time.time()
        print(f'\t refined the image the time takes {end - start}')

    # make a mask of large image
    # mask_large controls the unknown class
    masked_large = tmp_large_label_2d > 0
    # overlap_large controls the overlap of sampling
    overlap_large = tmp_large_label_2d > -1

    x, y = np.where(tmp_large_label_2d == selected_class_2d)
    # we either get constant_selection_lim images or all the possible small images in the large image
    small_image_area = small_image_size * small_image_size
    selection_lim = max(
        int(x.shape[0]/small_image_area), constant_selection_lim)
    # in case that we dont have any pixels
    if x.shape[0] < 1:
        selection_lim = 0
    if verbose == 1:
        print(
            f'\t x shape is {x.shape[0]} so the number of selection will be {selection_lim}')

    for select_no in range(selection_lim):
        # select one pixel to be the center of the frame
        # in order to avoid forever loops, we set a counter
        while_counter_lim = x.shape[0]
        while_counter = 0
        center_selection_flag = False
        # select the center of the image
        while True:
            # check the lim of the while
            if while_counter > while_counter_lim:
                break
            while_counter += 1

            pixel_idx = np.random.randint(0, x.shape[0])
            if (small_image_size / 2) < x[pixel_idx] < tmp_large_label.shape[0] - (
                    small_image_size / 2) \
                    and (small_image_size / 2) < y[pixel_idx] < tmp_large_label.shape[1] - (
                    small_image_size / 2):
                center_selection_flag = True
                break
        # if the pixel selected is not in the center
        if not center_selection_flag:
            break
        # center pixel
        ctr_x = x[pixel_idx]
        ctr_y = y[pixel_idx]
        # range of frame work
        x_min = int(ctr_x - (small_image_size / 2))
        x_max = int(ctr_x + (small_image_size / 2))
        y_min = int(ctr_y - (small_image_size / 2))
        y_max = int(ctr_y + (small_image_size / 2))

        # check if more than 50 percent of the labels are masked as True
        label_true_percent = (masked_large[x_min:x_max, y_min:y_max].sum()) / ((
            masked_large[x_min:x_max, y_min:y_max].size) + float_epsilon)
        overlap_true_percent = (overlap_large[x_min:x_max, y_min:y_max].sum()) / ((
            overlap_large[x_min:x_max, y_min:y_max].size) + float_epsilon)
        # get the image

        # For EX2 we have 2d labels
        tmp_small_label = tmp_large_label_2d[x_min:x_max, y_min:y_max]
        tmp_small_image = tmp_large_image[x_min:x_max, y_min:y_max]

        # if the frame is on the right spot
        if (label_true_percent > labeled_percentage) and (overlap_true_percent > overlap_percentage):
            image_list.append(tmp_small_image)
            label_list.append(tmp_small_label)
            csv_list.append(
                [global_counter, large_image_file_name, angle, ctr_x, ctr_y])
            global_counter += 1
            # change the overlap mask to indicate that this pixels are already taken
            overlap_large[x_min:x_max, y_min:y_max] = False
        # if frame is not on the right spot
        else:
            not_useful_count += 1
    print(f'\t there were {not_useful_count} not useful images here')
    print(f'\t there were {len(image_list)} useful images here')

    if len(image_list) == 0:
        return False, False, False, False, False
    return image_list, label_list, csv_list, masked_large, overlap_large


if __name__ == '__main__':
    # read image and label of corrected large tif file
    # water
    # selected_class = np.array([255, 255, 0], dtype=np.uint8)
    # grvel
    data_type = 'png'
    change_max_lbl_to_5 = True

    if change_max_lbl_to_5:
        print('\n\n 000000000000000000000000000000 \n\n')
        print('\n\n 000000000 change max 000000000 \n\n')
        print('\n\n 000000000000000000000000000000 \n\n')


    # selected_class = np.array([255, 0, 0], dtype=np.uint8) # Gravel
    selected_class = np.array([255, 255, 0], dtype=np.uint8) # Water
    rotation_angle_list = [angle for angle in range(0, 330, 30)]
    small_img_size = 2448
    selection_lim = 1000
    # if the colors have the most distance with each other
    most_distant = True

    base_path = '/data/nips/EX3'
    save_base_path = f'/data/nips/EX3/ML_G_{small_img_size}'

    # river_name_list = ['lærdal_1976_connected', 'gaula_1963_connected',
    #                    'nea_1962_connected', 'surna_1963_connected']

    river_name_list = [
        # 'gaula_1963_connected',
        # 'nea_1962_connected',
        # 'lærdal_1976_connected',
        # 'gaula_1998',
        # 'orkla_1962_test',
        # 'gaula_1947',
        'all_train_png',
        # 'surna_1963_connected',
    ]

    for river_name in river_name_list:
        start = time.time()
        print(f'\nworking on river {river_name}')

        # make a list to store the information of the river
        river_csv_list = []

        # get the path to image and labels
        river_path = os.path.join(base_path, river_name)

        if data_type == 'tif':
            tif_img_path_list = glob.glob(
                os.path.join(river_path, 'image', '*.tif'))
            tif_lbl_path_list = glob.glob(
                os.path.join(river_path, 'label', '*.tif'))
            print(f'there are {len(tif_img_path_list)} images in {river_path}')

        elif data_type == 'png':
            tif_img_path_list = glob.glob(
                os.path.join(river_path, 'image', '*.png'))
            tif_lbl_path_list = glob.glob(
                os.path.join(river_path, 'label', '*.png'))
            print(f'there are {len(tif_img_path_list)} images in {river_path}')
        
        # save directory
        # save_river_path = os.path.join(save_base_path, river_name)
        save_river_path = save_base_path
        save_img_path = os.path.join(save_river_path, 'image', 'img')
        save_lbl_path = os.path.join(save_river_path, 'label', 'img')
        os.makedirs(save_img_path, exist_ok=True)
        os.makedirs(save_lbl_path, exist_ok=True)

        # for each image in the river
        for img_idx in range(len(tif_lbl_path_list)):
            end = time.time()

            large_image_name = os.path.split(
                tif_lbl_path_list[img_idx])[-1].replace('.tif', '')
            print(
                f'\nworking on image {img_idx} name: {large_image_name} the time takes {end - start}')

            # read the image
            if data_type == 'tif':
                tif_img_arr = gdal_utils.read_tiff_file(
                    tif_img_path_list[img_idx])[0]
                tif_lbl_arr = gdal_utils.read_tiff_file(
                    tif_lbl_path_list[img_idx])[0]
            
            if data_type == 'png':
                tif_img_arr = image_processing_utils.read_png_file(
                    tif_img_path_list[img_idx])
                tif_lbl_arr = image_processing_utils.read_png_file(
                    tif_lbl_path_list[img_idx])

            
            if change_max_lbl_to_5:
                tif_lbl_arr[tif_lbl_arr == tif_lbl_arr.max()] = 5

            print(f'the shape of label: {tif_lbl_arr.shape}. max:{np.unique(tif_lbl_arr, return_counts=True)}')


            # transfer the label to rgb
            tif_lbl_arr_rgb = image_processing_utils.label_to_rgb(
                tif_lbl_arr.squeeze(), unknown_zero_flag=True, most_distant=most_distant)

            saved_image_count = 0
            # select the angle to start the operation
            for rotation_angle in rotation_angle_list:
                image_list = None
                label_list = None

                image_list, label_list, csv_list, _, _ = get_augment_rotate_double_colored(
                    image_matrix=tif_img_arr, label_matrix_changed=tif_lbl_arr_rgb, label_matrix_changed_2d=tif_lbl_arr,
                    small_image_size=small_img_size, selection_lim=selection_lim, selected_class=selected_class,
                    angle=rotation_angle, large_image_file_name=large_image_name,
                    labeled_percentage=0.8, overlap_percentage=0.3, method='skimage',
                    rotation_noise=0, large_image=True, most_distant=True)

                end = time.time()
                print(f'saving  the time takes {end - start}')

                if image_list:
                    for save_idx in range(len(label_list)):

                        # print(f'img shape is: {image_list[save_idx].shape}, label shape is: {label_list[save_idx].shape}')

                        image_name = f'{river_name}_{large_image_name}_{saved_image_count}_g.png'
                        # save image
                        image_processing_utils.save_to_png(
                            img_array=np.squeeze(image_list[save_idx]), img_path=os.path.join(save_img_path, image_name))
                        # save label
                        image_processing_utils.save_to_png(
                            img_array=label_list[save_idx], img_path=os.path.join(save_lbl_path, image_name))
                        saved_image_count += 1
                        river_csv_list += csv_list

            # save csv file
            df = pd.DataFrame(river_csv_list, columns=[
                'index', 'image', 'angle', 'xcenter', 'ycenter']).set_index('index')
            df.to_csv(os.path.join(save_river_path, f'{river_name}.csv'))

    print('process eneded successfully')
