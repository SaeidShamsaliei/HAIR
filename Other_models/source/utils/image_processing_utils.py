from asyncio import base_events
import gdal
import imageio
import numpy as np
from osgeo import ogr
from osgeo import osr
import time
import glob
import os
import pandas as pd
import shutil

from skimage.segmentation import slic, quickshift
from sklearn.cluster import MeanShift, estimate_bandwidth

from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image

from scipy import ndimage, misc
from urllib3 import Retry

from .gdal_utils import read_tiff_file

"""
This file contains function to help with using classical image segmentation
methods.
"""
color_dict = {
    # color code : rgb
    # water is blue
    0: [0, 0, 1],
    # gravel is yellow
    1: [1, 1, 0],
    # vegetation is green
    2: [0, 1, 0],
    # farmland is pink
    3: [1, 0, 1],
    # construction is red
    4: [1, 0, 0],
    # unknown is white
    5: [1, 1, 1]
}


def swap_labels(arr, lbl1, lbl2):
    """
    swaps the labl1 and lbl2 in arr
    """
    arr1_idx = np.where(arr == lbl1)
    arr2_idx = np.where(arr == lbl2)
    arr[arr1_idx], arr[arr2_idx] = lbl2, lbl1
    return arr


def save_to_png(img_array, img_path):
    # save to file
    img_png = Image.fromarray(img_array.astype(np.uint8))
    img_png.save(img_path)


def read_png_file(image_path):
    """
    read a png file as numpy array
    """
    original_image_matrix = imageio.imread(image_path)

    if original_image_matrix.ndim > 2 and original_image_matrix.shape[-1] > 3:
        print(
            f'image has more than 3 channels, only first 3 channels are used there are {original_image_matrix.shape} channels')
        original_image_matrix = original_image_matrix[:, :, :3]
    return original_image_matrix


def slic_boundary(image, n_segment, compactness, sigma=0):
    """
    this function performs the slic algorithm and output the
    boundary of the superpixels

    arguments
    image
    n_segment
    compactness
    sigma: for performing gaussian filter beforehand
    """
    segment = slic(image, compactness=compactness, n_segments=n_segment,
                   sigma=sigma)
    return find_boundaries(segment)


def qshitf_boundary(image, ratio):
    """
    this function performs the qshift algorithm and output the
    boundary of the clusters

    arguments
    image
    ratio
    sigma: for performing gaussian filter beforehand
    """
    # copy channels 3 times if needed
    if image.shape[-1] != 3:
        image = np.stack([image for _ in range(3)], axis=-1)
    segment = quickshift(image, ratio=ratio, sigma=5)
    return find_boundaries(segment)


def meanshift_image(image, quantile):
    """
    performs the meanshift on the images, returns the clusters
    """

    if image.ndim == 2:
        shape = image.shape
    elif image.ndim == 3:
        shape = image.shape[:2]
    else:
        raise ValueError(
            f"number of dimensions must be 2 or 3 not `{image.ndim}`")

    flat = image.reshape((-1, 1))
    bandwidth = estimate_bandwidth(flat,
                                   quantile=quantile, n_samples=100)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(flat)
    return model.labels_.reshape(shape)


def sobel_filter(img, sigma=5):
    """
    performs sobel after removed the noise using gaussian filter
    sigma indicates the std of gaussian filter
    """
    # remove noise
    img = ndimage.gaussian_filter(img, sigma=sigma)
    # apply laplacian
    result = ndimage.sobel(img)

    return result


def laplacian_filter(img, sigma=5):
    """
    performs laplacian after removed the noise using gaussian filter
    """
    # remove noise
    img = ndimage.gaussian_filter(img, sigma=sigma)
    # apply laplacian
    result = ndimage.laplace(img)

    return result


def show_label(label_elm):
    return color_dict[label_elm]


def show_label(label):
    label_color = np.zeros((label.shape[0], label.shape[1], 3))
    for idx in range(label_color.shape[0]):
        for jdx in range(label_color.shape[1]):
            label_color[idx][jdx][0] = color_dict[label[idx][jdx]][0]
            label_color[idx][jdx][1] = color_dict[label[idx][jdx]][1]
            label_color[idx][jdx][2] = color_dict[label[idx][jdx]][2]
    return label_color


def show_segment_label(img, seg_label, label):
    plt.figure(figsize=(20, 20))
    plt.subplot('121')
    plt.imshow(segmentation.mark_boundaries(img, seg_label))
    plt.subplot('122')
    plt.imshow(label)
    plt.show()


# show image and label by path
def load_image_label(image_path_val, label_path_val):
    img = imageio.imread(image_path_val).astype(np.float32) / 255
    img = np.log(1 + img)  # Slight increase of contrast
    # img = (img * 255).astype(np.uint8)
    label = imageio.imread(label_path_val).astype(np.uint8)
    return img, label


# gives us rgb for each label
def show_label_elm(label_elm, index):
    return np.array(color_dict[label_elm])[index]


# get the rgb array for label
def get_colored_label(label, max_dim=3):
    # vectorizing the function
    show_vec = np.vectorize(show_label_elm)
    label_color_arr = np.zeros((*label.shape, 3))
    for i in range(max_dim):
        label_color_arr[..., i] = show_vec(label, i)
    return label_color_arr


# replace the colors that are not in the color matrix with the closest color
def color_nearest_neighbors(img_matrix, clr_matrix):
    """
    find and replace colors that are in img_matrix and not in clr_matrix and replace them with the closest color in the clr_mtrix.
    """
    from einops import rearrange

    if img_matrix.ndim == 3:
        # get the original shape
        W_val, H_val, C_val = img_matrix.shape
        img_matrix = rearrange(img_matrix, 'w h c -> (w h) c')
    else:
        raise ValueError('img_matrix should have 3 dimentions, W, H, C')

    near_colored_img_matrix = clr_matrix[np.linalg.norm(
        img_matrix[None, :] - clr_matrix[:, None], axis=-1).argmin(axis=0)]
    return rearrange(near_colored_img_matrix, '(w h) c -> w h c', w=W_val, h=H_val)

# replace the colors that are not in the color matrix with the closest color


def color_nearest_neighbors_large(img_matrix, clr_matrix):
    """
    find and replace colors that are in img_matrix and not in clr_matrix and replace them with the closest color in the clr_mtrix.
    it devides image into 4 smaller one and connect them together
    """
    from einops import rearrange

    if img_matrix.ndim == 3:
        # get the original shape
        near_colored_img_matrix = np.zeros(img_matrix.shape)
        W_val, H_val, C_val = img_matrix.shape
        W_val_small, H_val_small = int(W_val/2), int(H_val/2)

        img_matrix_flatten1 = rearrange(
            img_matrix[:W_val_small, :H_val_small, :], 'w h c -> (w h) c')
        img_matrix_flatten2 = rearrange(
            img_matrix[W_val_small:, :H_val_small, :], 'w h c -> (w h) c')
        img_matrix_flatten3 = rearrange(
            img_matrix[:W_val_small, H_val_small:, :], 'w h c -> (w h) c')
        img_matrix_flatten4 = rearrange(
            img_matrix[W_val_small:, H_val_small:, :], 'w h c -> (w h) c')

    else:
        raise ValueError('img_matrix should have 3 dimentions, W, H, C')

    near_colored_img_matrix[:W_val_small, :H_val_small, :] = rearrange(clr_matrix[np.linalg.norm(
        img_matrix_flatten1[None, :] - clr_matrix[:, None], axis=-1).argmin(axis=0)], '(w h) c -> w h c', w=W_val_small, h=H_val_small)

    near_colored_img_matrix[W_val_small:, :H_val_small, :] = rearrange(clr_matrix[np.linalg.norm(
        img_matrix_flatten2[None, :] - clr_matrix[:, None], axis=-1).argmin(axis=0)], '(w h) c -> w h c', w=W_val_small, h=H_val_small)

    near_colored_img_matrix[:W_val_small, H_val_small:, :] = rearrange(clr_matrix[np.linalg.norm(
        img_matrix_flatten3[None, :] - clr_matrix[:, None], axis=-1).argmin(axis=0)], '(w h) c -> w h c', w=W_val_small, h=H_val_small)

    near_colored_img_matrix[W_val_small:, H_val_small:, :] = rearrange(clr_matrix[np.linalg.norm(
        img_matrix_flatten4[None, :] - clr_matrix[:, None], axis=-1).argmin(axis=0)], '(w h) c -> w h c', w=W_val_small, h=H_val_small)

    return near_colored_img_matrix

# Perform one hot encoding on label

# code from kaggle [https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-deeplabv3/notebook]


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def one_hot_decode(label):
    return np.argmax(label, axis=-1)


def label_to_rgb(label_2d, unknown_zero_flag=False, most_distant=False):
    """
    Convert a 2d lebel mask to rgb file, each color represents one class type
    most_distant: if set True then colors with most distance will be returned
    """
    # to color the segmentation mask
    if unknown_zero_flag:
        palette = np.array([[0, 0, 0],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [0, 128, 0],  # Green    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0, 255], ], dtype=np.uint8)  # Black  000000
    else:
        palette = np.array([[0, 0, 255],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [0, 128, 0],  # Green    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0,   0], ], dtype=np.uint8)  # Black  000000
    if most_distant:
        print('label to rgb most distance mode')
        palette = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
                            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)

    return palette[label_2d.astype(np.uint8)]


def median_filter(image, kernel_size):
    return ndimage.median_filter(image, size=kernel_size)


def replace_rgb_to_class(img_arr, unknown_zero_flag=False, most_distant=False):
    """
    Convers labels with RGB colors as value to number of class as value
    by default:
        0 : water
        5: unknown
    if unknown_zero_flag is set to ture:
        0 : unknown
        5: water
    """
    # list of valid colors
    color_matrix = np.array([[0, 0, 255], [255, 255, 0], [0, 128, 0], [
                            128, 128, 0], [128, 0, 128], [0, 0, 0]], dtype=np.uint8)
    if unknown_zero_flag:
        color_matrix = np.array([[0, 0, 0], [255, 255, 0], [0, 128, 0], [
                                128, 128, 0], [128, 0, 128], [0, 0, 255]], dtype=np.uint8)
    if most_distant:
        color_matrix = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)
    # convert image into 2d
    replaced_image_onehot = one_hot_encode(
        img_arr.astype(np.uint8), color_matrix)
    return one_hot_decode(replaced_image_onehot)


def rotate_image_scipy(img, angle):
    """
    rotates image "img" by angle, "angle"
    """
    from scipy import ndimage
    return ndimage.rotate(img, angle, reshape=False)


def rotate_image_skimage(img, angle):
    """
    rotates image "img" by angle, "angle" using skimage
    """
    from skimage.transform import rotate as sk_rotate
    return sk_rotate(img, angle, preserve_range=True).astype(np.uint8)

# just devide Normally


def most_selected_class(image_matrix, select_class, threshold):
    """
    True if select_class is more than threshold
    """
    # Check that the amount of the unknown class
    class_mask = image_matrix == select_class

    if np.sum(class_mask) > threshold * class_mask.size:
        return True
    else:
        return False


def make_small_image_index(image_matrix, label_matrix=None, image_size=512, do_overlap=False):
    """
    indecies to devide images into small images with image size of (image_size x image_size)
    """
    if do_overlap:
        shape_0_indices = list(
            range(image_size // 4, image_matrix.shape[0], image_size // 4))[:-4]
        shape_1_indices = list(
            range(image_size // 4, image_matrix.shape[1], image_size // 4))[:-4]
    else:
        shape_0_indices = list(range(0, image_matrix.shape[0], image_size))
        shape_0_indices[-1] = image_matrix.shape[0] - image_size
        shape_1_indices = list(range(0, image_matrix.shape[1], image_size))
        shape_1_indices[-1] = image_matrix.shape[1] - image_size

    return shape_0_indices, shape_1_indices


def swap_two_values_non_vectorized(element, val1=5, val2=0):
    if element == val1:
        return val2
    elif element == val2:
        return val1
    else:
        return element


def swap_value_np(array, val1, val2):
    vectorised_swap = np.vectorize(swap_two_values_non_vectorized)
    return vectorised_swap(element=array, val1=val1, val2=val2)


def count_class_list_directory_3d(src_label_path, unknown_zero, verbose=0, most_distant=False):
    """ returns list of vectors in which number of classes of one image is mentioned (RGB)"""
    class_count_list = []
    start = time.time()
    if unknown_zero == True:
        color_matrix = np.array([[0, 0, 0], [255, 255, 0], [0, 128, 0], [
            128, 128, 0], [128, 0, 128], [0, 0, 255]], dtype=np.uint8)
    else:
        color_matrix = np.array([[0, 0, 255], [255, 255, 0], [0, 128, 0], [
            128, 128, 0], [128, 0, 128], [0, 0, 0]], dtype=np.uint8)
    if most_distant:
        color_matrix = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)

    for label_path in glob.glob(os.path.join(src_label_path, '*.png')):
        if verbose == 1:
            print(f'working on label {os.path.split(label_path)[-1]}')
        # read label
        original_label_matrix = imageio.imread(label_path)

        if original_label_matrix.shape[-1] == 3:
            # it is a 3 channel label so we can calculate the class info
            # convert image into 2d
            replaced_image_onehot = one_hot_encode(
                original_label_matrix.astype(np.uint8), color_matrix)
            replaced_image_2d = one_hot_decode(
                replaced_image_onehot)
            # get the information about the classes
            (unique_tmp, counts_tmp) = np.unique(
                replaced_image_2d, return_counts=True)
            class_count_vec = np.zeros(6)
            class_count_vec[unique_tmp] = counts_tmp

            class_count_list.append(class_count_vec)
        else:
            raise NotImplementedError('it is not 2d')

    if verbose == 1:
        end = time.time()
        print(f'the time takes {end - start}')
    return class_count_list


def count_class_list_directory_2d(src_label_path, verbose=0):
    """ returns list of vectors in which number of classes of one image is mentioned (2d)"""
    class_count_list = []
    start = time.time()
    for label_path in glob.glob(os.path.join(src_label_path, '*.png')):
        if verbose == 1:
            print(f'working on label {os.path.split(label_path)[-1]}')

        # read label
        original_label_matrix = imageio.imread(label_path)

        if original_label_matrix.shape[-1] != 3:
            # it is not a 3 channel label so we can calculate the class info
            # get the information about the classes
            (unique_tmp, counts_tmp) = np.unique(
                original_label_matrix, return_counts=True)
            class_count_vec = np.zeros(6)
            class_count_vec[unique_tmp] = counts_tmp
            class_count_list.append(class_count_vec)
        else:
            raise NotImplementedError('it is not 2d')

    end = time.time()
    if verbose == 1:
        print(f'the time takes {end - start}')
    return class_count_list

# list of valid colors


def single_rgb_to_label(rgb_val, unknown_zero_flag=True, most_distant=False):
    """
    input rgb and outputs the corresponding label E.g input = [128, 0, 128] and output 4
    """
    assert rgb_val.ndim == 1 and rgb_val.shape[0] == 3
    color_matrix = np.array([[0, 0, 255], [255, 255, 0], [0, 128, 0], [
        128, 128, 0], [128, 0, 128], [0, 0, 0]], dtype=np.uint8)
    if unknown_zero_flag:
        color_matrix = np.array([[0, 0, 0], [255, 255, 0], [0, 128, 0], [
            128, 128, 0], [128, 0, 128], [0, 0, 255]], dtype=np.uint8)
    if most_distant:
        color_matrix = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)
    return np.where(np.all(color_matrix == rgb_val, axis=1))[0][0]

# image_matrix, label_matrix, geo_transform, projection = get_images(image_filepath, label_filepath)


def rotate_scipy(angle, img, lbl):
    from scipy import ndimage
    return ndimage.rotate(img, angle, reshape=False), ndimage.rotate(lbl, angle, reshape=False)


def rotate_skimage(sk_angel, img, lbl):
    from skimage.transform import rotate as sk_rotate
    return sk_rotate(img, sk_angel, preserve_range=True).astype(np.uint8), sk_rotate(lbl, sk_angel, preserve_range=True).astype(np.uint8)


def replace_labels_to_rgb_directory(lbl_path, dest_path):
    all_lbl = glob.glob(os.path.join(lbl_path, '*.png'))
    dest_path_lbl = os.path.join(dest_path, 'img')

    os.makedirs(dest_path_lbl, exist_ok=True)

    for idx, _ in enumerate(all_lbl):
        name = os.path.split(all_lbl[idx])[-1]

        lbl_2d = read_png_file(all_lbl[idx])
        lbl_3d = label_to_rgb(label_2d=lbl_2d,
                              unknown_zero_flag=True)
        save_to_png(img_array=lbl_3d,
                    img_path=os.path.join(dest_path_lbl, name))

        if idx % 1000 == 0:
            print(idx)


def write_name_to_txt(image_path, label_path, base_save, data_type='train'):

    all_image = glob.glob(os.path.join(image_path, '*.png'))
    all_label = glob.glob(os.path.join(label_path, '*.png'))
    img_lbl_list = []
    print(base_save)
    for img_idx, _ in enumerate(all_image):
        assert os.path.split(
            all_image[img_idx])[-1] == os.path.split(all_label[img_idx])[-1]
        current_pair = f'{all_image[img_idx].replace(f"{base_save}/", "").replace("img/", "")}\t{all_label[img_idx].replace(f"{base_save}/", "").replace("label_rgb","label").replace("img/", "")}\n'
        img_lbl_list.append(current_pair)

    if data_type == 'train':
        f = open(os.path.join(base_save, 'train.txt'), "w")
        f.writelines(img_lbl_list)
        f.close()
    elif data_type == 'val':
        f = open(os.path.join(base_save, 'val.txt'), "w")
        f.writelines(img_lbl_list)
        f.close()
    elif data_type == 'test':
        f = open(os.path.join(base_save, 'test.txt'), "w")
        f.writelines(img_lbl_list)
        f.close()
    else:
        raise NotImplementedError


def make_confusion_mask_pred(label, pred, class_id):
    # output is the prediction of areas where label indicates to be class_id

    # get the one hot vector of label
    label_one_hot_ecoded = one_hot_encode(
        label=label, label_values=[0, 1, 2, 3, 4, 5])
    if pred.ndim > label_one_hot_ecoded[:, :, class_id].ndim:
        masked_pred = pred.squeeze() * label_one_hot_ecoded[:, :, class_id]
    else:
        masked_pred = pred * label_one_hot_ecoded[:, :, class_id]
    return masked_pred


def color_masked_pred(label_2d, class_id, unknown_zero_flag=True, most_distant=False):
    # colors the prediction (the class id is light green and the rest is the same as others)
    # to color the segmentation mask
    if unknown_zero_flag:
        palette = np.array([[0, 0, 0],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [255, 0, 0],  # Red    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0, 255], ], dtype=np.uint8)  # Black  000000
    else:
        palette = np.array([[0, 0, 255],  # Blue     0000FF
                            [255, 255, 0],  # Yellow   FFFF00
                            [0, 128, 0],  # Green    008000
                            [128, 128, 0],  # Olive    808000
                            [128, 0, 128],  # Purple   800080
                            [0,   0,   0], ], dtype=np.uint8)  # Black  000000
    if most_distant:
        print('label to rgb most distance mode')
        palette = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255], [
                            0, 255, 255], [0, 0, 255], [255, 255, 0]],  dtype=np.uint8)

    palette[class_id] = np.array([0, 255, 127])
    return palette[label_2d.astype(np.uint8)]


def unique_values_axis(arr, axis=0):
    """
    input an array (C H W) and returns a 2D array each value corresponds to number of unique elements across axis=axis 
    """
    B = arr.copy()
    B.sort(axis=axis)
    C = np.diff(B, axis=axis) > 0
    D = C.sum(axis=axis)+1
    return D


def dataset_split_train_vlidation_test_notebook(src_path, dst_path,
                                                train_rate=0.7,
                                                val_rate=0.2):
    """
    function to split train validation and test based on the rate
    the source should be saved in /src_path/image or label/*.png

    it will be stored as /dst_path/image/img
    """
    src_imgs_list = glob.glob(os.path.join(src_path, 'image', 'img','*.png'))
    src_lbls_list = glob.glob(os.path.join(src_path, 'label', 'img','*.png'))
    print(f'there are {len(src_imgs_list)} images')
    if len(src_imgs_list) != len(src_lbls_list):
        raise ValueError('len of image and label are not the same')

    index = np.arange(len(src_imgs_list))
    np.random.shuffle(index)
    train_end = round(train_rate * len(src_imgs_list))
    val_end = round((train_rate+val_rate) * len(src_imgs_list))

    train_img_paths = [src_imgs_list[idx] for idx in index[:train_end]]
    train_lbl_paths = [src_lbls_list[idx] for idx in index[:train_end]]

    val_img_paths = [src_imgs_list[idx] for idx in index[train_end:val_end]]
    val_lbl_paths = [src_lbls_list[idx] for idx in index[train_end:val_end]]

    test_img_paths = [src_imgs_list[idx] for idx in index[val_end:]]
    test_lbl_paths = [src_lbls_list[idx] for idx in index[val_end:]]

    # check if train image and labels are the same
    dest_img_path = os.path.join(dst_path, 'trian', 'image', 'img')
    dest_lbl_path = os.path.join(dst_path, 'trian', 'label', 'img')
    os.makedirs(dest_img_path, exist_ok=True)
    os.makedirs(dest_lbl_path, exist_ok=True)
    for idx in range(len(train_img_paths)):
        if os.path.split(train_img_paths[idx])[-1] != os.path.split(train_lbl_paths[idx])[-1]:
            print(
                f'{os.path.split(train_img_paths[idx])[-1]} != {os.path.split(train_lbl_paths[idx])[-1]}')
        file_name = os.path.split(train_img_paths[idx])[-1]
        # move the image
        shutil.copyfile(src=train_img_paths[idx], dst=os.path.join(
            dest_img_path, file_name))
        # move the label
        shutil.copyfile(src=train_lbl_paths[idx], dst=os.path.join(
            dest_lbl_path, file_name))

    # check if val image and labels are the same
    dest_img_path = os.path.join(dst_path, 'val', 'image', 'img')
    dest_lbl_path = os.path.join(dst_path, 'val', 'label', 'img')
    os.makedirs(dest_img_path, exist_ok=True)
    os.makedirs(dest_lbl_path, exist_ok=True)
    for idx in range(len(val_img_paths)):
        if os.path.split(val_img_paths[idx])[-1] != os.path.split(val_lbl_paths[idx])[-1]:
            print(
                f'{os.path.split(val_img_paths[idx])[-1]} != {os.path.split(val_lbl_paths[idx])[-1]}')
        file_name = os.path.split(val_img_paths[idx])[-1]
        # move the image
        shutil.copyfile(src=val_img_paths[idx], dst=os.path.join(
            dest_img_path, file_name))
        # move the label
        shutil.copyfile(src=val_lbl_paths[idx], dst=os.path.join(
            dest_lbl_path, file_name))

    # check if test image and labels are the same
    dest_img_path = os.path.join(dst_path, 'test', 'image', 'img')
    dest_lbl_path = os.path.join(dst_path, 'test', 'label', 'img')
    os.makedirs(dest_img_path, exist_ok=True)
    os.makedirs(dest_lbl_path, exist_ok=True)
    for idx in range(len(test_img_paths)):
        if os.path.split(test_img_paths[idx])[-1] != os.path.split(test_lbl_paths[idx])[-1]:
            print(
                f'{os.path.split(test_img_paths[idx])[-1]} != {os.path.split(test_lbl_paths[idx])[-1]}')
        file_name = os.path.split(test_img_paths[idx])[-1]
        # move the image
        shutil.copyfile(src=test_img_paths[idx], dst=os.path.join(
            dest_img_path, file_name))
        # move the label
        shutil.copyfile(src=test_lbl_paths[idx], dst=os.path.join(
            dest_lbl_path, file_name))


def class_count_list_dataframe(class_count_list, unknown_zero):
    class_df = pd.DataFrame(class_count_list, dtype='float32')

    class_df.loc['class_sum'] = class_df.sum(axis=0)

    if unknown_zero == True:
        class_df = class_df.rename(columns={
                                   5: "water", 1: "gravel", 2: "vegetation", 3: "farmland", 4: "human construction", 0: "unknown"})
    else:
        class_df = class_df.rename(columns={
                                   0: "water", 1: "gravel", 2: "vegetation", 3: "farmland", 4: "human construction", 5: "unknown"})

    class_df.loc['class_sum_percent'] = (
        (class_df.loc['class_sum'] / class_df.loc['class_sum'].sum()) * 100)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    return class_df


def change2dto3d(input_type, lbl_path, lbl_new_path, verbose=1):
    """
    change 2d label to 3d, output is png 
     input_type: png or tif
     lbl_path: is the path to the labels
     lbl_new_path: place to save the labels

    """
    os.makedirs(lbl_new_path, exist_ok=True)

    label_list = glob.glob(os.path.join(lbl_path, f'*.{input_type}'))
    for idx, label_path in enumerate(label_list):
        label_name = os.path.split(label_path)[-1]
        if input_type == 'png':
            lbl_arr = read_png_file(label_path)

        elif input_type == 'tif':
            lbl_arr = read_tiff_file(large_image_path=label_path,
                                     normalize=False,
                                     zeropadsize=None,
                                     numpy_array_only=True,
                                     grayscale_only=False)
        else:
            print('the input_type is not correct')

        lbl_arr = label_to_rgb(label_2d=lbl_arr,
                               unknown_zero_flag=True,
                               most_distant=False)
        lbl_arr = np.squeeze(lbl_arr)

        save_to_png(
            img_array=lbl_arr,
            img_path=os.path.join(
                lbl_new_path, label_name.replace('tif', 'png')),
        )
        if idx % 100 == 0 and verbose == 1:
            print(idx)


def devide_small_image(large_img_arr, large_lbl_arr_2d,
                       image_size, dst_path_img, dst_path_lbl,
                       unknown_class, large_image_name,
                       ):
    """
    devide images and array into image_size and save them into dst_path
    """
    if large_img_arr.shape != large_lbl_arr_2d.shape:
        raise ValueError(
            f'shape of image and labels should be the same {large_img_arr.shape} != {large_lbl_arr_2d.shape}')

    # get indecies of small images
    list_0_idx, list_1_idx = make_small_image_index(large_lbl_arr_2d,
                                                    image_size=image_size, do_overlap=False)

    for i_idx in list_0_idx:
        for j_idx in list_1_idx:
            small_label = large_lbl_arr_2d[i_idx: i_idx+image_size,
                                           j_idx: j_idx+image_size]

            small_image = large_img_arr[i_idx: i_idx+image_size,
                                        j_idx: j_idx+image_size]

            # if more than 50% of class is unknown it will not be included
            if most_selected_class(image_matrix=small_image,
                                   select_class=unknown_class, threshold=0.5):
                #                 print(all_image_names'more than half is unknown so ignored')
                continue

            # image name
            tmp_img_name = f"{large_image_name.replace('.png', '')}_{i_idx}_{j_idx}.png"

            save_to_png(img_array=small_label,
                        img_path=os.path.join(dst_path_lbl, tmp_img_name))

            save_to_png(img_array=small_image,
                        img_path=os.path.join(dst_path_img, tmp_img_name))
    return True


def tif_to_png(tif_dir, dest_dir,
               perform_augmentation=False):
    """
    convert tif files to png
    param:tif_dir path to the tif file
    param:dest_dir destination path to save the png
    files.
    """

    # add output directory if not exists
    os.makedirs(dest_dir, exist_ok=True)
    tif_paths = glob.glob(os.path.join(tif_dir, "*.tif"))
    print(f'there are {len(tif_paths)} images')
    for idx, tif_path in enumerate(tif_paths):
        if idx % 1000 == 0:
            print(f'image {idx}')

        ds = gdal.Open(tif_path)
        band_arrays = []
        for band in range(ds.RasterCount):
            band_array = ds.GetRasterBand(band + 1).ReadAsArray()
            band_array = np.expand_dims(band_array, axis=-1)
            band_arrays.append(band_array)
        if ds.RasterCount > 1:
            image_array = np.concatenate(
                band_arrays, axis=-1).astype(np.uint8)
        else:
            image_array = band_array.squeeze(axis=-1)

        # if we want to perform augmentation
        if perform_augmentation:
            raise NotImplemented
        else:
            im = Image.fromarray(image_array)
            im = im.convert("L")
            im.save(os.path.join(dest_dir, os.path.split(
                tif_path)[-1]).replace(".tif", ".png"))


def convert_png_to_jpg(src_img, dst_path):
    """
    saves the 'src_img' with the same name into the 'dst_path' 
    """

    image_name = os.path.split(src_img)[-1]
    im1 = Image.open(src_img)
    im1.save(os.path.join(dst_path, image_name.replace('.png', '.jpg')))


def convert_jpg_to_png(src_img, dst_path):
    """
    saves the 'src_img' with the same name into the 'dst_path' 
    """

    image_name = os.path.split(src_img)[-1]
    im1 = Image.open(src_img)
    im1.save(os.path.join(dst_path, image_name.replace('.jpg', '.png')))


def move_all_files_to_dest(src_path_funct, dst_path_func, file_type='png', verbose=0):
    """
    move all the files with 'file_type format to dst_path with the same name'
    """
    os.makedirs(dst_path_func, exist_ok=True)
    src_files = glob.glob(os.path.join(src_path_funct, f"*.{file_type}"))
    for src_file_path in src_files:
        name = os.path.split(src_file_path)[-1]
        dst_file_path = os.path.join(dst_path_func, name)
        if verbose == 1:
            print(f'moving {src_file_path} to {dst_file_path}')
        shutil.copyfile(src=src_file_path, dst=dst_file_path)
