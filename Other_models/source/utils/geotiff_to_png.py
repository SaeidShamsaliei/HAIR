import faulthandler; faulthandler.enable()
import os
import gdal
import numpy as np
import sys
import glob
from PIL import Image
import sys

sys.path.insert(1, '/workspace/code/river-segmentation/source')

from model_utils import replace_class

"""
Convert geotiffs to pngs. The georeferences will be lost. 
"""

def single_image_augmentation(data):
    """
    Takes the original image matrix and add rotated images and mirrored images (with rotations).
    This adds 11 additional images for each original image.
    :param data:
    :return: An numpy array with the augmented images concatenated to the data array
    """
    rot_90 = np.rot90(data, axes=(0, 1))
    rot_180 = np.rot90(data, k=2, axes=(0, 1))
    rot_270 = np.rot90(data, k=3, axes=(0, 1))
    mirror = np.flip(data, axis=0)
    mirror_rot_90 = np.rot90(mirror, axes=(0, 1))
    mirror_rot_180 = np.rot90(mirror, k=2, axes=(0, 1))
    mirror_rot_270 = np.rot90(mirror, k=3, axes=(0, 1))
    mirror2 = np.flip(data, axis=1)
    mirror2_rot_90 = np.rot90(mirror2, axes=(0, 1))
    mirror2_rot_180 = np.rot90(mirror2, k=2, axes=(0, 1))
    mirror2_rot_270 = np.rot90(mirror2, k=3, axes=(0, 1))
    augments = [data, rot_90, rot_180, rot_270, mirror, mirror_rot_90, mirror_rot_180,
                mirror_rot_270, mirror2, mirror2_rot_90, mirror2_rot_180, mirror2_rot_270]
    augmented_image_matrix = np.stack(augments)

    return augmented_image_matrix

if __name__ == '__main__':
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    label_flag = sys.argv[3]

    tif_paths = glob.glob(os.path.join(source_dir, "*.tif"))
    counter = 0
    for tif_path in tif_paths:
        ds = gdal.Open(tif_path)
        band_arrays = []
        for band in range(ds.RasterCount):
            band_array = ds.GetRasterBand(band + 1).ReadAsArray()
            band_array = np.expand_dims(band_array, axis=-1)
            band_arrays.append(band_array)
        if ds.RasterCount > 1:
            image_array = np.concatenate(band_arrays, axis=-1).astype(np.uint8)
        else:
            image_array = band_array.squeeze(axis=-1)

        if label_flag == '1':
            image_array = replace_class(image_array, class_id=5)
            # im = Image.fromarray(image_array)
            # im.save(os.path.join(dest_dir, os.path.split(tif_path)[-1]).replace(".tif", ".png"))

        # elif label_flag == '2':
        print(f'image {counter}')
        counter += 1
        image_array = single_image_augmentation(image_array)

        for idx in range(image_array.shape[0]):
            im = Image.fromarray(image_array[idx])
            im.save(os.path.join(dest_dir, os.path.split(tif_path)[-1]).replace(".tif", f"_aug{idx}.png"))
            # print(f'{(os.path.split(tif_path)[-1]).replace(".tif", f"{idx}.png")} saved')
