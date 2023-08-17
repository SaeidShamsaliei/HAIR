import numpy as np
import matplotlib.pyplot as plt
import subprocess
import glob
import os
import osr
import glob
import os
import sys
from osgeo import gdal, ogr
from tensorflow.python.keras.backend import dtype

# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

"""
This file contains functions usefull for gdal library
"""


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Function to pad a vector from numpy.org
    """
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def read_tiff_file(large_image_path, normalize=False, zeropadsize=None, numpy_array_only=False, grayscale_only=False):
    """
    Outputs the numpy array and if specified geo_transform and projection
    :param large_image_path:
    :param normalize:
    :param zeropadsize:
    :param numpy_array_only: indicates if only numpy array is returned or all the info
    :return:
    """
    # Load image
    image_ds = gdal.Open(large_image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = np.array(
        ([image_ds.GetRasterBand(band_idx + 1).ReadAsArray() for band_idx in range(image_ds.RasterCount)]), dtype=np.uint8)
    if len(image_matrix.shape) > 2:
        # get (x,y, channel) form
        image_matrix = np.transpose(image_matrix, axes=[1, 2, 0])
    if grayscale_only:
        # get black and white
        from PIL import Image
        image_matrix = np.array(Image.fromarray(
            image_matrix.astype(np.uint8)))
    image_ds = None

    if normalize:
        image_matrix = image_matrix / 255
    if zeropadsize is not None:
        image_matrix = np.pad(image_matrix, 256, pad_with)

    # return only the numpy array if specified
    if numpy_array_only:
        return image_matrix
    else:
        return image_matrix, geo_transform, projection


def get_numpy_array_large_image(large_image_path, normalize=False, zeropadsize=None, numpy_array_only=True):
    """
    Outputs the numpy array and if specified geo_transform and projection
    :param large_image_path:
    :param normalize:
    :param zeropadsize:
    :param numpy_array_only: indicates if only numpy array is returned or all the info
    :return:
    """
    # Load image
    image_ds = gdal.Open(large_image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = np.array(
        ([image_ds.GetRasterBand(band_idx + 1).ReadAsArray() for band_idx in range(image_ds.RasterCount)]))
    if len(image_matrix.shape) > 2:
        # get (x,y, channel) form
        image_matrix = np.transpose(image_matrix, axes=[1, 2, 0])
    # get black and white
    from PIL import Image
    image_matrix = np.array(Image.fromarray(image_matrix).convert('L'))
    image_ds = None

    if normalize:
        image_matrix = image_matrix / 255
    if zeropadsize is not None:
        image_matrix = np.pad(image_matrix, 256, pad_with)

    # return only the numpy array if specified
    if numpy_array_only:
        return image_matrix
    else:
        return image_matrix, geo_transform, projection


# To merge all the tif files in one directory into one big tif file
def merge_all_rasters_directory(input_directory, output_name, output_directory=None, save_tif=True):
    """
    params:
    input_directory: path to input folder which should contains .tif files
    output_name: the name of output (should be without .tif or .vrt)
    output_directory: path to save the output files
    save_tif: save the all the images in one big tif file
    """
    # print('output directory is {output_directory}')
    files_to_mosaic = glob.glob(os.path.join(input_directory, '*.tif'))
    # build virtual raster and convert to geotiff
    if output_directory is not None:
        output_name = os.path.join(output_directory, output_name)
    print('before vrt')
    vrt = gdal.BuildVRT(f'{output_name}.vrt', files_to_mosaic)
    print('after vrt')
    if save_tif:
        gdal.Translate(f'{output_name}.tif', vrt)
        print(f'after saving {output_name}.tif')
    vrt = None


def merge_all_rasters_list(tif_file_list, output_name, output_directory, save_tif=True):
    """
    merge all the tif file paths of which are in tif_file_list
    tif_file_list: list which contains .tif files
    output_name: the name of output (should be without .tif or .vrt)
    output_directory: path to save the output files
    save_tif: save the all the images in one big tif file
    """
    if output_directory is not None:
        output_name = os.path.join(output_directory, output_name)
    vrt = gdal.BuildVRT(f'{output_name}.vrt', tif_file_list)
    if save_tif:
        gdal.Translate(f'{output_name}.tif', vrt)
    vrt = None


# To make polygon out of the raster file
def raster_band_to_polygon(raster_band, output_name, raster_proj_val):
    """
    save the polygon to a shape file
    """
    # prepare the gdal
    driver = ogr.GetDriverByName("ESRI Shapefile")
    outDatasource = driver.CreateDataSource(output_name)
    outLayer = outDatasource.CreateLayer('polygonized', srs=raster_proj_val)
    newField = ogr.FieldDefn(str(1), ogr.OFTInteger)
    outLayer.CreateField(newField)
    # save the polygon to the output path
    gdal.Polygonize(raster_band, None, outLayer, 0, [], callback=None)
    outDatasource.Destroy()
    sourceRaster = None
    print('done')


def raster_file_to_polygon(raster_path, output_path, band_number=1):
    # read the raster and get the band 1
    sourceRaster = gdal.Open(raster_path)
    sr_proj = sourceRaster.GetProjection()
    raster_proj = osr.SpatialReference()
    raster_proj.ImportFromWkt(sr_proj)
    band = sourceRaster.GetRasterBand(band_number)

    # write it to polygon
    raster_band_to_polygon(
        raster_band=band, output_name=output_path, raster_proj_val=raster_proj)


def convert_tif_to_shape_directory(input_folder_val, file_name=None):
    """
    read all the raster data and convert it to shapefile in a directory
    params:
    file_name: if there is a specific file that needs to be polygonize we use this param should end with .tif otherwise None
    """
    big_image_paths = glob.glob(os.path.join(input_folder_val, "*.tif"))
    # make a new folder for shape file
    if big_image_paths and not os.path.exists(os.path.join(input_folder_val, 'shapes')):
        os.mkdir(os.path.join(input_folder_val, 'shapes'))
    # loop through all the images in the file
    for tif_image in big_image_paths:
        if tif_image.endswith('.tif'):
            if not (file_name == None or tif_image.endswith(file_name)):
                continue
            poly_image = tif_image.replace('.tif', '.shp')
            # change the address to add shape directory
            poly_image = os.path.join(
                *os.path.split(poly_image)[:-1], 'shapes', os.path.split(poly_image)[-1])
            print(poly_image)
            if os.path.exists(poly_image):
                answer = input(
                    f'{os.path.split(poly_image)[-1]} already exist do you want to delete it (y/n)?')
                if answer == 'y':
                    temp_driver = ogr.GetDriverByName("ESRI Shapefile")
                    temp_driver.DeleteDataSource(poly_image)
                else:
                    continue
            print(f'working on {os.path.split(tif_image)[-1]}')
            raster_file_to_polygon(raster_path=tif_image,
                                   output_path=poly_image)

# clip the rasters based using a shapefile as mask


def clip_raster_using_polygon(rasin, shpin, rasout):
    result = gdal.Warp(srcDSOrSrcDSTab=rasin,
                       destNameOrDestDS=rasout, cutlineDSName=shpin, cropToCutline=True, dstNodata=-9999)
    return result
