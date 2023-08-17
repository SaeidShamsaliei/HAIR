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

from PIL import Image


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    """Compute confusion matrix

    Args:
        x (np.array): 1 x h x w
            prediction array
        y (np.array): 1 x h x w
            groundtruth array
        n (int): number of classes
        ignore_label (int, optional): index of ignored label. Defaults to None.
        mask (np.array, optional): mask of regions that is needed to compute. Defaults to None.

    Returns:
        np.array: n x n
            confusion matrix
    """
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n)


def getIoU(conf_matrix):
    """Compute IoU

    Args:
        conf_matrix (np.array): n x n
            confusion matrix

    Returns:
        np.array: (n,)
            IoU of classes
    """
    if conf_matrix.sum() == 0:
        return 0
    with np.errstate(divide="ignore", invalid="ignore"):
        union = np.maximum(1.0, conf_matrix.sum(axis=1) +
                           conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        intersect = np.diag(conf_matrix)

        # changed by Saeid
        # IU = np.nan_to_num(intersect / union)
        IU = intersect / union
    return IU


def iou_single_class(labels, predictions, class_no):
    """
    class should be number of classes
    """
    labels_c = (labels == class_no)
    pred_c = (predictions == class_no)
    labels_c_sum = (labels_c).sum()
    pred_c_sum = (pred_c).sum()

    if (labels_c_sum > 0) or (pred_c_sum > 0):
        intersect = np.logical_and(labels_c, pred_c).sum()
        union = labels_c_sum + pred_c_sum - intersect
        with np.errstate(divide="ignore", invalid="ignore"):
            return (intersect / union)
    return 0


def get_miou(true_label, predict_label, number_classes):
    # get the miou of single image

    # both should be 2 dim
    conf_matrix = confusion_matrix(x=predict_label,
                                   y=true_label,
                                   n=number_classes)

    iou_arr = getIoU(conf_matrix)

    return np.nanmean(iou_arr[1:])


def get_cof_arr(true_label, predict_label, number_classes):
    # get the miou of single image

    # both should be 2 dim
    conf_matrix = confusion_matrix(x=predict_label,
                                   y=true_label,
                                   n=number_classes)

    iou_arr = getIoU(conf_matrix)
    # changed by Saeid
    iou_arr = np.nan_to_num(iou_arr)

    return conf_matrix, iou_arr
