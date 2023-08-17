# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Edited by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, "../../..")
add_path(lib_path)


def make_untrainable(model):
    """
    freezing the parameters of a model
    """
    for param in model.parameters():
        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_seg_model(cfg, **kwargs):

    torch.cuda.empty_cache()
    from magnet.model.fpn import ResnetFPN
    from magnet.model.hrnet_ocr import HRNetW18_OCR, HRNetW18, HRNetW48, HRNetW48_OCR, HRNetW32

    if cfg.MODEL.NAME == "ResnetFPN":
        model_class = ResnetFPN
    elif cfg.MODEL.NAME == "HRNetW18_OCR":
        model_class = HRNetW18_OCR
    elif cfg.MODEL.NAME == "HRNetW18":
        model_class = HRNetW18
    elif cfg.MODEL.NAME == "HRNetW48":
        model_class = HRNetW48
    elif cfg.MODEL.NAME == "HRNetW32":
        model_class = HRNetW32
    elif cfg.MODEL.NAME == "HRNetW48_OCR":
        model_class = HRNetW48_OCR

    # initialize with pre-trained
    if cfg.MODEL.CHANGE_LAST_LAYER != None and cfg.MODEL.NAME == "ResnetFPN":
        print('initializing with the deepglobe and changing the last layer')
        model = model_class(7)
        model.init_weights(cfg.MODEL.PRETRAINED)
        # here we freeze everything
        if cfg.MODEL.FREEZE_MODEL:
            print('freeze everything but last layers')
            make_untrainable(model)
        
        model.classify = nn.Conv2d(
            128 * 4, cfg.DATASET.NUM_CLASSES, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(model.classify.weight)
        # freeze encoder if it is set to be frozen
        if cfg.MODEL.FREEZE_ENCODER:
            print('freeze the encoder')
            make_untrainable(model.resnet_backbone)

        if cfg.MODEL.GRAY_TO_RGB_LAYER:
            print('\n grayscale to rgb layer')
            first_layer = nn.Conv2d(in_channels=1, out_channels=3,
                                    kernel_size=1, stride=1, padding=0)
            torch.nn.init.xavier_uniform_(first_layer.weight)
            model = nn.Sequential(first_layer, model)
        print(f'Total trainable parameters: {count_parameters(model)}')

    else:
        model = model_class(cfg.DATASET.NUM_CLASSES)
        print(f'Total trainable parameters: {count_parameters(model)}')
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
