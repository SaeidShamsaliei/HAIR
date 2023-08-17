# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .base_dataset import BaseDataset
import albumentations

AUGMENTATIONS = albumentations.Compose([
        # pixel level aumentation
            albumentations.Transpose(p=0.1),
            albumentations.Flip(p=0.1),
            # albumentations.OneOf([
            #     albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            #     albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
            # ],p=0.2),

        #     albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            
        #     # bluring filters
        # #     albumentations.GaussianBlur(p=0.05),
        #     albumentations.OneOf([
        #         albumentations.MedianBlur(),
        #         albumentations.RandomGamma(),
        #         albumentations.MotionBlur(),
        #     ],p=0.1),
            
            # will be added
            # albumentations.GridDistortion(p=0.1),
            # albumentations.ImageCompression(p=0.1),
            # albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
        ])

# AUGMENTATIONS = albumentations.Compose([
#         # pixel level aumentation
#             albumentations.Transpose(p=0.2),
#             albumentations.Flip(p=0.2),
#             albumentations.OneOf([
#                 albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
#                 albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
#             ],p=0.2),
            
#             # bluring filters
#         #     albumentations.GaussianBlur(p=0.05),
#             albumentations.OneOf([
#                 albumentations.MedianBlur(),
#                 albumentations.RandomGamma(),
#                 albumentations.MotionBlur(),
#             ],p=0.2),
            
#             # will be added
#             albumentations.GridDistortion(p=0.1),
#             albumentations.ImageCompression(p=0.1),
#             albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
#         ])

class DeepGlobeRiver(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        num_samples=None,
        num_classes=6,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=2048,
        crop_size=(512, 512),
        resize=(508, 508),
        downsample_rate=1,
        scale_factor=16,
        # mean=[0.485],
        # std=[0.229],
        mean=[0.485],
        std=[0.229],
        gray_scale_use=False,
    ):

        if not gray_scale_use:
            print()
            mean = [mean[0], mean[0], mean[0]]
            std = [std[0], std[0], std[0]]
            cv2_reading_mode = cv2.IMREAD_COLOR
        else:
            cv2_reading_mode = cv2.IMREAD_GRAYSCALE

        # since it is hrnet it is 
        # self.original_size = (512, 512)

        # since it is magnet
        self.original_size = (2448, 2448)

        super(DeepGlobeRiver, self).__init__(
            ignore_label,
            base_size,
            crop_size,
            resize,
            downsample_rate,
            scale_factor,
            mean,
            std,
        )
        
        self.gray_scale_use = gray_scale_use
        self.cv2_reading_mode = cv2_reading_mode
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split()
                         for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label2color = {
            0: [0, 0, 0],    # unknown
            1: [255, 255, 0],  # gravel
            2: [0, 128, 0],  # vegetation
            3: [128, 128, 0],  # farmland
            4: [128, 0, 128],  # human_construction
            5: [0, 0, 255],  # water
        }

        self.class_weights = torch.FloatTensor(
            # [0.1, 1.3, 1.0, 1.0, 1.4, 1.3]).cuda()
            [0.1, 3.0, 1.0, 1.0, 2.0, 3.0]).cuda()

        self.multi_crops = [(612, 612), (1224, 1224), (2448, 2448)]
        # print('################ only 612x612 crops ################')
        # self.multi_crops = [(512, 512)]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({"img": image_path, "label": label_path,
                         "name": name, "weight": 1})
        return files

    def convert_label(self, label):
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w), dtype="uint8")

        for classnum, color in self.label2color.items():
            indices = np.where(np.all(label == tuple(color)[::-1], axis=-1))
            classmap[indices[0].tolist(), indices[1].tolist()] = classnum
        return classmap

    def __getitem__(self, index):

        image_index = index // (self.n_x * self.n_y)
        patch_idx = index % (self.n_x * self.n_y)
        # print(f'nx is {self.n_x}')

        item = self.files[image_index]
        name = item["name"]
        try:
            image = cv2.imread(os.path.join(
                self.root, "deepglobe_river", item["img"]), self.cv2_reading_mode)
            # print(item['img'])
        except:
            raise ValueError(f'the image {item["img"]}')
            # self.root, "deepglobe_river", item["img"]), cv2.IMREAD_GRAYSCALE)
        if self.gray_scale_use:
            image = np.expand_dims(image, axis=-1)
        
        if image is None:
            img_err = item["img"]
            raise ValueError(f'the image {img_err} is {type(image)}')
        size = image.shape
        # print(size)
        try:
            label = cv2.imread(os.path.join(
                self.root, "deepglobe_river", item["label"]), cv2.IMREAD_COLOR)
        except:
            raise ValueError(f'the image {item["label"]}')
        # print(f'before crop: {image.shape}')

        assert item["label"].replace("label", "") == item["img"].replace("image", "")


        if self.crop_size != self.original_size:
            # print('\n \n it is here')
            crop = random.choice(self.multi_crops)
            # if crop == (612, 612):
            #     x = (patch_idx // self.n_x) * int(crop[1])
            #     y = (patch_idx % self.n_x) * int(crop[0])
            # else:
            x = random.randint(0, self.original_size[1] - crop[1])
            y = random.randint(0, self.original_size[0] - crop[0])
            image = image[y: y + crop[0], x: x + crop[1]]
            label = label[y: y + crop[0], x: x + crop[1]]

        
        # print("SHAPE", image.shape, label.shape)
        # print(f'before resize: {image.shape}')
        # apply furthur augmentations here

        # transformed = AUGMENTATIONS(image=image, mask=label)
        # image = transformed['image']
        # label = transformed['mask']
        

        image, label = self.image_resize(image, self.resize[1], label)
        size = image.shape

        label = self.convert_label(label)
        # print(f'before gen sample: {image.shape}')
        image, label = self.gen_sample(
            image, label, self.multi_scale, self.flip)
        
        
        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros(
            [1, self.num_classes, ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(
                image=image, rand_scale=scale, rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(
                    np.ceil(1.0 * (new_h - self.crop_size[0]) / stride_h)) + 1
                cols = np.int(
                    np.ceil(1.0 * (new_w - self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes, new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:,
                                                          :, 0: h1 - h0, 0: w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), mode="bilinear", align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
                palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
                palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + ".png"))
    
    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        original_image_ndim = image.ndim
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if original_image_ndim != image.ndim:
            image = np.expand_dims(image, axis=-1)

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        original_image_ndim = image.ndim
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if original_image_ndim != image.ndim:
            image = np.expand_dims(image, axis=-1)

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = np.int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = np.int(h * short_length / w + 0.5)
        original_image_ndim = image.ndim
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if original_image_ndim != image.ndim:
            image = np.expand_dims(image, axis=-1)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image
