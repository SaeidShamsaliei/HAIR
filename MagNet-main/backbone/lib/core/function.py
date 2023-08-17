# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Editted by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from cgi import print_arguments
from errno import ETIMEDOUT
import logging
import os
import time

import lib.utils.distributed as dist
import numpy as np
import torch
from lib.utils.utils import AverageMeter, adjust_learning_rate, get_confusion_matrix
from torch.nn import functional as F
from tqdm import tqdm


@torch.no_grad()
def magnet_update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        
        if isinstance(input, (list, tuple)):
            input_img, input_lbl, _, _ = input

        if device is not None:
            # input = input.to(device)
            input_img = input_img.to(device)
            input_lbl = input_lbl.long().to(device)
        model(input_img, input_lbl)


    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

# from torch.optim.swa_utils import AveragedModel, SWALR

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict["writer"]
    global_steps = writer_dict["train_global_steps"]

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        _ = adjust_learning_rate(
            optimizer, base_lr, num_iters, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = "Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, " "lr: {}, Loss: {:.6f}".format(
                epoch,
                num_epoch,
                i_iter,
                epoch_iters,
                batch_time.average(),
                [x["lr"] for x in optimizer.param_groups],
                ave_loss.average(),
            )
            logging.info(msg)

    writer.add_scalar("train_loss", ave_loss.average(), global_steps)
    # get the learning rate
    lr_to_write = get_lr(optimizer=optimizer)
    writer.add_scalar("learning_rate", lr_to_write, global_steps)
    writer.add_scalar("train_loss", ave_loss.average(), global_steps)
    writer_dict["train_global_steps"] = global_steps + 1


def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:], mode="bilinear", align_corners=config.MODEL.ALIGN_CORNERS)

                confusion_matrix[..., i] += get_confusion_matrix(
                    label, x, size, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL
                )

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = tp / np.maximum(1.0, pos + res - tp)
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info("{} {} {}".format(i, IoU_array, mean_IoU))

    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]
    writer.add_scalar("valid_loss", ave_loss.average(), global_steps)
    writer.add_scalar("valid_mIoU", mean_IoU, global_steps)
    writer_dict["valid_global_steps"] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model, sv_dir="", sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config, model, image, scales=config.TEST.SCALE_LIST, flip=config.TEST.FLIP_TEST
            )

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0: pred.size(
                    2) - border_padding[0], 0: pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:], mode="bilinear", align_corners=config.MODEL.ALIGN_CORNERS)

            confusion_matrix += get_confusion_matrix(
                label, pred, size, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL
            )

            if sv_pred:
                sv_path = os.path.join(sv_dir, "test_results")
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info("processing: %d images" % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = tp / np.maximum(1.0, pos + res - tp)
                mean_IoU = IoU_array.mean()
                logging.info("mIoU: %.4f" % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = tp / np.maximum(1.0, pos + res - tp)
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model, sv_dir="", sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config, model, image, scales=config.TEST.SCALE_LIST, flip=config.TEST.FLIP_TEST
            )

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:], mode="bilinear", align_corners=config.MODEL.ALIGN_CORNERS)

            if sv_pred:
                sv_path = os.path.join(sv_dir, "test_results")
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

# add swa
def train_swa(config, epoch, num_epoch, epoch_iters, base_lr, num_iters, trainloader, optimizer, model, writer_dict,
                swa_model, swa_scheduler, swa_start_epoch):
    # Training
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict["writer"]
    global_steps = writer_dict["train_global_steps"]

    for i_iter, batch in enumerate(trainloader, 0):

        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        _ = adjust_learning_rate(
            optimizer, base_lr, num_iters, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = "Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, " "lr: {}, Loss: {:.6f}".format(
                epoch,
                num_epoch,
                i_iter,
                epoch_iters,
                batch_time.average(),
                [x["lr"] for x in optimizer.param_groups],
                ave_loss.average(),
            )
            logging.info(msg)
        
        # changing the iteration
        if i_iter > epoch_iters:
            print('breaking the iteration')
            break 
    
    # update swa if needed
    if epoch >= swa_start_epoch:
        logging.info('swa update')
        swa_model.update_parameters(model)
        swa_scheduler.step()
    
    # swa save the model
    if (epoch+1) == num_epoch:
        logging.info('swa bn update')
        swa_lr = [x["swa_lr"] for x in optimizer.param_groups]
        logging.info(f"swa_lr : {swa_lr}")
        # print(swa_model)
        # print(swa_model.__dict__)
        magnet_update_bn(trainloader, swa_model, device=device)


    writer.add_scalar("train_loss", ave_loss.average(), global_steps)
    # get the learning rate
    lr_to_write = get_lr(optimizer=optimizer)
    writer.add_scalar("learning_rate", lr_to_write, global_steps)
    writer.add_scalar("train_loss", ave_loss.average(), global_steps)
    writer_dict["train_global_steps"] = global_steps + 1