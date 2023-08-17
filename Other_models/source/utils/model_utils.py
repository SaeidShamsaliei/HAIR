from scipy import ndimage, misc
from skimage.segmentation import mark_boundaries, find_boundaries
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.segmentation import slic, quickshift

from PIL import Image
import imageio
from unicodedata import name
import tensorflow as tf
import sklearn.metrics
import scipy.ndimage as nd
import pandas as pd
import numpy as np
import gdal
import sys
import random
import os
import glob
from scipy.special import softmax
from scipy.stats import entropy
import shutil
import segmentation_models as sm

from keras_vision_transfrom.keras_vision_transformer import swin_layers
from keras_vision_transfrom.keras_vision_transformer import transformer_layers
from keras_vision_transfrom.keras_vision_transformer import utils
from keras_vision_transfrom.swin_transformer import swin_transformer_stack, swin_unet_2d_base, swin_transformers
from model import DeeplabV3Plus
# New models
from keras_vision_transfrom.swin_transformer import swin_transformers
from hrnet import seg_hrnet
sys.path.append("../..")
sys.path.append("..")


# from image_processing_utils import qshitf_boundary, laplacian_filter
# sys.path.append("..")
# ----------------------------------------hacky solution for import------------------------------


def vgg16_unet(image_size=512, n_max_filters=512,
               freeze="all", context_mode=False, dropout=0.0, num_classes=5):
    """
    A unet model that uses a pre-trained VGG16 CNN as the encoder part.
    :param num_classes: The number of classes
    :param image_size: The size of the input images
    :param n_max_filters: The number of filters at the bottom layer of the unet model.
    :param freeze: Specifies what layers to freeze during training. The frozen layers will not be trained.
                all: all of the VGG16 layers are frozen. first: all but the last conv block of VGG16 is frozen.
                none: no layers are frozen. number: freeze all conv blocks upto and including the number.
    :return: A keras model
    """

    # Determine what layers to freeze
    freeze = str(freeze).lower()
    freeze_until = None
    if freeze == "all":
        freeze_until = 19
    elif freeze == "first":
        freeze_until = 15
    elif freeze == "1":
        freeze_until = 3
    elif freeze == "2":
        freeze_until = 6
    elif freeze == "3":
        freeze_until = 10
    elif freeze == "4":
        freeze_until = 14
    elif freeze == "5":
        freeze_until = 18
    else:
        freeze_until = 0

    # Define input. It has 3 color channels since vgg is trained on a color dataset
    input = tf.keras.Input(shape=(image_size, image_size, 3))

    # Load pre-trained model
    vgg16 = tf.keras.applications.vgg16.VGG16(weights="imagenet",
                                              include_top=False, input_tensor=input)
    for i, layer in enumerate(vgg16.layers):
        if i < freeze_until:
            layer.trainable = False

    skip_connections = []

    # Get first conv block
    x = vgg16.layers[1](input)  # Conv layer
    x = vgg16.layers[2](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[3](x)  # Pooling layer

    # Get 2nd conv block
    x = vgg16.layers[4](x)  # Conv layer
    x = vgg16.layers[5](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[6](x)  # Pooling layer

    # Get 3rd conv block
    x = vgg16.layers[7](x)  # Conv layer
    x = vgg16.layers[8](x)  # Conv layer
    x = vgg16.layers[9](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[10](x)  # Pooling layer

    # Get 4th conv block
    x = vgg16.layers[11](x)  # Conv layer
    x = vgg16.layers[12](x)  # Conv layer
    x = vgg16.layers[13](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[14](x)  # Pooling layer

    # # Get 5th conv block
    x = vgg16.layers[15](x)  # Conv layer
    x = vgg16.layers[16](x)  # Conv layer
    x = vgg16.layers[17](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[18](x)  # Pooling layer

    # Starting upscaling and decoding
    for i, skip_i in enumerate(reversed(range(len(skip_connections)))):
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)), kernel_size=(3, 3), strides=2,
                                            padding="same", activation="relu")(x)  # Upsample
        # Add skip connection to the channels
        x = tf.concat([x, skip_connections[skip_i]], axis=-1)

    # Last conv layers
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), padding="same", activation="relu")(x)
    if context_mode:
        # Crop to only predict on the middle pixels
        x = tf.keras.layers.MaxPool2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=(
        3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model


def predict_proba(X, model_function, num_samples):
    """
    performs prediction in training mode so that the dropout will be performed
    for num_samples times and takes the mean of them 

    """
    preds = [model_function(X, training=True) for _ in range(num_samples)]
    return np.stack(preds).mean(axis=0)


def predict_proba_stacked(X, model_function, num_samples):
    """
    pperforms prediction in training mode so that the dropout will be performed
    and stack them
    """
    preds = [softmax(model_function(X, training=True), axis=-1).squeeze()
             for _ in range(num_samples)]
    return np.stack(preds, axis=0)


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


def laplacian_filter(img, sigma=5):
    """
    performs laplacian after removed the noise using gaussian filter
    """
    # remove noise
    img = ndimage.gaussian_filter(img, sigma=sigma)
    # apply laplacian
    result = ndimage.laplace(img)

    return result

# -----------------------------------------------------------------------------------------------


def load_png_to_np_normalized(img_path, image_name, normalized=False):
    from imageio import imread
    """
    return: A numpy array with shape (n, image_size, image_size, 3)
    """
    val_x_direct = glob.glob(os.path.join(img_path,
                                          image_name,
                                          "img/*.png"))
    if normalized:
        image_np = np.stack(
            [imread(img_path) / 255.0 for img_path in val_x_direct], axis=0)
    else:
        image_np = np.stack([imread(img_path)
                            for img_path in val_x_direct], axis=0)
    if image_np.ndim < 4:
        image_np = np.expand_dims(image_np, axis=-1)

    return image_np


def load_data(image_path, label_path=None):
    # Load image
    sys.path.append("..")
    import data_processing
    image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None
    if np.isnan(np.min(image_matrix)):
        # The image contains a NaN value and will therefore be discarded
        return None
    if label_path == None:
        label_matrix = None
    else:
        # Load label
        label_ds = gdal.Open(label_path, gdal.GA_ReadOnly)
        label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
        label_ds = None
        if np.isnan(np.min(label_matrix)):
            # The labels contains a NaN value and will therefore be discarded
            return None

    training_image = data_processing.TrainingImage(image_matrix, label_matrix, geo_transform,
                                                   name=os.path.split(image_path)[-1], projection=projection)
    return training_image


def load_dataset(data_folder_path):
    """
    A function to load an entire dataset given the data folder.
    :param data_folder_path: Path to the folder with a subfolders called images and labels. These subfolder should
    contain images in .tif format. A image in images should also have a corresponding image in labels with the same
    name.
    :return: The dataset as a list of TrainingImage objects.
    """

    file_paths = glob.glob(
        os.path.join(data_folder_path, "images", "*.tif"))
    file_path_endings = [os.path.split(path)[-1] for path in file_paths]
    data = []

    for ending in file_path_endings:
        image_path = os.path.join(data_folder_path, "images", ending)
        label_path = os.path.join(data_folder_path, "labels", ending)
        data.append(load_data(image_path, label_path))

    return data


def convert_training_images_to_numpy_arrays(training_images, one_hot_encode=False, normalize=True,
                                            add_filter_channel=False):
    """
    Converts the images from a list of TrainingImage objects to a numpy array for data and labels.
    :param training_images: A list of TrainingImage objects.
    :return: A tuple with (data_X, data_y) where data_X is a numpy array with shape (n, image_size, image_size, 1).
    (n is the number of images). The same applies to data_y but for the labels instead of images.
    add_filter_channel: determines if filters will be added as external channel or not
    """
    # Training set
    if add_filter_channel:
        data_set_X = np.concatenate(
            [np.expand_dims(add_extra_channels(image.data), 0) for image in training_images], 0)
        data_set_y = np.concatenate(
            [np.expand_dims(image.labels, 0) for image in training_images], 0)
    else:
        data_set_X = np.concatenate(
            [np.expand_dims(image.data, 0) for image in training_images], 0)
        data_set_y = np.concatenate(
            [np.expand_dims(image.labels, 0) for image in training_images], 0)

    # Add channel axis
    if not add_filter_channel:
        data_set_X = np.expand_dims(data_set_X, -1)
    data_set_y = np.expand_dims(data_set_y, -1)
    # Normalize images to the range [0, 1]
    if normalize:
        # 2**8 because of 8 bit encoding in original
        data_set_X = data_set_X / (2 ** 8 - 1)

    if one_hot_encode:
        data_set_y = tf.keras.utils.to_categorical(data_set_y, num_classes=6)

    return data_set_X, data_set_y


def fake_colors(data):
    """
    Adds copies of the first channel to two new channels to simulate a color image.
    :param data: A numpy array with shape (n, image_size, image_size, 1)
    :return: A numpy array with shape (n, image_size, image_size, 3)
    """

    new_data = np.concatenate([data, data, data], -1)
    return new_data


def image_augmentation(data):
    """
    Takes the original image matrix and add rotated images and mirrored images (with rotations).
    This adds 11 additional images for each original image.
    :param data:
    :return: An numpy array with the augmented images concatenated to the data array
    """
    rot_90 = np.rot90(data, axes=(1, 2))
    rot_180 = np.rot90(data, k=2, axes=(1, 2))
    rot_270 = np.rot90(data, k=3, axes=(1, 2))
    mirror = np.flip(data, axis=1)
    mirror_rot_90 = np.rot90(mirror, axes=(1, 2))
    mirror_rot_180 = np.rot90(mirror, k=2, axes=(1, 2))
    mirror_rot_270 = np.rot90(mirror, k=3, axes=(1, 2))
    mirror2 = np.flip(data, axis=2)
    # mirror2_rot_90 = np.rot90(mirror2, axes=(1, 2))
    # mirror2_rot_180 = np.rot90(mirror2, k=2, axes=(1, 2))
    # mirror2_rot_270 = np.rot90(mirror2, k=3, axes=(1, 2))
    augments = [data, rot_90, rot_180, rot_270, mirror, mirror_rot_90, mirror_rot_180,
                mirror_rot_270,
                # I realized that they are redundant
                # mirror2, mirror2_rot_90, mirror2_rot_180, mirror2_rot_270
                ]
    augmented_image_matrix = np.concatenate(augments, axis=0)

    return augmented_image_matrix


def replace_class(data, class_id=5):
    """
    Replaces the class id with the nearest neighbor class.
    :param data: A numpy array with shape (n, image_size, image_size, channels)
    :return: A numpy array with the new classes
    """
    # A boolean array with a 1 at the locations with the class to be replaced
    invalid = data == class_id

    ind = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def miou(y_true, y_pred, num_classes=6):
    """
    The intersection over union metric. Implemented with numpy.
    :param y_true: A flat numpy array with the true classes.
    :param y_pred: A flat numpy array with the predicted classes.
    :param num_classes: The number of classes.
    :return: The mean intersection over union.
    """
    ious = []
    for i in range(num_classes):
        y_true_class = y_true == i
        y_pred_class = y_pred == i
        intersection = np.sum(np.logical_and(y_true_class, y_pred_class))
        union = np.sum(np.logical_or(y_true_class, y_pred_class))
        ious.append(intersection/union)
    return np.sum(ious)/num_classes


def evaluate_model(model, data, labels, num_classes=6):
    pred = model.predict(data, batch_size=1)
    pred = np.argmax(pred, axis=-1)

    f_labels = labels.flatten()
    f_pred = pred.flatten()

    conf_mat = sklearn.metrics.confusion_matrix(f_labels, f_pred)
    mean_intersection_over_union = miou(
        f_labels, f_pred, num_classes=num_classes)
    print(conf_mat)
    print(f"miou: {mean_intersection_over_union}")
    return conf_mat, mean_intersection_over_union


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr


def load_model(model_file_path,
               have_lr_scheduler=False,
               swin_transformer_flag=False,
               lr=0.0001):
    """
    Loads the model at the file path. The model must include both architecture and weights
    :param model_file_path: The path to the model. (.hdf5 file)
    :return: The loaded model.
    changed to use for deeplab as well
    """

    # Load the model
    # model = tf.keras.models.load_model(model_file_path)
    if have_lr_scheduler:
        print('load model with Adam optimizer')
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(lr)
        lr_metric = get_lr_metric(opt)

        if swin_transformer_flag:
            print('loading the path wieghts')
            # number of channels in the first downsampling block; it is also the number of embedded dimensions
            filter_num_begin = 128
            # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
            depth = 4
            stack_num_down = 2         # number of Swin Transformers per downsampling level
            stack_num_up = 2           # number of Swin Transformers per upsampling level
            # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
            patch_size = (4, 4)
            # number of attention heads per down/upsampling level
            num_heads = [4, 8, 8, 8]
            # the size of attention window per down/upsampling level
            window_size = [4, 2, 2, 2]
            num_mlp = 512              # number of MLP nodes within the Transformer
            shift_window = True          # Apply window shifting, i.e., Swin-MSA
            # input size and output size
            input_size = 512
            classes = 6

            model = swin_transformers(
                input_size=input_size,
                n_labels=classes,
                filter_num_begin=filter_num_begin,
                depth=depth,
                stack_num_down=stack_num_up,
                stack_num_up=stack_num_up,
                patch_size=patch_size,
                num_heads=num_heads,
                window_size=window_size,
                num_mlp=num_mlp,
                shift_window=shift_window,
                name='swin_unet')

            # model = tf.keras.models.load_model(model_file_path, custom_objects={
            #     'patch_extract': transformer_layers.patch_extract,
            #     'patch_merging': transformer_layers.patch_merging, 'patch_embedding': transformer_layers.patch_embedding,
            #     'patch_expanding': transformer_layers.patch_expanding, 'Mlp': swin_layers.Mlp, 'SwinTransformerBlock': swin_layers.SwinTransformerBlock,
            #     'WindowAttention': swin_layers.WindowAttention, 'Conv2D': swin_layers.Conv2D,
            #     'drop_path': swin_layers.drop_path, 'Dense': swin_layers.Dense,
            #     'LayerNormalization': swin_layers.LayerNormalization,
            #     'swin_transformer_stack': swin_transformer_stack,
            #     'swin_unet_2d_base': swin_unet_2d_base,
            #     'swin_transformers': swin_transformers,
            # },
            #     compile=False)

            model.compile(
                optimizer=opt,
                loss=loss,
                metrics=["accuracy", lr_metric],)

            model.load_weights(model_file_path)

            print(model.summary())

        else:
            model = tf.keras.models.load_model(model_file_path, custom_objects={
                'tf': tf, 'relu6': tf.nn.relu6}, compile=False)

            model.compile(
                optimizer=opt,
                loss=loss,
                metrics=["accuracy", lr_metric],)
    else:
        model = tf.keras.models.load_model(model_file_path, custom_objects={
            'tf': tf, 'relu6': tf.nn.relu6})
    return model


def sliding_window_function(l_image_val, window_size_val, i_offset_val, j_offset_val, model_path_val):
    """
    Using the model it will predict the segmentation using sliding window of specified size with offset
    :param l_image_val: numpy array of the image
    :param window_size_val: size of window to be predicted
    :param i_offset_val: offset in x axis
    :param j_offset_val: offset of y axis
    :param model_path_val: path to the model
    :return: label of the l_image
    """
    model = load_model(model_path_val)
    window_range_i = list(range(0, l_image_val.shape[0], window_size_val))
    window_range_i[-1] = l_image_val.shape[0] - window_size_val - i_offset_val
    window_range_j = list(range(0, l_image_val.shape[1], window_size_val))
    window_range_j[-1] = l_image_val.shape[1] - window_size_val - j_offset_val
    counter = 0
    prediction_label_val = np.ones(l_image_val.shape)*5
    for i in window_range_i:
        for j in window_range_j:
            counter += 1
            temp_idx = i + i_offset_val
            temp_jdx = j + j_offset_val
            end_i = temp_idx + window_size_val
            end_j = temp_jdx + window_size_val
            #             print(f'Working on picture {counter} i={temp_idx}-{end_i} j={temp_jdx}-{end_j}')
            if end_i > l_image_val.shape[0]:
                end_i = l_image_val.shape[0]

            if end_j > l_image_val.shape[1]:
                end_j = l_image_val.shape[1]

            temp_image = l_image_val[temp_idx:end_i, temp_jdx:end_j]
            temp_image = np.expand_dims(temp_image, 0)
            temp_image = np.expand_dims(temp_image, -1)
            # make predictions using the model here
            #             print(temp_image.shape)
            data = fake_colors(temp_image)

            #             print(data.shape)
            try:
                prediction = model.predict(data)
                prediction = np.argmax(prediction, axis=-1)
                prediction = np.squeeze(prediction)
                prediction_label_val[temp_idx:end_i,
                                     temp_jdx:end_j] = prediction
            except:
                print("An exception occurred")
    return prediction_label_val


def sliding_window_ensemble(input_image_val, model_path, window_size=512, window_offset=256, number_ensembles=4,
                            verbose=1):
    """
    get prediction of different sliding windows and using majority voting provides the final label
    :param input_image_val:
    :param model_path:
    :param window_size:
    :param window_offset:
    :param number_ensembles:
    :param verbose: identifies whether to print the outputs or not, 1 by default
    :return: mode of all ensembles
    """
    prediction_list = []
    # define numpy arrays to store the mask
    for idx in range(number_ensembles):
        # prediction_list.append(np.zeros(input_image_val.shape))
        prediction_list.append(np.ones(input_image_val.shape)*5)

    if verbose == 1:
        print('Sliding window 0')
    prediction_list[0] = sliding_window_function(
        input_image_val, window_size, 0, 0, model_path)
    if verbose == 1:
        print('Sliding window 1')
    prediction_list[1] = sliding_window_function(
        input_image_val, window_size, window_offset, 0, model_path)
    if verbose == 1:
        print('Sliding window 2')
    prediction_list[2] = sliding_window_function(
        input_image_val, window_size, 0, window_offset, model_path)
    if verbose == 1:
        print('Sliding window 3')
    prediction_list[3] = sliding_window_function(input_image_val, window_size, window_offset, window_offset,
                                                 model_path)

    # get mode of all predictions
    if verbose == 1:
        print('Prediction voting')

    from scipy import stats
    all_predictions = np.stack(prediction_list)
    vote_prediction, count = stats.mode(all_predictions, axis=0)

    # swap the non difined with the first prediction which does not have any offset
    vote_prediction[0][vote_prediction[0] ==
                       5] = prediction_list[0][vote_prediction[0] == 5]
    return vote_prediction[0]


def add_extra_channels(img_matrix):
    """
    add filters to the image so convert image of (image_size, image_size)
    to (image_size, image_size, 3)
    """

    # changed to increase the edge from 1 to 255
    filter_channel = qshitf_boundary(
        img_matrix, ratio=0.9)
    cluster_channel = laplacian_filter(
        img_matrix, sigma=5)

    return np.stack(
        [img_matrix, cluster_channel, filter_channel], -1)


def add_extra_channel_save(image_file_list, output_directory):
    """
    This function adds two channels to the images and save them with the same name at output_directory
    """
    output_image_directory = os.path.join(output_directory, 'image_3channel')

    os.makedirs(output_image_directory, exist_ok=True)

    for idx, _ in enumerate(image_file_list):
        # Check if the image and label have the same name
        image_name = os.path.split(image_file_list[idx])[-1]

        # Get all the channels
        image_object = load_data(
            image_file_list[idx])
        data_channel = image_object.data
        # changed to increase the edge from 1 to 255
        filter_channel = qshitf_boundary(
            data_channel, ratio=0.9) * 255.0
        cluster_channel = laplacian_filter(
            data_channel, sigma=5)

        # Add channels as data for image object
        image_object.data = np.stack(
            [data_channel, cluster_channel, filter_channel], -1)
        image_object.write_data_to_raster(
            os.path.join(output_image_directory, image_name))
        print(f'made image {idx}')


def dataset_split_train_vlidation_test(src_path, dst_path, train_rate=0.7,
                                       val_rate=0.2, test_rate=0.1):
    """
    function to split train validation and test based on the rate
    """
    src_imgs_list = glob.glob(os.path.join(src_path, 'image', 'img', '*.png'))
    src_lbls_list = glob.glob(os.path.join(src_path, 'label', 'img', '*.png'))
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


def replace_class_directory_png(src_path, dst_path, class_id=5):
    """
    for a directory of png labels, replaces the class_id to closest neighbor
    """
    os.makedirs(dst_path, exist_ok=True)
    len_img = len(glob.glob(os.path.join(src_path, '*.png')))
    print(f"there are {len_img} images")
    for image_path in glob.glob(os.path.join(src_path, '*.png')):
        name = os.path.split(image_path)[-1]
        dest_image_path = os.path.join(dst_path, name)
        original_image_matrix = imageio.imread(image_path)
        original_image_matrix = replace_class(
            np.expand_dims(original_image_matrix, axis=0),  class_id=class_id)
        original_image_matrix = np.squeeze(original_image_matrix, axis=0)

        # save to file
        replaced_image_png = Image.fromarray(
            original_image_matrix.astype(np.uint8))
        replaced_image_png.save(dest_image_path)
    print('done')


def get_topk_softmax_layer(arr, k=2):
    return np.sort(arr, axis=-1)[..., -k:]


# lets try it for our model
def enable_dropout_at_inference_for_model(model):
    """
    input is a model, output is the same model but dropout will be enabled during the inference
    """
    model_config = model.get_config()
    for layers, layer_conf in enumerate(model_config['layers']):
        layer_index = layers  # layer index you want to modify
        if model_config['layers'][layer_index]['class_name'] == 'Dropout':
            model_config['layers'][layer_index]['inbound_nodes'][0][0][-1]['training'] = True
    model = tf.keras.models.model_from_config(model_config)
    return model


def model_by_name(model_name, freeze=0, num_classes=6, dropout=0.1, batch_size=12):
    """
    returns the tf model, models are not compiled
    params
    :model_name: string of name of the model
    :num_classes: number of output classes
    :dropout: dropout rate of the model, default=0.1
    :batch_size: batch size
    """

    # Load and compile model
    if model_name.lower() == "vgg16":
        model = vgg16_unet(freeze=freeze, context_mode=False,
                           num_classes=num_classes, dropout=dropout)
    elif model_name.lower() == 'deeplabv3':
        # model = deeplab_model.Deeplabv3(input_shape=(512, 512, 3), classes=num_classes)
        raise ValueError('model issue, check the code')
    elif model_name.lower() == "deeplabv3_resnet50":
        model = DeeplabV3Plus(image_size=512, num_classes=num_classes)
        print('\n deep lab with resnet')
        # print(model.summary())
    elif model_name.lower() == "fpn_resnet34":
        model = sm.FPN('resnet34', input_shape=(512, 512, 3),
                       classes=num_classes, encoder_weights='imagenet', pyramid_dropout=dropout)
        print('\n FPN with resnet34')
        # print(model.summary())
    elif model_name.lower() == "fpn_resnet50":
        model = sm.FPN('resnet50', input_shape=(512, 512, 3),
                       classes=num_classes, encoder_weights='imagenet', pyramid_dropout=dropout)
        print('\n FPN with resnet50')
        #
    elif model_name.lower() == "fpn_resnext101":
        model = sm.FPN('resnext101', input_shape=(512, 512, 3),
                       classes=num_classes, encoder_weights='imagenet', pyramid_dropout=dropout)
        print('\n FPN with resnext101')
        # print(model.summary())
    elif model_name.lower() == "diresunet":
        raise NotImplementedError('not implemented')
    elif model_name.lower() == "unet_resnet50":
        model = sm.Unet('resnet50', input_shape=(512, 512, 3), classes=num_classes,
                        encoder_weights='imagenet')
        print('\n Unet with resnet')
    # hrnet
    elif model_name == 'hrnet':
        model = seg_hrnet(batch_size=batch_size,
                          height=512,
                          width=512,
                          channel=3,
                          classes=num_classes)

    elif model_name == 'swin_transformer':
        # number of channels in the first downsampling block; it is also the number of embedded dimensions
        filter_num_begin = 128
        # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level
        depth = 4
        stack_num_down = 2         # number of Swin Transformers per downsampling level
        stack_num_up = 2           # number of Swin Transformers per upsampling level
        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
        patch_size = (4, 4)
        # number of attention heads per down/upsampling level
        num_heads = [4, 8, 8, 8]
        # the size of attention window per down/upsampling level
        window_size = [4, 2, 2, 2]
        num_mlp = 512              # number of MLP nodes within the Transformer
        shift_window = True          # Apply window shifting, i.e., Swin-MSA
        # input size and output size
        input_size = 512
        classes = num_classes

        model = swin_transformers(
            input_size=input_size,
            n_labels=classes,
            filter_num_begin=filter_num_begin,
            depth=depth,
            stack_num_down=stack_num_up,
            stack_num_up=stack_num_up,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=window_size,
            num_mlp=num_mlp,
            shift_window=shift_window,
            name='swin_unet')

        print('\n Transformers')

    else:
        print('else it seems')
        model = None
        raise ValueError(f'{model_name} not defined')

    return model


def load_model_with_different_output(
        model_original,
        no_class_new,
        output_idx,
        is_hrnet):
    """
    Changes the output layer of the 'model_original' from original class number to 'no_class_new'.
    :param output_idx for FPN and UNet, it is -2, for others it is -1 
    :return the new model which is not compiled
    """
    # for hrnet
    if is_hrnet:
        print('loading HRNet')
        x = model_original.layers[output_idx].output
        # define the last layer for hrnet
        x = tf.keras.layers.Conv2D(
            no_class_new, 1, use_bias=False, kernel_initializer='he_normal',
            name='hairs_output_conv')(x)
        x = tf.keras.layers.BatchNormalization(
            axis=3, name='hairs_batch_normalization_2')(x)
        x = tf.keras.layers.Activation(
            'sigmoid', name='hair_Classification')(x)
    else:
        if output_idx < -3:
            raise ValueError('the is_hrnet should be set to True')
        # make the pretrained model
        x = model_original.layers[output_idx].output
        # define the last layer
        x = tf.keras.layers.Conv2D(
            no_class_new,
            kernel_size=model_original.layers[output_idx+1].kernel_size,
            padding="same", name='hairs_output_conv')(x)

    return tf.keras.Model(inputs=model_original.input, outputs=x)

if __name__ == '__main__':
    # set_type_list = ['val', 'test']
    # for set_type in set_type_list:
    #     image_directory = f'/home/saeid/phd/segmentation/dataset/machine_learning_dataset/{set_type}/images'
    #     label_directory = f'/home/saeid/phd/segmentation/dataset/machine_learning_dataset/{set_type}/labels'

    #     image_file_list = glob.glob(os.path.join(image_directory, "*.tif"))
    #     label_file_list = glob.glob(os.path.join(label_directory, "*.tif"))
    #     output_path = f'/home/saeid/phd/segmentation/dataset/machine_learning_dataset_3channel/{set_type}'
    #     add_extra_channel_save(image_file_list=image_file_list,
    #                            output_directory=output_path)
    # src_path = '/home/saeid/phd/segmentation/dataset/machine_learning_dataset_3channel/train/labels_png3d/img'
    # dst_path = '/home/saeid/phd/segmentation/dataset/machine_learning_dataset_3channel/train/labels_augment_replaced/img'
    # replace_class_directory_png(src_path, dst_path, class_id=5)
    pass
