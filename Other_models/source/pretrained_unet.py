import tensorflow as tf
import numpy as np
import os
import datetime
import model_utils
import time
import resource
import deepv3_plus.model as deeplab_model
from model import DeeplabV3Plus
from diresunet import DIResUNet
import json
from swa.tfkeras import SWA
from ImageDataAugmentor.image_data_augmentor import *
import albumentations
# import matplotlib.pyplot as plt
import segmentation_models as sm

# New models
from keras_vision_transfrom.swin_transformer import swin_transformers
from hrnet import seg_hrnet

# to solve problem with cudnn I will limit the TF
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_device, True)
        print('\nSet memory growth to True\n')
    except:
        print('\nDid not set growth rate\n')

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([0.1, 1.3, 1.0, 1.0, 1.1, 1.3])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights

def balanced_flow_from_directory(flow_from_directory):
    for x, y in flow_from_directory:
         yield add_sample_weights(x, y)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

def vgg16_unet(image_size=512, n_max_filters=512, freeze="all", context_mode=False, dropout=0.0, num_classes=5):
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
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)), kernel_size=(3, 3), strides=2,
                                            padding="same", activation="relu")(x)  # Upsample
        x = tf.concat([x, skip_connections[skip_i]], axis=-1)  # Add skip connection to the channels

    # Last conv layers
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if context_mode:
        # Crop to only predict on the middle pixels
        x = tf.keras.layers.MaxPool2D()(x)
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model


def run_from_dir(train_data_folder_path, val_data_folder_path, model_name="vgg16", freeze="all",
                 run_path="/home/kitkat/PycharmProjects/river-segmentation/runs", batch_size=1, dropout=0, 
                 train_image_directory_name='images', train_label_directory_name='labels',
                 val_image_directory_name='images', val_label_directory_name='labels', lr=0.0001, num_classes=5, conf_exp=None, 
                 use_swa=False, callback_patience=10, swa_start_epochs=10, use_albumentation=False, img_per_epoch=9353,
                 regularizer_flag=False, use_grayscale_only=False, adam_schedule=False,
                 class_weight=False, use_reduce_lr_platueau=False, seed_val=2):
    """
        Trains a CNN Unet model and saves the best model to file. Uses training images from disk instead of loading
        everything into RAM.

        :param train_data_folder_path: Path to the folder containing training images (.png format)
        :param val_data_folder_path: Path to the folder containing validation images (.png format)
        :param model_name: The name of the model. Supported models are: vgg16
        :param freeze: Determine how many blocks in the encoder that are frozen during training.
        :param conf_exp: a dictionary which will only be used to write to a file in order to track the experiments. if sets to None nothing will be written
        Should be all, first, 1, 2, 3, 4, 5 or none
        :param run_path: Folder where the run information and model will be saved.
        :param dropout: Drop rate, [0.0, 1)
        :return: Writes model to the run folder, nothing is returned.
        """

    steps_per_epoch=int(np.ceil(img_per_epoch/batch_size))
    # steps_per_epoch=int(np.ceil(38424/batch_size))
    tf.keras.backend.clear_session()
    start_time = time.time()

    # Make run name based on parameters and timestamp
    run_name = f"{model_name}_freeze_{freeze}"
    date = str(datetime.datetime.now())
    run_path = os.path.join(run_path, f"{date}_{run_name}".replace(" ", "_"))
    os.makedirs(run_path, exist_ok=True)
    if conf_exp is not None:
        # write the config to file (in a hacky way)
        file_name = 'config.txt'
        with open(os.path.join(run_path, file_name), 'w') as convert_file:
            convert_file.write(json.dumps(conf_exp))

    # train and validation datapath
    train_img_path = os.path.join(train_data_folder_path, train_image_directory_name)
    train_lbl_path = os.path.join(train_data_folder_path, train_label_directory_name)
    val_img_path = os.path.join(val_data_folder_path, val_image_directory_name)
    val_lbl_path = os.path.join(val_data_folder_path, val_label_directory_name)

    print(f'train \n\timage path: {train_img_path} \n\tlabel path: {train_lbl_path}')
    print(f'val \n\timage path: {val_img_path} \n\tlabel path: {val_lbl_path}')

    # Setup data generators
    if not use_albumentation:

        print('\n\nno extra augmentations\n\n')
        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=lambda x: x/(2**8 -1))
        mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        if use_grayscale_only:
            print('\n loading Grayscale image \n')
            image_generator = image_datagen.flow_from_directory(
                train_img_path,class_mode=None, 
                target_size=(512, 512), seed=seed_val, 
                batch_size=batch_size, color_mode="grayscale")
        else:
            print('\n loading RGB image \n')
            image_generator = image_datagen.flow_from_directory(
                train_img_path,class_mode=None, 
                target_size=(512, 512), seed=seed_val, 
                batch_size=batch_size)

        mask_generator = mask_datagen.flow_from_directory(train_lbl_path,
                                        class_mode=None, target_size=(512, 512), 
                                        seed=seed_val, batch_size=batch_size,
                                        color_mode="grayscale")
    
    # albumentation data loader part
    # ------------------------------------------------------
    else:

        
        # print('\n\nusing albumentations v1\n\n')
        # AUGMENTATIONS = albumentations.Compose([
        # # pixel level aumentation
        #     albumentations.Transpose(p=0.2),
        #     albumentations.Flip(p=0.2),
        #     albumentations.OneOf([
        #         albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        #         albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        #     ],p=0.2),
            
        #     # bluring filters
        # #     albumentations.GaussianBlur(p=0.05),
        #     albumentations.OneOf([
        #         albumentations.MedianBlur(),
        #         albumentations.RandomGamma(),
        #         albumentations.MotionBlur(),
        #     ],p=0.2),
            
        #     # will be added
        #     albumentations.GridDistortion(p=0.1),
        #     albumentations.ImageCompression(p=0.1),
        #     albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
        # ])

        print('\n\nusing albumentations v2\n\n')
        print('\n\nusing albumentations v2\n\n')
        AUGMENTATIONS = albumentations.Compose([
        # pixel level aumentation
            albumentations.Transpose(p=0.1),
            albumentations.Flip(p=0.1),
            # albumentations.OneOf([
            #     albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            #     albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
            # ],p=0.2),

            albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            
            # bluring filters
        #     albumentations.GaussianBlur(p=0.05),
            albumentations.OneOf([
                albumentations.MedianBlur(),
                albumentations.RandomGamma(),
                albumentations.MotionBlur(),
            ],p=0.1),
            
            # will be added
            # albumentations.GridDistortion(p=0.1),
            # albumentations.ImageCompression(p=0.1),
            # albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
        ])

        # print('\n\nusing albumentations v3\n\n')
        # AUGMENTATIONS = albumentations.Compose([
        # # pixel level aumentation
        #     albumentations.Transpose(p=0.1),
        #     albumentations.Flip(p=0.1),
        #     # albumentations.OneOf([
        #     #     albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        #     #     albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        #     # ],p=0.2),

        #     # albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.1),
            
        #     # bluring filters
        # #     albumentations.GaussianBlur(p=0.05),
        #     # albumentations.OneOf([
        #     #     albumentations.MedianBlur(),
        #     #     albumentations.RandomGamma(),
        #     #     albumentations.MotionBlur(),
        #     # ],p=0.1),
            
        #     # will be added
        #     # albumentations.GridDistortion(p=0.1),
        #     # albumentations.ImageCompression(p=0.1),
        #     # albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
        # ])

        image_datagen = ImageDataAugmentor(
            input_augment_mode='image',
                augment=AUGMENTATIONS,
                preprocess_input=lambda x: x/(2**8 -1),
                seed=seed_val)
        if use_grayscale_only:
            image_generator = image_datagen.flow_from_directory(
                train_img_path,class_mode=None, 
                target_size=(512, 512), seed=seed_val, 
                batch_size=batch_size, color_mode="grayscale")
        else:
            image_generator = image_datagen.flow_from_directory(
                train_img_path,class_mode=None, 
                target_size=(512, 512), seed=seed_val, 
                batch_size=batch_size)

        mask_datagen = ImageDataAugmentor(
            augment=AUGMENTATIONS,
            input_augment_mode='mask',
            seed=seed_val
        )
        mask_generator = mask_datagen.flow_from_directory(
            train_lbl_path,class_mode=None, 
            target_size=(512, 512), seed=seed_val, 
            batch_size=batch_size, color_mode='grayscale',
            )


    # --------------------------------------------------------    
    train_generator = (pair for pair in zip(image_generator, mask_generator))

    # tmp_img_instance, tmp_mask_instance = next(train_generator)
    # print(os.path.join(train_data_folder_path, "images", 'img'))
    # print(f'\n\n temp pair is: {tmp_img_instance.shape} {tmp_mask_instance.shape}')
    # fig, axs = plt.subplots(2)
    # fig.suptitle('check the image 1')
    # axs[0].imshow(tmp_img_instance[0])
    # axs[1].imshow(tmp_mask_instance[0].squeeze())
    # plt.savefig('/workspace/data/foo1.png')

    # tmp_img_instance, tmp_mask_instance = next(train_generator)
    # fig, axs = plt.subplots(2)
    # fig.suptitle('check the image 2')
    # axs[0].imshow(tmp_img_instance[0])
    # axs[1].imshow(tmp_mask_instance[0].squeeze())
    # plt.savefig('/workspace/data/foo2.png')

    # tmp_img_instance, tmp_mask_instance = next(train_generator)
    # fig, axs = plt.subplots(2)
    # fig.suptitle('check the image 3')
    # axs[0].imshow(tmp_img_instance[0])
    # axs[1].imshow(tmp_mask_instance[0].squeeze())
    # plt.savefig('/workspace/data/foo3.png')
    # return 13

    #000000000000000000000000000000000000000000000000Arild part0000000000000000000000000000000000000000000
    # Validation data
    # val = model_utils.load_dataset(val_data_folder_path)
    # val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    # val_X = model_utils.fake_colors(val_X)
    # val_y = model_utils.replace_class(val_y, class_id=5)

    #0000000000000000000000000000000000000000000000000My code000000000000000000000000000000000000000000000

    # print('loading val images')
    # val_X = model_utils.load_png_to_np_normalized(img_path=val_data_folder_path, 
    #                  image_name=val_image_directory_name, normalized=True)
    
    # # repeat channels if necessary
    # if val_X.shape[-1] != 3:
    #     print('add fake channels for validation')
    #     val_X = model_utils.fake_colors(val_X)

    # print('loading val label')
    # val_y = model_utils.load_png_to_np_normalized(img_path=val_data_folder_path, 
    #                  image_name=val_label_directory_name, normalized=False)
    
    # #TODO (can be changed with ignore)
    # if num_classes == 5:
    #     print('replace classes')
    #     val_y = model_utils.replace_class(val_y, class_id=5)
    #     print('done')


    # validation using the generator
    val_batch_size = batch_size
    val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=lambda x: x/(2**8 -1))
    val_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    if use_grayscale_only:
        val_image_generator = val_image_datagen.flow_from_directory(val_img_path,
                                                        class_mode=None, target_size=(512, 512), 
                                                        seed=seed_val, batch_size=val_batch_size,
                                                        color_mode='grayscale')
    else:
        val_image_generator = val_image_datagen.flow_from_directory(val_img_path,
                                                        class_mode=None, target_size=(512, 512), 
                                                        seed=seed_val, batch_size=val_batch_size)

    val_mask_generator = val_mask_datagen.flow_from_directory(val_lbl_path,
                                                    class_mode=None, target_size=(512, 512), seed=seed_val, 
                                                    batch_size=val_batch_size,
                                                    color_mode="grayscale")
    val_dataset = (val_pair for val_pair in zip(val_image_generator, val_mask_generator))
    STEP_SIZE_VALID=val_image_generator.n // val_batch_size
    print(f'validation steps: {STEP_SIZE_VALID}')

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
        model = DIResUNet(input_shape=(512, 512, 3), num_class=num_classes)
        print('\n DIResUNet')
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

    # print(model.summary())
    
    if adam_schedule:
        print('learning rate exponential decay activated')
        # lr_adam = tf.keras.optimizers.schedules.PolynomialDecay(
        #         initial_learning_rate=1e-4, decay_steps=60000, 
        #         end_learning_rate=1e-6, power=1.0,
        #         cycle=False, name=None)
        # decay in 100 steps
        decay_steps = int(steps_per_epoch * 100)
        lr_adam = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-4, decay_steps=decay_steps, 
                end_learning_rate=1e-7, power=1.0,
                cycle=False, name=None)
        
        if model_name == 'swin_transformer':
            opt = tf.keras.optimizers.Adam(lr_adam, clipvalue=0.5)
        else:
            opt = tf.keras.optimizers.Adam(lr_adam)

    else:
        print('lr is fixed')
        if model_name == 'swin_transformer':
            opt = tf.keras.optimizers.Adam(lr, clipvalue=0.5)
        else:
            opt = tf.keras.optimizers.Adam(lr)


    lr_metric = get_lr_metric(opt)

    
    # if we need to add something in the beginning
    if use_grayscale_only:
        # add a layer to convert 1d to 3d channel
        print('\nadding extra conv layer at the beginning\n')
        from tensorflow.keras.layers import Input, Conv2D
        from tensorflow.keras.models import Model

        inp = Input(shape=(512, 512, 1))
        l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
        out = model(l1)

        model = Model(inp, out, name='grayscaled_seg')

    if regularizer_flag:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    if class_weight:
        print('applying class weight')
        train_generator = balanced_flow_from_directory(train_generator)
    
    # change the from_logits for FPN and HRNet
    print('------------change in the logit difinition of loss---------------')
    if model_name == 'hrnet' or model_name == 'unet_resnet50'  or model_name == 'fpn_resnet50':
        print('\n\n --------------------------------------------------')
        print(' --------------------------------------------------')
        print('        from_logits is False   ')
        print(' --------------------------------------------------')
        print('\n\n --------------------------------------------------\n')
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy", lr_metric],)

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=callback_patience, monitor="val_loss"))

    if model_name == 'swin_transformer':
        print('\n weights will be saved')
        os.makedirs(os.path.join(run_path, 'checkpoints'), exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, 'checkpoints', 'cp.ckpt'),
                                                    monitor="val_loss", save_weights_only=True, save_best_only=True)
    else:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, "model.hdf5"),
                                                    monitor="val_loss", save_best_only=True)
    callbacks.append(checkpoint)

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_path, histogram_freq=1)
    # callbacks.append(tensorboard_callback)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_path, "log.csv"))
    callbacks.append(csv_logger)

    if use_reduce_lr_platueau:
        # platueau_patience = int(callback_patience/2.)
        platueau_patience = 5
        print(f'using platueau and patience of {platueau_patience}')
        platueau_callvack = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=platueau_patience, 
                verbose=1, min_delta=0.0001, min_lr=1e-6,
            )
        callbacks.append(platueau_callvack)
        
    # try SWA here
    if use_swa:
        swa = SWA(start_epoch=swa_start_epochs, 
                lr_schedule='constant', 
                batch_size = batch_size,
                swa_lr=0.00005, 
                # swa_lr=1e-6, 
                verbose=1)
        callbacks.append(swa)
    
    # Train the model just to test deeplab
    # model.fit_generator(train_generator, epochs=100, validation_data=(val_X, val_y), steps_per_epoch=steps_per_epoch,
    #                     callbacks=callbacks, verbose=1)
    # model.fit(train_generator, epochs=100, validation_data=(val_X, val_y), steps_per_epoch=steps_per_epoch,
    #                     callbacks=callbacks, verbose=1)
    model.fit(train_generator, epochs=100, validation_data=val_dataset, 
        steps_per_epoch=steps_per_epoch,
        validation_steps=STEP_SIZE_VALID,
        callbacks=callbacks, verbose=1)
    
    if use_swa:
        model.save(os.path.join(run_path, "model_swa.hdf5"))


    # Print and save confusion matrix
    conf_matrix_flag = False
    print("Confusion matrix on the validation data")
    if conf_matrix_flag:
        conf_mat = model_utils.evaluate_model(model, val_X, val_y)
        with open(os.path.join(run_path, "conf_mat.txt"), "w+") as f:
            f.write(str(conf_mat))

    try:
        print("The current process uses the following amount of RAM (in GB) at its peak")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        print(resource.getpagesize())
    except Exception:
        print("Failed to print memory usage. This function was intended to run on a linux system.")




def evaluate_model_from_dir(model_path, val_data_folder_path, freeze, num_classes=5):
    tf.keras.backend.clear_session()
    
    # Validation data
    val = model_utils.load_dataset(val_data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    val_X = model_utils.fake_colors(val_X)
    val_y = model_utils.replace_class(val_y, class_id=5)

    # Model
    model = vgg16_unet(freeze=freeze, context_mode=False, num_classes=num_classes, dropout=0.0)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=[MeanIoU(num_classes=num_classes), 'accuracy'])

    loss, miou_tf, acc = model.evaluate(val_X, val_y, verbose=1)
    print(f'Untrained model acc: {acc}  loss: {loss} miou: {miou_tf}')
    
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(val_X, val_y, verbose=1)
    print(f'Trained model acc: {acc}  loss: {loss}')
    conf_mat = model_utils.evaluate_model(model, val_X, val_y, num_classes=num_classes)


if __name__ == '__main__':
    model_name = "vgg16"
    freeze = "2"
    image_augmentation = True
    context_mode = False
    run_path = "/workspace/runs"

    train_data_folder_path = r"/workspace/data/train"
    val_data_folder_path = r"/workspace/data/val"

    run(train_data_folder_path, val_data_folder_path, model_name=model_name, freeze=freeze,
        image_augmentation=image_augmentation, run_path=run_path ,context_mode=context_mode, replace_unknown=True)
