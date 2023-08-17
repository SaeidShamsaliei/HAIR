import tensorflow as tf
import numpy as np
import os
import datetime
import model_utils
import time
import resource
import deepv3_plus.model as deeplab_model
from model import DeeplabV3Plus
import json
from swa.tfkeras import SWA
from ImageDataAugmentor.image_data_augmentor import *
import albumentations
import kerastuner as kt

from utils.model_utils import model_by_name
# import matplotlib.pyplot as plt

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


def model_compiled_tune(hp, num_classes=7, batch_size=16, regularizer_flag_tuner=True):
    """
    Return a model
    but batch size and num_classes needs to be hardcoded
    :return: a compiled model
    """

    # hp_dropout = hp.Choice("dropout", values=[0.0, 0.1, 0.2])
    # for deeplab v3+
    if model_name_tuner == 'deeplabv3_resnet50' or model_name_tuner == 'swin_transformer' or model_name_tuner == 'hrnet':
        hp_dropout = 0.0
    else:
        hp_dropout = hp.Choice("dropout", values=[0.0, 0.1, 0.2])

    hp_learning_rate = hp.Choice("learning_rate", values=[0.0001, 0.001, 0.00001, 0.0005, 0.005, 0.00005])

    model = model_by_name(model_name=model_name_tuner, 
            num_classes=num_classes, 
            dropout=hp_dropout, 
            batch_size=batch_size,)

    print('lr is fixed')
    if model_name_tuner == 'swin_transformer':
        opt = tf.keras.optimizers.Adam(
            hp_learning_rate, clipvalue=0.5)
    else:
        opt = tf.keras.optimizers.Adam(
            hp_learning_rate)

    # lr_metric = get_lr_metric(opt)

    if regularizer_flag_tuner:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    if (model_name_tuner.find('unet') != -1) or (model_name_tuner.find('fpn') != -1):
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
        metrics=["accuracy"])
    
    return model


def vgg16_unet_tune(hp, image_size=512, n_max_filters=512, freeze="0", context_mode=False, 
                    num_classes=6):
    """
    A unet model that uses a pre-trained VGG16 CNN as the encoder part.
    :param num_classes: The number of classes
    :param image_size: The size of the input images
    :param n_max_filters: The number of filters at the bottom layer of the unet model.
    :param freeze: Specifies what layers to freeze during training. The frozen layers will not be trained.
                all: all of the VGG16 layers are frozen. first: all but the last conv block of VGG16 is frozen.
                none: no layers are frozen. number: freeze all conv blocks upto and including the number.
    :param hp: hyper param tuning
    :return: A keras model
    """

    hp_dropout = hp.Choice("dropout", values=[0.0, 0.1, 0.2])
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

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
        if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)), kernel_size=(3, 3), strides=2,
                                            padding="same", activation="relu")(x)  # Upsample
        x = tf.concat([x, skip_connections[skip_i]], axis=-1)  # Add skip connection to the channels

    # Last conv layers
    if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if context_mode:
        # Crop to only predict on the middle pixels
        x = tf.keras.layers.MaxPool2D()(x)
    if hp_dropout > 0: x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model


def tune_from_dir_all(train_data_folder_path, val_data_folder_path, 
                model_name, run_path, batch_size=1, train_image_directory_name='images', 
                train_label_directory_name='labels', val_image_directory_name='images', 
                val_label_directory_name='labels', num_classes=6, conf_exp=None, 
                use_swa=False, callback_patience=10, swa_start_epochs=10, 
                use_albumentation=False, img_per_epoch=9353, regularizer_flag=False, 
                use_grayscale_only=False, adam_schedule=False,
                class_weight=False, use_reduce_lr_platueau=False, seed_val=2, 
                hp_max_epochs=20, hp_factor=3, hp_hyperband_iterations=10):
    """"
    Train the model and fine tune the hp, which is LR in this case.
    """
    global model_name_tuner
    model_name_tuner = model_name    

    global regularizer_flag_tuner
    regularizer_flag_tuner = regularizer_flag

    global class_weight_tuner
    class_weight_tuner = class_weight

    freeze = 99
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
    # use album v3
    else:
        print('\n\nusing albumentations v3\n\n')
        AUGMENTATIONS = albumentations.Compose([
        # pixel level aumentation
            albumentations.Transpose(p=0.1),
            albumentations.Flip(p=0.1),
        ])

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
    
    train_generator = (pair for pair in zip(image_generator, mask_generator))
    
    if class_weight:
        print('applying class weight')
        train_generator = balanced_flow_from_directory(train_generator)

    # validation 
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
    
    tuner = kt.Hyperband(model_compiled_tune,
                     objective="val_loss",
                     max_epochs=hp_max_epochs,
                     factor=hp_factor,
                     hyperband_iterations=hp_hyperband_iterations,
                     directory=run_path,
                     project_name=f"new_project_{start_time}",)
    


    tuner.search_space_summary()
    tuner.search(train_generator, epochs=20, validation_data=val_dataset, callbacks=callbacks, 
                    verbose=1, steps_per_epoch=steps_per_epoch, validation_steps=STEP_SIZE_VALID,)
    # Get the optimal hyperparameters from the results
    best_hps=tuner.get_best_hyperparameters()[0]
    print('---------best hyperparam---------------')
    print(best_hps)
    print('---------------------------------------')

    # Build model
    h_model = tuner.hypermodel.build(best_hps)
    
    h_model.fit(train_generator, epochs=100, validation_data=(val_X, val_y), 
        steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=1)


def tune_from_dir(train_data_folder_path, val_data_folder_path, model_name="vgg16", 
                 # freeze="all",
                 run_path="/home/kitkat/PycharmProjects/river-segmentation/runs", batch_size=1, 
                 train_image_directory_name='images', train_label_directory_name='labels',
                 val_image_directory_name='images', val_label_directory_name='labels', 
                 num_classes=6, 
                 conf_exp=None, 
                 use_swa=False, callback_patience=10, swa_start_epochs=10, use_albumentation=False, img_per_epoch=9353):
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
    freeze='tiner'
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
        
        image_generator = image_datagen.flow_from_directory(train_img_path,
                                                            class_mode=None, target_size=(512, 512), seed=1, batch_size=batch_size)
        mask_generator = mask_datagen.flow_from_directory(train_lbl_path,
                                                        class_mode=None, target_size=(512, 512), seed=1, batch_size=batch_size,
                                                        color_mode="grayscale")
    
    # albumentation data loader part
    # ------------------------------------------------------
    else:
        seed_val = 0
        print('\n\nusing albumentations\n\n')
        AUGMENTATIONS = albumentations.Compose([
        # pixel level aumentation
            albumentations.Transpose(p=0.2),
            albumentations.Flip(p=0.2),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
            ],p=0.2),
            
            # bluring filters
        #     albumentations.GaussianBlur(p=0.05),
            albumentations.OneOf([
                albumentations.MedianBlur(),
                albumentations.RandomGamma(),
                albumentations.MotionBlur(),
            ],p=0.2),
            
            # will be added
            albumentations.GridDistortion(p=0.1),
            albumentations.ImageCompression(p=0.1),
            albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.1),
        ])

        image_datagen = ImageDataAugmentor(
            input_augment_mode='image',
                augment=AUGMENTATIONS,
                preprocess_input=lambda x: x/(2**8 -1),
                seed=seed_val)

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

    #000000000000000000000000000000000000000000000000Arild part0000000000000000000000000000000000000000000
    # Validation data
    # val = model_utils.load_dataset(val_data_folder_path)
    # val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    # val_X = model_utils.fake_colors(val_X)
    # val_y = model_utils.replace_class(val_y, class_id=5)

    #0000000000000000000000000000000000000000000000000My code000000000000000000000000000000000000000000000

    print('loading val images')
    val_X = model_utils.load_png_to_np_normalized(img_path=val_data_folder_path, 
                     image_name=val_image_directory_name, normalized=True)
    
    # repeat channels if necessary
    if val_X.shape[-1] != 3:
        print('add fake channels for validation')
        val_X = model_utils.fake_colors(val_X)

    print('loading val label')
    val_y = model_utils.load_png_to_np_normalized(img_path=val_data_folder_path, 
                     image_name=val_label_directory_name, normalized=False)
    
    #TODO (can be changed with ignore)
    if num_classes == 5:
        print('replace classes')
        val_y = model_utils.replace_class(val_y, class_id=5)
        print('done')

    # Load and compile model
    if model_name.lower() == "vgg16":
        # model = vgg16_unet(freeze=freeze, context_mode=False,
        #                    num_classes=num_classes, dropout=dropout)
        print('vgg16 will be tuned')
    
    elif model_name.lower() == 'deeplabv3':
        raise NotImplementedError('only vgg is availabel for tuning')

    elif model_name.lower() == "deeplabv3_resnet50":
        raise NotImplementedError('only vgg is availabel for tuning')
    
    else:
        model = None

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=callback_patience, monitor="val_loss"))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, "model.hdf5"),
                                                    monitor="val_loss", save_best_only=True)
    callbacks.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_path, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_path, "log.csv"))
    callbacks.append(csv_logger)

    # try SWA here
    if use_swa:
        swa = SWA(start_epoch=swa_start_epochs, 
                lr_schedule='constant', 
                batch_size = batch_size,
                swa_lr=0.00005, 
                verbose=1)
        callbacks.append(swa)
    
    # Train the model just to test deeplab
    # model.fit_generator(train_generator, epochs=100, validation_data=(val_X, val_y), steps_per_epoch=steps_per_epoch,
    #                     callbacks=callbacks, verbose=1)
    tuner = kt.Hyperband(vgg16_unet_tune,
                     objective="val_accuracy",
                     max_epochs=20,
                     factor=3,
                     hyperband_iterations=10,
                     directory="kt_dir",
                     project_name="kt_hyperband",)
    
    tuner.search_space_summary()
    tuner.search(train_generator, epochs=20, validation_data=(val_X, val_y), callbacks=callbacks, 
                    verbose=1, steps_per_epoch=steps_per_epoch)
    # Get the optimal hyperparameters from the results
    best_hps=tuner.get_best_hyperparameters()[0]
    print('---------best hyperparam---------------')
    print(best_hps)
    print('---------------------------------------')
    # Build model
    h_model = tuner.hypermodel.build(best_hps)
    

    h_model.fit(train_generator, epochs=100, validation_data=(val_X, val_y), steps_per_epoch=steps_per_epoch,
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
    model = vgg16_unet_tune(freeze=freeze, context_mode=False, num_classes=num_classes, dropout=0.0)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=[MeanIoU(num_classes=num_classes), 'accuracy'])

    loss, miou_tf, acc = model.evaluate(val_X, val_y, verbose=1)
    print(f'Untrained model acc: {acc}  loss: {loss} miou: {miou_tf}')
    
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(val_X, val_y, verbose=1)
    print(f'Trained model acc: {acc}  loss: {loss}')
    conf_mat = model_utils.evaluate_model(model, val_X, val_y, num_classes=num_classes)


if __name__ == '__main__':
    print('main')