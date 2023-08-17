# DIResUNet
import tensorflow as tf

import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import tensorflow as tf

def dilconv(x,d,f):
  x1 = Conv2D(f,(1,1),dilation_rate=d)(x)
  x2 = Conv2D(f,(3,3),dilation_rate=d, padding='same')(x1)
  return x2

def dgspp(x,f):
  x1 = dilconv(x,7,f)
  x11 = x1+x
  x2 = dilconv(x11,14,f)
  x22 = x2+x1+x
  x3 = dilconv(x,21,f)
  x33 = x3+x2+x1+x
  # x4 = tf.math.reduce_mean(x, axis=None)
  x4 = tf.math.reduce_mean(x, axis=1, keepdims=True, name=None)
  x4 = tf.math.reduce_mean(x4, axis=2, keepdims=True, name=None)
  x4 = Conv2D(f,(1,1))(x4)
  x4 = tf.keras.layers.UpSampling2D(size=(128,128), interpolation='bilinear')(x4)
  x = concatenate([x4,x33],axis=3)
  return x

from tensorflow.keras import layers
from tensorflow import keras

img_size = (512,512)

def downblock(filters, filter_size, previous_layer):
  x = layers.Conv2D(filters, filter_size, padding="same")(previous_layer)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  x = layers.Conv2D(filters, filter_size, padding="same")(x)

  
  residual = layers.Conv2D(filters, 1, padding="same")(previous_layer)      #separate layer for addintion
  x = layers.add([x, residual])  # Add back residual
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  p = layers.MaxPooling2D(2)(x)

  return x,p

def bottleneck(filters, filter_size, previous_layer):
  x = layers.Conv2D(filters, filter_size, padding="same")(previous_layer)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(filters, filter_size, padding="same")(x)

  residual = layers.Conv2D(filters, 1, padding="same")(previous_layer)      #separate layer for addintion
  x = layers.add([x, residual])  # Add back residual
  
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  return x

def upblock(filters, filter_size, previous_layer, layer_to_concat):
  x = layers.Conv2DTranspose(filters, filter_size, strides=2, padding="same")(previous_layer)       #upconvolution
  concat = layers.concatenate([x, layer_to_concat])                                                      #concatenation

  x = layers.Conv2D(filters, filter_size, padding="same")(concat)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)
  x = layers.Conv2D(filters, filter_size, padding="same")(x)
  x = layers.BatchNormalization()(x)

  residual = layers.Conv2D(filters, 1, padding="same")(concat)      #separate layer for addintion
  x = layers.add([x, residual])  # Add back residual
  
  x = layers.Activation("relu")(x)

  return x

def DIResUNet(input_shape=(512, 512, 3), num_class=6):
    input_layer = layers.Input(shape = input_shape)

    conv1, pool1 = downblock(32, 3, input_layer)
    conv2, pool2 = downblock(64, 3, pool1)
    conv3, pool3 = downblock(128, 3, pool2)
    conv3  = dgspp(conv3,128)
    tower_1 = Conv2D(256, (1,1), padding='same', activation='relu')(pool3)
    tower_1 = Conv2D(256, (3,3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(256, (1,1), padding='same', activation='relu')(pool3)
    tower_2 = Conv2D(256, (5,5), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(pool3)
    tower_3 = Conv2D(256, (1,1), padding='same', activation='relu')(tower_3)
    conv4 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)

    upconv1 = upblock(128, 3, conv4, conv3)
    upconv2 = upblock(64, 3, upconv1, conv2)
    upconv3 = upblock(32, 3, upconv2, conv1)

    output_layer = layers.Conv2D(num_class, 1, padding="same", activation='softmax')(upconv3)
    model = keras.Model(input_layer, output_layer)
    return model