#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import keras
from keras.models import Model
from keras.layers import Input, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Cropping2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import concatenate, add, multiply
from easy_converter.keras_models.utility.base_model import BaseModel
from easy_converter.keras_models.utility.keras_model_name import KerasModelName
from easy_converter.keras_models.seg.instance_normalization import InstanceNormalization
import keras.backend as K


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


class FgSegNetV2(BaseModel):

    def __init__(self, lr=1e-4, img_shape=(512, 440, 3), vgg_weights_path=None):
        super().__init__()
        self.lr = lr
        self.img_shape = img_shape
        self.vgg_weights_path = vgg_weights_path
        self.set_name(KerasModelName.FgSegNetV2)

    def VGG16(self, x):

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)

        return x, a, b

    def decoder(self, x, a, b):
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        b = GlobalAveragePooling2D()(b)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x1 = multiply([x, b])
        x = add([x, x1])
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x2 = multiply([x, a])
        x = add([x, x2])
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        return x

    def M_FPM(self, x):

        pool = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)
        pool = Conv2D(64, (1, 1), padding='same')(pool)

        d1 = Conv2D(64, (3, 3), padding='same')(x)

        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)

        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)

        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=16)(y)

        x = concatenate([pool, d1, d4, d8, d16], axis=-1)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout2D(0.25)(x)
        return x

    def init_model(self):
        assert len(self.img_shape) == 3
        h, w, d = self.img_shape

        net_input = Input(shape=(h, w, d), name='net_input')
        vgg_output = self.VGG16(net_input)
        model = Model(inputs=net_input, outputs=vgg_output, name='model')
        model.load_weights(self.vgg_weights_path, by_name=True)

        unfreeze_layers = ['block4_conv1', 'block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if (layer.name not in unfreeze_layers):
                layer.trainable = False

        x, a, b = model.output

        x = self.M_FPM(x)
        x = self.decoder(x, a, b)

        vision_model = Model(inputs=net_input, outputs=x, name=KerasModelName.FgSegNetV2)
        opt = keras.optimizers.RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        vision_model.compile(loss=loss, optimizer=opt, metrics=[acc])
        return vision_model
