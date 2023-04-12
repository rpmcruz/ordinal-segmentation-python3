from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import copy
import math
import cv2
import os


# Keras
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.compat.v1.keras.layers import Reshape
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.layers import PReLU
from tensorflow.compat.v1.keras.layers import MaxPooling2D
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.initializers import glorot_uniform
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Concatenate
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Model


from .utils import dice_coef, dice_coef_loss
from . import SegmentationNetwork, bg_to_mean, rescale_img


class FlatEncoderDecoder(SegmentationNetwork):
    def __init__(self,
                 conv_num=3,
                 conv_filter_num=32,
                 conv_filter_factor=0,
                 conv_filter_size=3,
                 conv_activation='relu',
                 conv_padding='same',
                 conv_consecutive=2,
                 pool_size=2,
                 pool_strides=None,
                 dense_num=1,
                 dense_width=1024,
                 dense_activation='sigmoid',
                 dropout=True,
                 dropout_rate=0.05,
                 output_activation='sigmoid',
                 output_scale_factor=1.,
                 generator_batch_size=32,
                 max_epochs=100,
                 img_size=(75, 75),
                 l2=0.0001,
                 keras_filepath='output/models/flat.hdf5',
                 loss='binary_crossentropy',
                 img_preprocess=bg_to_mean,
                 mask_preprocess=rescale_img,
                 ):
        super(FlatEncoderDecoder,
              self).__init__(conv_num=conv_num,
                             conv_filter_num=conv_filter_num,
                             conv_filter_factor=conv_filter_factor,
                             conv_filter_size=conv_filter_size,
                             conv_activation=conv_activation,
                             conv_padding=conv_padding,
                             conv_consecutive=conv_consecutive,
                             pool_size=pool_size,
                             pool_strides=pool_strides,
                             dense_num=dense_num,
                             dense_width=dense_width,
                             dense_activation=dense_activation,
                             dropout=dropout,
                             dropout_rate=dropout_rate,
                             generator_batch_size=generator_batch_size,
                             max_epochs=max_epochs,
                             img_size=img_size,
                             l2=l2,
                             loss=loss,
                             keras_filepath=keras_filepath,
                             img_preprocess=bg_to_mean,
                             mask_preprocess=rescale_img)

        self.output_activation='sigmoid'
        self.output_scale_factor=output_scale_factor
        self.output_size = np.prod((np.asarray(self.img_size[: 2]) *
                                    self.output_scale_factor).astype(np.int))

    def build_network(self, X=None, y=None):
        shape = (self.img_size[0], self.img_size[1], 3)

        input_layer = Input(shape=shape, name='input')
        last_ = input_layer

        # Convolutional section
        for conv in range(self.conv_num):
            nfilters = self.conv_filter_num * 2 ** conv

            # Convolutional layers
            for c in range(self.conv_consecutive):
                last_ = Conv2D(nfilters, self.conv_filter_size,
                               activation=self.conv_activation,
                               padding=self.conv_padding,
                               kernel_initializer=glorot_uniform(seed=42),
                               name='conv%d-%d' % (conv, c))(last_)

            # Pooling layer
            last_ = MaxPooling2D(pool_size=self.pool_size,
                                 strides=self.pool_strides,
                                 name='pool%d' % conv)(last_)

        last_ = Flatten(name='flatten')(last_)

        # Dense section
        for dense in range(self.dense_num):
            # Dense layer
            last_ = Dense(self.dense_width,
                          activation=self.dense_activation,
                          kernel_regularizer=regularizers.l2(self.l2),
                          name='dense%d' % dense)(last_)

            # Dropout layer
            if self.dropout:
                last_ = Dropout(rate=self.dropout_rate,
                                name='dropout%d' % dense,
                                seed=42)(last_)

        # Output layer
        output_layer = Dense(self.output_size,
                             activation=self.output_activation)(last_)

        out_shape = (np.asarray(self.img_size) *
                     self.output_scale_factor).astype(np.int)
        out_shape = list(out_shape) + [1]

        output_layer = Reshape(out_shape)(output_layer)
        self.final_shape = [None] + out_shape

        model = Model(inputs=input_layer,
                      output=[output_layer])
        model.compile(loss=[self.loss], metrics=[dice_coef],
                      optimizer='adadelta')

        return model

    def transfer_weights(self):
        if self.transfer_from is not None:
            path = self.keras_filepath
            self.keras_filepath = self.transfer_from
            self.load_weights()
            self.keras_filepath = path

        return self
        
    def load_weights(self):
        self.network = self.build_network()
        self.network.load_weights(self.keras_filepath)
        return self

