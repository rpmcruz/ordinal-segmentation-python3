from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import cv2
import os


# Keras
from tensorflow.compat.v1.keras.layers import Conv2D, UpSampling2D, ZeroPadding2D
from tensorflow.compat.v1.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.compat.v1.keras.layers import MaxPooling2D
from tensorflow.compat.v1.keras.initializers import glorot_uniform
from tensorflow.compat.v1.keras.layers import Concatenate
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras.layers import Reshape
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Model 


from . import SegmentationNetwork
from .utils import dice_coef, dice_coef_loss


class EncoderDecoder(SegmentationNetwork):
   def build_network(self, X=None, y=None):
        shape = (self.img_size[0], self.img_size[1], 3)

        input_layer = Input(shape=shape, name='input')
        last_ = input_layer

        # Convolutional Encoding
        for conv in range(self.conv_num):
            nfilters = self.conv_filter_num * (1 + self.conv_filter_factor *
                                               conv)

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

        shape_conv_encoding = last_._keras_shape
        last_ = Flatten(name='flatten')(last_)

        # Dense transformations
        for d in range(self.dense_num):
            # Dense layer
            last_ = Dense(self.dense_midwidth,
                          activation=self.dense_midactivation,
                          kernel_regularizer=regularizers.l2(self.l2),
                          name='densemid-before%d' % d)(last_)
        
        last_ = Dense(self.dense_width,
                      activation=self.dense_activation,
                      kernel_regularizer=regularizers.l2(self.l2),
                      name='dense')(last_)

        for d in range(self.dense_num):
            # Dense layer
            last_ = Dense(self.dense_midwidth,
                          activation=self.dense_midactivation,
                          kernel_regularizer=regularizers.l2(self.l2),
                          name='densemid-after%d' % d)(last_)

        last_ = Dense(np.prod(shape_conv_encoding[1:]),
                      activation='relu')(last_)
        last_ = Reshape(shape_conv_encoding[1:])(last_)

        # Convolutional Decoding
        for conv in range(self.conv_num)[::-1]:
            nfilters = self.conv_filter_num * (1 + self.conv_filter_factor *
                                               conv)

            # Upsampling layer
            last_ = UpSampling2D(size=self.pool_size)(last_)

            # Convolutional layers
            for c in range(self.conv_consecutive):
                last_ = Conv2D(nfilters, self.conv_filter_size,
                               activation=self.conv_activation,
                               padding=self.conv_padding,
                               kernel_initializer=glorot_uniform(seed=42),
                               name='deconv%d-%d' % (conv, c))(last_)

        output_layer = Conv2D(1, self.conv_filter_size,
                              activation='sigmoid', padding='same',
                              kernel_initializer=glorot_uniform(seed=42),
                              name='output')(last_)
        self.final_shape = output_layer._keras_shape

        model = Model(inputs=input_layer,
                      output=[output_layer])
        model.compile(loss=[self.loss], metrics=[dice_coef],
                      optimizer='rmsprop')
        
        return model

