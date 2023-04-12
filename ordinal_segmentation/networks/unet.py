from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import cv2
import os


# Keras
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D, UpSampling2D, \
    Conv2DTranspose, SeparableConv2D, Conv3D
from tensorflow.compat.v1.keras.layers import Input, Dense, Dropout, Flatten, Cropping2D
from tensorflow.compat.v1.keras.layers import Activation, Lambda, ActivityRegularization
from tensorflow.compat.v1.keras.layers import MaxPooling2D
from tensorflow.compat.v1.keras.initializers import glorot_uniform
from tensorflow.compat.v1.keras.layers import Concatenate, Maximum, Add, Multiply
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.compat.v1.keras.layers import Reshape
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import Model


from . import SegmentationNetwork
from .utils import dice_coef, dice_coef_loss, worst_dice_coef, \
    macro_dice_coef, best_dice_coef, dice_coef_class


def mean_value_loss(y_true, y_pred):
    return K.mean(K.flatten(y_pred))


class OrdinalUNet(SegmentationNetwork):
   def build_network(self, X=None, y=None):
        shape = (self.img_size[0], self.img_size[1], 3)

        input_layer = Input(shape=shape, name='input')
        last_ = input_layer

        # Convolutional section
        last_conv_per_level = []
        sizes = []
        for conv_level in range(self.conv_num + 1):
            nfilters = self.conv_filter_num * 2 ** conv_level
            sizes.append(nfilters)

            # Convolutional layers
            for c in range(self.conv_consecutive):
                #cfs = max(2, self.conv_filter_size[0] - conv_level)
                #cfs = (cfs, cfs)
                cfs = self.conv_filter_size

                last_ = Conv2D(nfilters, cfs,
                               activation=self.conv_activation,
                               padding=self.conv_padding,
                               kernel_initializer=glorot_uniform(seed=42),
                               name='conv%d-%d' % (conv_level, c))(last_)

            last_conv_per_level.append(last_)

            # Pooling layer
            if conv_level != self.conv_num:
                last_ = MaxPooling2D(pool_size=self.pool_size,
                                     strides=self.pool_strides,
                                     name='pool%d' % conv_level)(last_)

        last_per_label = [None]
        if self.multiple_out_streams:
            last_before = last_
            for d in range(1, self.num_labels + 1):
                last_ = last_before
                # Deconvolutional section
                for conv_level in range(self.conv_num)[::-1]:
                    # FIXME
                    #cc = Concatenate(axis=3)
                    cc = Add()
                    last_ = cc(
                            [Conv2DTranspose(sizes[conv_level],
                                self.pool_size,
                                strides=(self.pool_size, self.pool_size),
                                )(last_),
                            last_conv_per_level[conv_level]])

                    for c in range(self.conv_consecutive):
                        glorot = glorot_uniform(seed=42)
                        last_ = Conv2D(sizes[conv_level],
                                       self.conv_filter_size,
                                       activation=self.conv_activation,
                                       kernel_initializer=glorot,
                                       padding=self.conv_padding)(last_)
                last_per_label.append(last_)
        else:
            # Deconvolutional section
            for conv_level in range(self.conv_num)[::-1]:
                # FIXME
                #cc = Concatenate(axis=3)
                cc = Add()
                last_ = cc(
                        [Conv2DTranspose(sizes[conv_level],
                            self.pool_size,
                            strides=(self.pool_size, self.pool_size),
                            )(last_),
                        last_conv_per_level[conv_level]])

                for c in range(self.conv_consecutive):
                    #cfs = max(2, self.conv_filter_size[0] - conv_level)
                    #cfs = (cfs, cfs)
                    cfs = self.conv_filter_size

                    last_ = Conv2D(sizes[conv_level], cfs,
                                   activation=self.conv_activation,
                                   kernel_initializer=glorot_uniform(seed=42),
                                   padding=self.conv_padding)(last_)

            for d in range(1, self.num_labels + 1):
                last_per_label.append(last_)

        outputs = []
        scoring_outputs = []
        prob_outputs = []

        reverse_layer = Lambda(lambda x: -x)
        minimum_layer = \
            lambda x, y: reverse_layer(Maximum()([reverse_layer(x),
                                                  reverse_layer(y)]))

        scoring = Conv2D(1, (1, 1), activation='linear')(last_per_label[-1])

        for d in range(1 , self.num_labels + 1):
            if self.pointwise_ordinal:
                if d > 1:
                    if self.parallel_hyperplanes:
                        next_output = Conv2D(1, (1, 1), activation='linear')(
                            scoring)
                    else:
                        next_output = Conv2D(1, (1, 1), activation='linear')(
                            last_per_label[d])

                    next_output = Activation(self.final_act)(next_output)  # Ricardo

                    scoring_outputs.append(next_output)
                    prob_outputs.append(next_output)

                    if self.pointwise_ordinal == 'min':
                        next_output = minimum_layer(outputs[-1], next_output)
                    elif self.pointwise_ordinal == 'mul':
                        next_output = Multiply()([outputs[-1], next_output])
                else:
                    if self.parallel_hyperplanes:
                        next_output = Conv2D(1, (1, 1), activation='linear')(
                            scoring)
                    else:
                        next_output = Conv2D(1, (1, 1), activation='linear')(
                            last_per_label[d])

    
                    next_output = Activation(self.final_act,
                                            name='sigmoid%d' % d)(next_output)  # Ricardo

                    scoring_outputs.append(next_output)
                    prob_outputs.append(next_output)
            else:
                next_output = Conv2D(1, (1, 1), activation='linear',
                                     name='out%d' % d)(last_per_label[d])

                next_output = Activation(self.final_act,
                                         name='sigmoid%d' % d)(next_output)  # Ricardo

                scoring_outputs.append(next_output)
                prob_outputs.append(next_output)

            outputs.append(next_output)

        out = Concatenate(axis=-1, name='output')(outputs)

        subtract = lambda x, y: Add()([x, reverse_layer(y)])
        elem_hinge = lambda x: Lambda(lambda k: K.maximum(k, 0.))(x)
        sign = lambda x: Lambda(lambda k: K.sign(k))(x)
        
        hor_monoticity_output = None
        ver_monoticity_output = None
        
        if self.include_distances:
            dshape = (self.img_size[0], self.img_size[1], 1)
            distance_input = Input(shape=dshape, name='dists')

            w = 5
            left_crop = Cropping2D(((0, 0), (w, 0)))
            right_crop = Cropping2D(((0, 0), (0, w)))

            top_crop = Cropping2D(((w, 0), (0, 0)))
            bottom_crop = Cropping2D(((0, w), (0, 0)))

            left_crop_dists = left_crop(distance_input)
            right_crop_dists = right_crop(distance_input)
            top_crop_dists = top_crop(distance_input)
            bottom_crop_dists = bottom_crop(distance_input)

            horizontal_dist_diff = sign(subtract(left_crop_dists,
                                                 right_crop_dists))
            vertical_dist_diff = sign(subtract(top_crop_dists,
                                               bottom_crop_dists))

            hor_monoticity_outputs = []
            ver_monoticity_outputs = []
            for p in scoring_outputs:
                left_crop_probs = left_crop(p)
                right_crop_probs = right_crop(p)
                top_crop_probs = top_crop(p)
                bottom_crop_probs = bottom_crop(p)

                horizontal_prob_diff = subtract(left_crop_probs,
                                                right_crop_probs)
                horizontal_monoticity = Multiply()([horizontal_dist_diff,
                                                    horizontal_prob_diff])

                vertical_prob_diff = subtract(top_crop_probs,
                                              bottom_crop_probs)
                vertical_monoticity = Multiply()([vertical_dist_diff,
                                                  vertical_prob_diff])

                horizontal_monoticity = elem_hinge(horizontal_monoticity)
                vertical_monoticity = elem_hinge(vertical_monoticity)

                hor_monoticity_outputs.append(horizontal_monoticity)
                ver_monoticity_outputs.append(vertical_monoticity)

            hor_monoticity_output = \
                Add(name='hor-monoticity')(hor_monoticity_outputs)
            ver_monoticity_output = \
                Add(name='ver-monoticity')(ver_monoticity_outputs)

        print(len(prob_outputs))

        """
        if self.num_labels > 2:
            ord_regs = []
            for d in range(self.num_labels - 1):
                ordinal_diff = Add()([prob_outputs[d + 1],
                                    Lambda(lambda x: -x)(prob_outputs[d])])
                ordinal_reg = Lambda(lambda x: K.maximum(x, 0.))(ordinal_diff)
                ord_regs.append(ordinal_reg)

            ord_regs = Add(name='ord-reg')(ord_regs)
        else:
            ordinal_diff = Add()([prob_outputs[1],
                                  Lambda(lambda x: -x)(prob_outputs[0])])
            ord_regs = Lambda(lambda x: K.maximum(x, 0.),
                              name='ord-reg')(ordinal_diff)
        """

        model_inputs = [input_layer]
        model_outputs = [out]
        loss = {'output': self.loss}
        loss_weights = {'output': 1.}

        if self.include_distances:
            smoothness_weight = 0.5 * self.smoothness_weight

            model_inputs.append(distance_input)
            model_outputs.append(hor_monoticity_output)
            model_outputs.append(ver_monoticity_output)
            loss['hor-monoticity'] = mean_value_loss
            loss['ver-monoticity'] = mean_value_loss
            loss_weights['hor-monoticity'] = smoothness_weight
            loss_weights['ver-monoticity'] = smoothness_weight

        model = Model(inputs=[input_layer], outputs=[out])

        opt_model = Model(inputs=model_inputs, outputs=model_outputs)
        opt_model.compile('adadelta',
                          loss=loss,
                          loss_weights=loss_weights,
                          metrics={'output': [worst_dice_coef, best_dice_coef],
                                   },
                          )

        self.final_shape = out.shape  # Ricardo: _keras_shape no longer exists, I think it's the same as shape ..

        return opt_model, model
