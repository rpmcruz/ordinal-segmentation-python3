from scipy.ndimage.morphology import distance_transform_edt
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import izip
import numpy as np
import copy
import cv2
import os

# Keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.preprocessing.image import ImageDataGenerator
from augmentation import MyImageDataGenerator


import utils

from keras.layers import advanced_activations as adactivations


def rescale_img(img):
    return img.astype(np.float) / 255.


def bg_to_mean(img):
    zero_mask = np.max(img, axis=2) < 10
    ret = img.copy()
    ret[zero_mask] = ret.mean(axis=(0, 1))

    ret = ret.astype(np.float)
    return ret / 255.


def ordinal_gen(generator, mapping):
    keys = np.arange(mapping.shape[0])[mapping > 0]
    values = mapping[mapping > 0]

    for batch in generator:
        next_ = np.concatenate([(batch >= class_).astype(np.float)
                                for class_ in keys],
                               axis=3)

        #yield {'output': next_, 'ord-reg': np.zeros((next_.shape[0],
                                                     #next_.shape[1],
                                                     #next_.shape[2],
                                                     #1))}

        yield {'output': next_, 'hor-monoticity': next_,
               'ver-monoticity': next_}


def distance_to_center(generator):
    for batch in generator:
        images = batch[0]
        masks = batch[1]
        #masks = masks == masks.max(axis=(1, 2, 3))

        dists = []
        for next_mask in masks:
            next_mask = np.max(next_mask) - next_mask
            dist = next_mask[:, :, 0]
            
            #print(np.unique(next_mask.ravel()))
            #next_mask = next_mask[:, :, 0]
            #next_mask = (next_mask == np.max(next_mask))

            #dist = np.ones(next_mask.shape)
            #dist[next_mask] = 0
            #dist = distance_transform_edt(dist.copy())
            #dist /= np.max(dist)

            dists.append(dist[:, :, np.newaxis])

        dists = np.asarray(dists)

        yield {'input': images, 'dists': dists}


def nominal_gen(generator, mapping):
    keys = np.arange(mapping.shape[0])[mapping != -1]
    values = mapping[mapping != -1]

    for batch in generator:
        next_ = np.concatenate([(batch == class_).astype(np.float)
                                for class_ in keys],
                               axis=3)

        #yield {'output': next_, 'ord-reg': np.zeros((next_.shape[0],
                                                     #next_.shape[1],
                                                     #next_.shape[2],
                                                     #1))}
        yield {'output': next_}


class SegmentationNetwork(BaseEstimator, TransformerMixin):
    def __init__(self,
                 conv_num=2,
                 conv_filter_num=32,
                 conv_filter_size=3,
                 conv_activation='relu',
                 conv_padding='same',
                 conv_consecutive=2,
                 pool_size=2,
                 pool_strides=None,
                 max_epochs=10,
                 img_size=(112, 112),
                 l2=0.001,
                 #loss=utils.worst_times_best,
                 #loss=utils.macro_dice_coef_loss,
                 #loss=utils.geometric_dice_loss,
                 loss=utils.average_crossentropy,
                 #loss=utils.worst_dice_coef_loss,
                 #loss=utils.macro_balanced_dice_coef_loss,
                 #loss=utils.worst_balanced_dice_coef_loss,
                 loss_aggregation='sum',
                 keras_filepath=os.path.join('output', 'models',
                                             'network.hdf5'),
                 img_preprocess=None,
                 mask_preprocess=None,
                 generator_batch_size=8,
                 multiple_out_streams=True,
                 pointwise_ordinal=True,
                 ordinal_output=True,
                 include_distances=False,
                 smoothness_weight=1e-4,
                 parallel_hyperplanes=False,
                 augment={},
                 ):
        self.img_size = img_size

        # Convolutional layers
        self.conv_num = conv_num
        self.conv_filter_num = conv_filter_num
        self.conv_filter_size = (conv_filter_size, conv_filter_size)
        self.conv_activation = conv_activation
        self.conv_padding = conv_padding
        self.conv_consecutive = conv_consecutive

        # Max-Pooling layers
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        # Loss function
        self.l2 = l2
        self.loss = loss
        self.loss_aggregation = loss_aggregation

        self.max_epochs = max_epochs
        self.keras_filepath = keras_filepath
        self.img_preprocess = img_preprocess
        self.mask_preprocess = mask_preprocess

        self.generator_batch_size = generator_batch_size

        self.multiple_out_streams = multiple_out_streams
        self.pointwise_ordinal = pointwise_ordinal
        self.ordinal_output = ordinal_output
        self.include_distances = include_distances
        self.smoothness_weight = smoothness_weight
        self.parallel_hyperplanes = parallel_hyperplanes

        self.augment = augment

    def fit_from_dir(self, path):
        self.ordinal_label_mapping(path)

        self.opt_network, self.network = self.build_network()
        #self.opt_network.load_weights(self.keras_filepath)
        #self.network.load_weights(self.keras_filepath)

        #print 'Summary', self.network.summary()

        train_gen = self.get_generator(path, 'train')
        val_gen = self.get_generator(path, 'validation')

        checkpoint = ModelCheckpoint(self.keras_filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto', period=1)
        early_stop = EarlyStopping(patience=100)

        if os.path.exists(self.keras_filepath):
            os.remove(self.keras_filepath)
        elif not os.path.exists(os.path.dirname(self.keras_filepath)):
            os.makedirs(os.path.dirname(self.keras_filepath))

        tr_steps = len(list(os.walk(os.path.join(path, 'train', 'imgs',
                                                 'seg')))[0][2]) / \
            self.generator_batch_size + 1
        val_steps = len(list(os.walk(os.path.join(path, 'validation', 'imgs',
                                                  'seg')))[0][2]) / \
             self.generator_batch_size + 1

        self.opt_network.fit_generator(train_gen,
                                       steps_per_epoch=tr_steps,
                                       epochs=self.max_epochs,
                                       verbose=2,
                                       callbacks=[early_stop, checkpoint],
                                       validation_data=val_gen,
                                       validation_steps=val_steps,
                                       )

        self.history = self.opt_network.history.history

        self.opt_network.load_weights(self.keras_filepath)
        self.network.load_weights(self.keras_filepath)

        return self

    def build_network(self, X=None, y=None):
        return None, None

    def predict(self, X):
        probs = self.transform(X) / 255.

        if self.ordinal_output:
            ret = np.zeros((probs.shape[0],
                            probs.shape[2], probs.shape[3]),
                           dtype=np.uint8)

            for class_ in range(self.num_labels):
                ret[probs[:, class_, :, :] >= 0.5] = class_ + 1

            """
            class_ = np.round((probs.sum(axis=1))).astype(np.int)
            print 'class_', class_.shape
            #print np.unique(class_.ravel())
            #print 'here', np.bincount(class_.ravel())
            for i in range(ret.shape[0]):
                ret[i] = np.zeros(class_[i].shape, dtype=np.uint8)
                for j in range(4):
                    ret[i][j][class_[i] == j] = 255
            #ret[class_] = 255
            ret = ret[:, 1:, :, :]
            ret = class_.astype(np.uint8)
            """

            #print(ret.shape)
            #print ret.sum(axis=1).shape
            #print np.unique(np.sum(ret, axis=1).ravel())
            #print np.unique(ret.ravel())
        else:
            ret = np.zeros(probs.shape, dtype=np.uint8)
            ret = probs.argmax(axis=1)
            ret = ret.astype(np.uint8)

        ret = [cv2.resize(p, x.shape[: 2][::-1],
                          interpolation=cv2.INTER_NEAREST)
               for p, x in zip(ret, X)]

        return ret

    def transform(self, X):
        X_rsz = [cv2.resize(x, self.img_size) for x in X]
        if self.img_preprocess is not None:
            X_rsz = [self.img_preprocess(x) for x in X]

        X_rsz = np.asarray(X_rsz).astype(np.float32)
        ret = self.network.predict(X_rsz)
        
        ret = np.squeeze(np.asarray(ret))
        ret = np.asarray([cv2.split(x) for x in ret])

        ret = np.asarray(ret) * 255
        print ret.shape
        return ret

    def load_weights(self):
        self.opt_network, self.network = self.build_network()
        try:
            self.opt_network.load_weights(self.keras_filepath)
            self.network.load_weights(self.keras_filepath)
        except:
            pass
        return self
    
    def get_generator(self, path, subset):
        params = self.augment

        #params = dict(
                      ##rotation_range=10.,
                      ##width_shift_range=0.1,
                      ##height_shift_range=0.1,
                      ##shear_range=0.1,
                      ##zoom_range=0.1,
                      ##fill_mode='reflect',
                      ##horizontal_flip=True,
                      #contrast_stretching_perc=0.4
                      #)

        params['preprocessing_function'] = self.img_preprocess
        img_gen = MyImageDataGenerator(**params)

        val_params = copy.deepcopy(params)
        val_params['preprocessing_function'] = self.mask_preprocess
        val_params['contrast_stretching_perc'] = 0.

        mask_gen = MyImageDataGenerator(**val_params)

        seed = 42

        img_gen_ = img_gen.flow_from_directory(
            os.path.join(path, subset, 'imgs'),
            target_size=self.img_size, class_mode=None, seed=seed,
            batch_size=self.generator_batch_size)

        mask_gen_ = mask_gen.flow_from_directory(
            os.path.join(path, subset, 'masks'),
            target_size=self.img_size, class_mode=None, seed=seed,
            color_mode='grayscale', batch_size=self.generator_batch_size,
            )

        out_gen_ = mask_gen.flow_from_directory(
            os.path.join(path, subset, 'masks'),
            target_size=self.img_size, class_mode=None, seed=seed,
            color_mode='grayscale', batch_size=self.generator_batch_size,
            )

        if self.include_distances:
            img_gen_ = distance_to_center(izip(img_gen_, mask_gen_))

        if self.ordinal_output:
            out_gen_ = ordinal_gen(out_gen_, self.labels_mapping)
        else:
            out_gen_ = nominal_gen(out_gen_, self.labels_mapping)

        return izip(img_gen_, out_gen_)

    def ordinal_label_mapping(self, path):
        lpath = os.path.join(path, 'train', 'masks', 'seg')
        labels = [set(cv2.imread(os.path.join(lpath, f), 0).ravel())
                             for f in list(os.walk(lpath))[0][2]]
        labels = reduce(lambda acc, n: acc | n, labels, set([]))

        self.labels_mapping = -np.ones(256, dtype=np.int)
        self.labels_mapping[sorted(list(labels))] = np.arange(len(labels))

        if self.ordinal_output:
            self.num_labels = sum(self.labels_mapping > 0)
        else:
            self.num_labels = sum(self.labels_mapping != -1)

        print self.num_labels
