from scipy.ndimage.morphology import distance_transform_edt
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import cv2
import os


from networks.unet import OrdinalUNet
from networks import utils


def mask_dilate(img, factor=1.):
    ret = img.copy()
    max_lesion_dist = np.max(distance_transform_edt(ret != 0).ravel())
    dist = distance_transform_edt(1. - (ret != 0))

    dist[dist > factor * max_lesion_dist] = 0.
    dist /= np.max(dist.ravel()) + 1
    dist = 1 - dist
    dist[dist == 1] = 0.
    ret = 255. * dist
    ret[img != 0] = 255

    return ret.astype(np.float)


def my_logloss(tr_masks, tr_preds):
    def logloss(m, p):
        m_ = m.ravel() > 10
        if np.sum(m_) == 0:
            return None

        p_ = p.ravel() / 255.
        eps = 1e-5
        p_[0] = 0.
        p_ = eps + (1 - eps) * p_
        return log_loss(m_, p_, labels=[False, True])

    ret = [logloss(m, p) for m, p in zip(tr_masks, tr_preds)]
    ret = [x for x in ret if x is not None]

    if len(ret) == 0:
        return 0.
    return np.mean(ret)


def get_args():
    parser = argparse.ArgumentParser(
                        description="Segmentation tasks.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', metavar="D", nargs='?',
                        default='', help='Path to the dataset configuration')
    parser.add_argument('--first', metavar="F", type=int,
                        default=0, help='First fold')
    parser.add_argument('--last', metavar="L", type=int,
                        default=1, help='Last fold')

    return parser.parse_args()


def load_set(config, fold, subset):
    test_imgs = []
    test_masks = []
    base_path = os.path.join(config['path'], 'fold%d' % fold, subset, 'imgs')

    for label_ in list(os.walk(base_path))[0][1]:
        for img_filename in list(os.walk(os.path.join(base_path,
                                                      label_)))[0][2]:
            img = cv2.imread(os.path.join(base_path, label_, img_filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            format_ = '.' + img_filename.split('.')[-1]
            mask = cv2.imread(os.path.join(config['path'],
                                           'fold%d' % fold, subset,
                                           'masks', label_,
                                           img_filename.replace(format_,
                                                                '.bmp')), 0)

            test_imgs.append(img)
            test_masks.append(mask)

    return test_imgs, test_masks


DICE_SMOOTH = 1.

def dice_coef(y_true, y_pred):
    def dice_score(y, p):
        values = []
        for l in set(y.ravel()) | set(p.ravel()):
            y_ = (y.ravel() == l).astype(np.float)
            p_ = (p.ravel() == l).astype(np.float)

            intersection = np.sum(y_ * p_)
            values.append((2. * intersection + DICE_SMOOTH) / \
                          (np.sum(y_) + np.sum(p_) + DICE_SMOOTH))

        return np.mean(values)

    return np.mean([dice_score(y, p)
                    for y, p in zip(map_to_ordinal(y_true), y_pred)])


def dice_at_boundary(y_true, y_pred, bandwidth=5):
    def dice_score(y, p, mask):
        values = []
        for l in set(y.ravel()) | set(p.ravel()):
            y_ = (y.ravel() == l).astype(np.float)
            p_ = (p.ravel() == l).astype(np.float)
            m_ = mask.ravel() != 0
            
            y_ = y_[m_]
            p_ = p_[m_]

            intersection = np.sum(y_ * p_)
            values.append((2. * intersection + DICE_SMOOTH) / \
                          (np.sum(y_) + np.sum(p_) + DICE_SMOOTH))

        return np.mean(values)

    def boundary_mask(y):
        ret = np.zeros(y.shape, dtype=np.uint8)
        ret[:, 1::] = (ret[:, 1:]) | (y[:, :-1] != (y[:, 1:]))
        ret[1::, :] = (ret[1:, :]) | (y[:-1, :] != (y[1:, :]))

        ret = distance_transform_edt(1 - ret) <= bandwidth
        return ret

    return np.mean([dice_score(y, p, boundary_mask(yo))
                    for yo, y, p in zip(y_true, map_to_ordinal(y_true),
                                        y_pred)])


def map_to_ordinal(y_true):
    values = reduce(lambda ac, n: ac | set(n.ravel()), y_true, set([]))
    values = np.asarray(sorted(list(values)))
    mapping = np.zeros(256, dtype=np.uint8)
    mapping[values] = np.arange(len(values))
    
    return [mapping[y] for y in y_true]


def nonordinal_error(y_true, y_pred):
    def error(y, p):
        y_ = y.ravel()
        p_ = p.ravel()

        return np.mean(np.abs(y_ - p_) > 1)

    return np.mean([error(y, p)
                    for y, p in zip(map_to_ordinal(y_true), y_pred)])


def accuracy(y_true, y_pred):
    def acc_score(y, p):
        y_ = y.ravel()
        p_ = p.ravel()

        return np.mean(y_ == p_)

    return np.mean([acc_score(y, p)
                    for y, p in zip(map_to_ordinal(y_true), y_pred)])


def ordinal_accuracy(y_true, y_pred):
    def acc_score(y, p):
        y_ = y.ravel()
        p_ = p.ravel()
        return np.mean(y_ <= p_)

    return np.mean([acc_score(y, p)
                    for y, p in zip(map_to_ordinal(y_true), y_pred)])


def accuracy_at_boundary(y_true, y_pred, bandwidth=5):
    def acc_score(y, p, mask):
        y_ = y.ravel()
        p_ = p.ravel()
        m_ = mask.ravel()
        return np.mean((y_ == p_)[m_ != 0])

    def boundary_mask(y):
        ret = np.zeros(y.shape, dtype=np.uint8)
        ret[:, 1::] = (ret[:, 1:]) | (y[:, :-1] != (y[:, 1:]))
        ret[1::, :] = (ret[1:, :]) | (y[:-1, :] != (y[1:, :]))

        ret = distance_transform_edt(1 - ret) <= bandwidth
        return ret

    return np.mean([acc_score(y, p, boundary_mask(yo))
                    for yo, y, p in zip(y_true, map_to_ordinal(y_true),
                                        y_pred)])


def mae(y_true, y_pred):
    def mae_score(y, p):
        y_ = y.ravel().astype(np.float)
        p_ = p.ravel().astype(np.float)

        return np.mean(np.abs(y_ - p_))
    
    ret = [mae_score(y, p) for y, p in zip(map_to_ordinal(y_true), y_pred)]

    return np.mean(ret)


def mae_at_boundary(y_true, y_pred, bandwidth=5):
    def mae_score(y, p, mask):
        y_ = y.ravel().astype(np.float)
        p_ = p.ravel().astype(np.float)
        m_ = mask.ravel()

        return np.mean(np.abs(y_ - p_)[m_ != 0])

    def boundary_mask(y):
        ret = np.zeros(y.shape, dtype=np.uint8)
        ret[:, 1::] = (ret[:, 1:]) | (y[:, :-1] != (y[:, 1:]))
        ret[1::, :] = (ret[1:, :]) | (y[:-1, :] != (y[1:, :]))

        ret = distance_transform_edt(1 - ret) <= bandwidth
        return ret

    return np.mean([mae_score(y, p, boundary_mask(yo))
                    for yo, y, p in zip(y_true, map_to_ordinal(y_true),
                                        y_pred)])


np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

args = get_args()

f = open(args.dataset, 'r')
dataset_config = json.load(f)
f.close()

#tr_imgs, tr_masks = load_set(args, 'train')
#val_imgs, val_masks = load_set(args, 'validation')
test_imgs, test_masks = load_set(dataset_config, 0, 'test')

dataset_path = dataset_config['path']
losses = [('GDice', utils.geometric_dice_loss),
          ('Crossentropy', utils.average_crossentropy)]
augment = dataset_config['augment']


ordinal = (True, False, True, True)
conv_num = 4
conv_filter_size = 3
loss_name, loss = losses[0]

for foldid in range(dataset_config['folds']):
    ord_ = ordinal
    print 'Fold %d' % foldid

    for w in [1e-4, 1e-3, 1e-2, 1e-1]:
        print 'Smoothness weight %f' % w

        path = os.path.join('output', 'spatial-models',
                            dataset_config['name'],
                            'fold%d' % foldid,
                            'keras',
                            'spatial-unet%d-%d-%s-%d-%s-%d-%f.hdf5' %
                            (conv_num, conv_filter_size, loss_name,
                            ord_[0], ord_[1], ord_[2], w))

        pckl_path = os.path.join('output', 'spatial-models',
                                    dataset_config['name'],
                                    'fold%d' % foldid,
                                    'spatial-unet%d-%d-%s-%d-%s-%d-%f.pckl' %
                                    (conv_num, conv_filter_size, loss_name,
                                    ord_[0], ord_[1], ord_[2], w))

        if os.path.exists(path) or os.path.exists(pckl_path):
            print path, 'already exists'
            continue

        model = OrdinalUNet(conv_num=conv_num,
                            conv_filter_size=conv_filter_size,
                            loss=loss,
                            ordinal_output=ord_[0],
                            pointwise_ordinal=ord_[1],
                            parallel_hyperplanes=ord_[2],
                            include_distances=ord_[3],
                            smoothness_weight=w,
                            augment=augment,
                            max_epochs=500,
                            keras_filepath=path,
                            )

        model.fit_from_dir(os.path.join(dataset_path,
                                        'fold%d' % foldid))

        model.opt_network = None
        model.network = None

        f = open(pckl_path, 'wb')
        pickle.dump(model, f)
        f.close()
