from scipy.ndimage.morphology import distance_transform_edt
#from tensorflow.compat.v1.keras.backend.tensorflow_backend import set_session
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import argparse
import pickle
import json
import cv2
import os


from networks.unet import OrdinalUNet
from networks import utils
from functools import reduce

nlabels = None

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
                        default=5, help='Last fold')

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

            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256),
                              interpolation=cv2.INTER_NEAREST)

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
                    for y, p in zip(map_to_ordinal(y_true), y_pred)]) * 100.


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
                    for y, p in zip(map_to_ordinal(y_true), y_pred)]) * 100.


def accuracy(y_true, y_pred):
    def acc_score(y, p):
        y_ = y.ravel()
        p_ = p.ravel()

        return np.mean(y_ == p_)

    return np.mean([acc_score(y, p)
                    for y, p in zip(map_to_ordinal(y_true), y_pred)]) * 100.


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


def confmat(y_true, y_pred):
    global nlabels
    ret = np.zeros((nlabels, nlabels))

    for m, p in zip(map_to_ordinal(y_true), y_pred):
        m_ = m.ravel()
        p_ = p.ravel()
        cc = Counter(list(zip(m_, p_)))
        for (actual_label, predicted_label), freq in list(cc.items()):
            ret[actual_label][predicted_label] += freq * 100. / len(m_)

    ret /= len(y_true)

    return ret


def hausdorff(y_true, y_pred):
    global nlabels

    ret = []
    for m, p in zip(map_to_ordinal(y_true), y_pred):
        values = []
        for l in range(nlabels):
            diff = (m == l) != (p == l)
            dist = distance_transform_edt(diff != 0)

            next_dist = 2. * dist.max()
            next_dist /= np.sqrt(np.sum(np.square(m.shape[: 2])))

            values.append(next_dist)
        ret.append(np.mean(values))
    return 100. * np.mean(ret)


np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))

args = get_args()

f = open(args.dataset, 'r')
dataset_config = json.load(f)
f.close()

#tr_imgs, tr_masks = load_set(args, 'train')
val_imgs, val_masks = load_set(dataset_config, 0, 'validation')
test_imgs, test_masks = load_set(dataset_config, 0, 'test')

dataset_path = dataset_config['path']
losses = [('GDice', utils.geometric_dice_loss),
          ('Crossentropy', utils.average_crossentropy)]
augment = dataset_config['augment']

ordinal_list = [
                ## Baseline
                #(False, False, False, False),
                ## Frank & Hall output
                (True, False, False, False),
                # (True, False, True, False),

                ## Frank & Hall output + Min consistency
                #(True, 'min', False, False),
                #(True, 'min', True, False),
                ##(True, 'min', False, True),

                ## Frank & Hall output + Multiply consistency
                #(True, 'mul', False, False),
                #(True, 'mul', True, False),
                #(True, 'mul', False, True),
                ]

metrics = [('Hausdorff', hausdorff, lambda x, y: x < y),
           ('Accuracy', accuracy, lambda x, y: x > y),
           ('Dice', dice_coef, lambda x, y: x > y),
           ('Non-ordinal Error', nonordinal_error, lambda x, y: x < y),
           ('MAE', mae, lambda x, y: x < y)]

results = {ord_: {m: [] for m, _, _ in metrics} for ord_ in ordinal_list}

for ord_ in ordinal_list:
    folds = [0, 1, 2, 3, 4]
    #folds = [0]

    for foldid in folds:
        print(foldid)
        val_best_results = {}
        test_results = {}

        for conv_num in range(2, 5):
            conv_filter_size = 3
            loss_name, loss = losses[0]
            pckl_path = os.path.join('output', 'models',
                                     dataset_config['name'],
                                     'fold%d' % foldid,
                                     'unet%d-%d-%s-%d-%s-%d.pckl' %
                                     (conv_num, conv_filter_size, loss_name,
                                      ord_[0], ord_[1], ord_[2]))
            if not os.path.exists(pckl_path):
                print(pckl_path, 'doesn\'t exist')
                continue

            _original_stdout = os.sys.stdout
            os.sys.stdout = open(os.devnull, 'w')

            try:
                f = open(pckl_path, 'rb')
                model = pickle.load(f)
                f.close()

                model.load_weights()
            except:
                break


            val_preds = model.predict(val_imgs)
            preds = model.predict(test_imgs)

            os.sys.stdout = _original_stdout

            classes = set([y for x in test_masks
                           for y in list(set(x.ravel()))])
            nlabels = len(classes)

            for metric_name, metric, better in metrics:
                val_result = metric(val_masks, val_preds)
                test_result = metric(test_masks, preds)
                if metric_name not in val_best_results:
                    if metric_name == 'Dice':
                            print('fold', foldid, 'best', conv_num)
                    val_best_results[metric_name] = val_result
                    test_results[metric_name] = test_result
                else:
                    if better(val_result, val_best_results[metric_name]):
                        if metric_name == 'Dice':
                            print('fold', foldid, 'best', conv_num)
                        val_best_results[metric_name] = val_result
                        test_results[metric_name] = test_result
        print()
        for metric, _, _ in metrics:
            if metric in test_results:
                results[ord_][metric].append(test_results[metric])

    print(ord_)
    for metric, _, _ in metrics:
        print('%20s: %11.6f' % (metric, np.mean(results[ord_][metric])), \
            results[ord_][metric])

    res = ['%11.6f' % np.mean(results[ord_][metric])
           for metric, _, _ in metrics]
    print(' & '.join(res), '\\\\')
    print()

"""
            cm = confmat(test_masks, preds)
            print
            for line in cm:
                print ' '.join(['%11.4f' % y for y in line])
            print
"""
