from scipy.ndimage.morphology import distance_transform_edt
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import cv2
import os


from networks.unet import OrdinalUNet
from networks import utils

nlabels = None

def map_to_ordinal(y_true):
    values = reduce(lambda ac, n: ac | set(n.ravel()), y_true, set([]))
    values = np.asarray(sorted(list(values)))
    mapping = np.zeros(256, dtype=np.uint8)
    mapping[values] = np.arange(len(values))
    
    return [mapping[y] for y in y_true]


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


def ordinal_transitions(y):
    global nlabels
    ret = []
    for m in map_to_ordinal(y):
        mvert = np.abs(m[1:, :] - m[: -1, :])
        mhor = np.abs(m[:, 1:] - m[:, : -1])
        mdiag1 = np.abs(m[1:, 1:] - m[: -1, : -1])
        mdiag2 = np.abs(m[1:, :-1] - m[: -1, 1:])

        ordinal = (mvert == 1).sum() + (mhor == 1).sum() + \
            (mdiag1 == 1).sum() + (mdiag2 == 1).sum()
        total = (mvert != 0).sum() + (mhor != 0).sum() + \
            (mdiag1 != 0).sum() + (mdiag2 != 0).sum()
        ret.append(float(ordinal + 1.) / float(total + 1.))

    return 100. * np.mean(ret)


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
val_imgs, val_masks = load_set(dataset_config, 0, 'validation')
test_imgs, test_masks = load_set(dataset_config, 0, 'test')

dataset_path = dataset_config['path']
losses = [('GDice', utils.geometric_dice_loss),
          ('Crossentropy', utils.average_crossentropy)]
augment = dataset_config['augment']

ordinal_list = [
                # Baseline
                (False, False, False, False),
                # Frank & Hall output
                (True, False, False, False),
                (True, False, True, False),

                # Frank & Hall output + Min consistency
                #(True, 'min', False, False),
                #(True, 'min', True, False),
                #(True, 'min', False, True),

                # Frank & Hall output + Multiply consistency
                (True, 'mul', False, False),
                (True, 'mul', True, False),
                #(True, 'mul', False, True),
                ]


results = {ord_: [] for ord_ in ordinal_list}

for ord_ in ordinal_list:
    folds = [0, 1, 2, 3, 4]
    #folds = [0]

    for foldid in folds:
        print('Fold %d' % foldid)
        val_best_results = -np.inf
        test_results = -np.inf

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
                print pckl_path, 'doesn\'t exist'
                continue

            #print pckl_path

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

            val_result = ordinal_transitions(val_preds)

            os.sys.stdout = _original_stdout
            #print(np.asarray(val_imgs).shape, np.asarray(val_preds).shape)

            if val_result > val_best_results:
                val_best_results = val_result

                preds = model.predict(test_imgs)
                test_results = ordinal_transitions(preds)

            print('Best: %f' % test_results)
        results[ord_].append(test_results)
        print
    
    print('======= FINAL =======')
    print(ord_, np.mean(results[ord_]))
