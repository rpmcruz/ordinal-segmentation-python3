from scipy.ndimage.morphology import distance_transform_edt
#from tensorflow.compat.v1.keras.backend.tensorflow_backend import set_session
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
#import tensorflow as tf
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
#set_session(tf.Session(config=config))

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

ordinal_list = [
                # Baseline
                (False, False, False, False),
                # Ricardo: Baseline with softmax
                (False, False, False, False, 'softmax'),
                # Frank & Hall output
                (True, False, False, False),
                (True, False, True, False),

                # Frank & Hall output + Min consistency
                (True, 'min', False, False),
                (True, 'min', True, False),
                #(True, 'min', False, True),

                # Frank & Hall output + Multiply consistency
                (True, 'mul', False, False),
                (True, 'mul', True, False),
                #(True, 'mul', False, True),
                
                ]

for foldid in range(dataset_config['folds']):
    for conv_num in range(2, 5):
        conv_filter_size = 3
        loss_name, loss = losses[0]

        for ord_ in ordinal_list:
            # Ricardo: allow using 'softmax' instead of 'sigmoid' (default)
            # Also added this suffix to the path to save both baselines.
            final_act = ord_[4] if len(ord_) > 4 else 'sigmoid'
            suffix = '-' + ord_[4] if len(ord_) > 4 else ''

            path = os.path.join('output', 'models',
                                dataset_config['name'],
                                'fold%d' % foldid,
                                'keras',
                                'unet%d-%d-%s-%d-%s-%d%s.hdf5' %
                                (conv_num, conv_filter_size, loss_name,
                                ord_[0], ord_[1], ord_[2], suffix))

            pckl_path = os.path.join('output', 'models',
                                     dataset_config['name'],
                                     'fold%d' % foldid,
                                     'unet%d-%d-%s-%d-%s-%d%s.pckl' %
                                     (conv_num, conv_filter_size, loss_name,
                                      ord_[0], ord_[1], ord_[2], suffix))

            if os.path.exists(path) or os.path.exists(pckl_path):
                print(path, 'already exists')
                continue

            model = OrdinalUNet(conv_num=conv_num,
                                conv_filter_size=conv_filter_size,
                                loss=loss,
                                ordinal_output=ord_[0],
                                pointwise_ordinal=ord_[1],
                                parallel_hyperplanes=ord_[2],
                                include_distances=ord_[3],
                                augment=augment,
                                max_epochs=5,
                                keras_filepath=path,
                                final_act=final_act,  # Ricardo
                                )

            model.fit_from_dir(os.path.join(dataset_path,
                                            'fold%d' % foldid))
            

            #model.opt_network = None
            #model.network = None

            # Ricardo: disabled saving - not currently working
            #f = open(pckl_path, 'wb')
            #pickle.dump(model, f)
            #f.close()

            # Ricardo: uncommented this code to evaluate results
            #"""
            preds = model.predict(test_imgs)
            probs = model.transform(test_imgs)

            print(path)

            classes = set([y for x in test_masks for y in list(set(x.ravel()))])
            nlabels = len(classes)

            for label_id, label in enumerate(sorted(list(classes))):
                current_mask = [t >= label for t in test_masks]
                current_preds = [p >= label_id for p in preds]

                print(np.mean([np.mean(y == p) for y, p in zip(current_mask, current_preds)]) * 100.)

            print(accuracy(test_masks, preds) * 100.)

            print('Accuracy:', accuracy(test_masks, preds) * 100.)
            print('Non-ordinal Error:', nonordinal_error(test_masks, preds) * 100.)
            print('ord-Accuracy:', ordinal_accuracy(test_masks, preds) * 100.)
            print('Dice:', dice_coef(test_masks, preds) * 100.)
            print('MAE:', mae(test_masks, preds))
            print()
            #"""
            print()

os.sys.exit(0)

"""
print 'Accuracy@10:', accuracy_at_boundary(test_masks, preds,
                                           bandwidth=10) * 100.
print 'Dice@10:', dice_at_boundary(test_masks, preds, bandwidth=10) * 100.
print 'MAE@10:', mae_at_boundary(test_masks, preds, bandwidth=10)
print
print 'Accuracy@5:', accuracy_at_boundary(test_masks, preds,
                                          bandwidth=5) * 100.
print 'Dice@5:', dice_at_boundary(test_masks, preds, bandwidth=5) * 100.
print 'MAE@5:', mae_at_boundary(test_masks, preds, bandwidth=5)
print
"""

for id_, (i, m, p) in enumerate(zip(test_imgs, test_masks, probs)[: 20]):
    if m is None:
        continue

    i2 = cv2.resize(i, p[0].shape[::-1])[:, :, ::-1]
    si2 = np.hstack([i2] * (2 + nlabels))

    m2 = cv2.resize(m, p[0].shape[::-1])

    d = (si2.astype(np.float) * 0.5 +
         np.hstack([i2, cv2.merge([m2] * 3)] +
                   [cv2.merge([p[l]] * 3)
                    for l in range(nlabels)]).astype(np.float) * 0.5)
    cv2.imwrite('%d.png' % id_, d.astype(np.uint8))
"""
"""

"""
    for res in [128, 256]:
        for nconv in [2, 3, 4, 5]:
            for convsize in [3, 5]:
                for loss_name, loss in [('dice', dice_coef_loss),
                                        ('cross-entropy',
                                         'binary_crossentropy')]:
                    net_name = 'unet-conv%dx%d-res-%d-%s' % (nconv, convsize,
                                                             res, loss_name)

                    name = os.path.join('output', 'models', args.task,
                                        'unet' + net_name + '.hdf5')
                    model = UNet(conv_num=nconv,
                                conv_activation='relu',
                                conv_filter_size=convsize,
                                conv_consecutive=2https://scholar.google.pt/,
                                generator_batch_size=16,
                                max_epochs=500,
                                img_size=(res, res),
                                l2=0.0001,
                                loss=loss,
                                keras_filepath=name,
                                )

                    else:
                        try:
                            model.load_weights()
                        except:
                            model.fit_from_dir(os.path.join('data',
                                                            'partitions',
                                                            args.task))
                            model.load_weights()

                    tr_preds = model.transform(tr_imgs)
                    val_preds = model.transform(val_imgs)
                    test_preds = model.transform(test_imgs)

                    f.write('%100s %f %f %f :: %f %f %f\n' %
                            (net_name,
                            dice_coef(tr_masks, tr_preds),
                            dice_coef(val_masks, val_preds),
                            dice_coef(test_masks, test_preds),
                            my_logloss(tr_masks, tr_preds),
                            my_logloss(val_masks, val_preds),
                            my_logloss(test_masks, test_preds)
                            ))

                    seg_path = os.path.join('output', 'segmentation',
                                            args.task, net_name)
                    if not os.path.exists(seg_path):
                        os.makedirs(seg_path)
                    for ix, (i, p) in enumerate(zip(test_imgs, test_preds)):
                        cv2.imwrite(os.path.join(seg_path, '%05d.jpg' % ix),
                                    np.hstack((i[:, :, ::-1],
                                            cv2.merge([p.astype(np.uint8)] *
                                                        3))))
"""
