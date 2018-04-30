import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import json
import cv2
import os


RESIZE_FACTOR = 0.1

def ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_img(path, f, interp='bilinear'):
    img = cv2.imread(os.path.join(path, f))
    img = imresize(img, RESIZE_FACTOR, interp=interp)
    return img


def read_mask(path, f):
    try:
        img = cv2.imread(os.path.join(path, f), 0)
        assert img is not None
    except:
        img = cv2.imread(os.path.join(path, f.replace('.jpg', '.png')), 0)
    img = imresize(img, RESIZE_FACTOR, interp='nearest')
    img = (img > 50).astype(np.uint8)
    return img


in_path = os.sys.argv[1]
out_path = os.sys.argv[2]

classes = ['Type_1', 'Type_2', 'Type_3']
labels = ['speculum', 'roi', 'union']
#classes = ['Type_3']

out_img_path = os.path.join(out_path, 'imgs')
out_mask_path = os.path.join(out_path, 'masks')

ensuredir(out_img_path)
ensuredir(out_mask_path)

for i, c in zip(np.arange(len(classes)), classes):
    print c

    f = open(os.path.join(in_path, 'annotations', 'os', 'train',
                          'os_%d.json' % (i + 1)))
    os_coords = json.load(f)
    f.close()

    img_path = os.path.join(in_path, 'train', c)
    
    filenames = list(os.walk(img_path))[0][2]
    filenames = sorted(filenames)
    #filenames = filenames[:20]
    
    for if_, f in enumerate(filenames):
        print if_, f

        img = read_img(img_path, f)

        masks = {}
        for label in labels:
            try:
                mask_path = os.path.join(in_path, 'annotations', label,
                                         'train', c)
                masks[label] = read_mask(mask_path, f)
            except:
                masks[label] = np.zeros(img.shape[: 2], dtype=np.bool)

        mask = np.zeros(img.shape[: 2], dtype=np.uint8)
        mask[masks['speculum'] != 0] = 1
        mask[masks['roi'] != 0] = 2
        mask[masks['union'] != 0] = 3

        if f in os_coords:
            coords = (np.asarray(os_coords[f]) * RESIZE_FACTOR).astype(int)
            coords = list(coords) + list(coords[::-1])
            coords = np.asarray(coords)

            cv2.drawContours(mask, [coords], -1, 4, 10)
        else:
            print 'Error: no external os'

        mask *= 60

        cv2.imwrite(os.path.join(out_img_path, c + '_' + f), img)
        cv2.imwrite(os.path.join(out_mask_path,
                                 c + '_' + f.replace('.jpg', '.bmp')), mask)
        print 'done'
        print

"""
border = 10

for modality in modalities:
    images = [cv2.imread(os.path.join(in_path, 'images', modality, f))
              for f in filenames]

    mask_path = os.path.join(in_path, 'masks')
    speculum = [cv2.imread(os.path.join(mask_path, 'speculum', modality, f),
                           0) > 10 for f in filenames]

    cervix = [cv2.imread(os.path.join(mask_path, 'cervix', modality, f),
                         0) > 10 for f in filenames]
    ext_os = [cv2.imread(os.path.join(mask_path, 'os', modality, f),
                         0) > 10 for f in filenames]

    mod_img_path = os.path.join(out_path, modality, 'imgs')
    mod_mask_path = os.path.join(out_path, modality, 'masks')

    full_img_path = os.path.join(out_path, 'full', 'imgs')
    full_mask_path = os.path.join(out_path, 'full', 'masks')

    ensuredir(mod_img_path)
    ensuredir(mod_mask_path)
    ensuredir(full_img_path)
    ensuredir(full_mask_path)
    
    for f, i, s, c, o in zip(filenames, images, speculum, cervix, ext_os):
        ordinal = np.ones(s.shape, dtype=np.uint8)
        ordinal[s] = 0
        ordinal[c] = 2
        ordinal[o] = 3
        ordinal *= 85

        i = i[border: -border, border: -border]
        ordinal = ordinal[border: -border, border: -border]

        cv2.imwrite(os.path.join(mod_img_path, f), i)
        cv2.imwrite(os.path.join(mod_mask_path, f.replace('.jpg', '.bmp')),
                    ordinal)

        cv2.imwrite(os.path.join(full_img_path, modality + '_' + f), i)
        cv2.imwrite(os.path.join(full_mask_path,
                                 modality + '_' + f.replace('.jpg', '.bmp')),
                    ordinal)
"""