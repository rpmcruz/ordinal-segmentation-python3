import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path)


modalities = ['green', 'hinselmann', 'schiller']

in_path = os.sys.argv[1]
out_path = os.sys.argv[2]

border = 10

for modality in modalities:
    filenames = list(os.walk(os.path.join(in_path, 'images', modality)))[0][2]
    filenames = sorted(filenames)

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
