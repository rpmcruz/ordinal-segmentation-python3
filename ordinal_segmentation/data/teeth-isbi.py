import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


tr_img_path = 'data/datasets/teeth-ISBI/v2-TrainingData/'
tr_mask_path = 'data/datasets/teeth-ISBI/v2-TrainingData/'

out_img_path = 'data/datasets/teeth-ISBI/subset/imgs/'
out_mask_path = 'data/datasets/teeth-ISBI/subset/masks/'

if not os.path.exists(out_img_path):
    os.makedirs(out_img_path)

if not os.path.exists(out_mask_path):
    os.makedirs(out_mask_path)

for i in range(1, 41):
    img = cv2.imread(os.path.join(tr_img_path, '%d.bmp' % i), 0)

    mask = np.zeros(img.shape, dtype=np.uint8)
    masks = [cv2.imread(os.path.join(tr_img_path, str(i),
                                     '%d_%d.bmp' % (i, l)), 0) != 0
             for l in range(1, 8)]

    for j, m in enumerate(masks):
        m = cv2.resize(m.astype(np.uint8), mask.shape[::-1]).astype(np.bool)
        masks[j] = m

    for other_mask in masks:
        try:
            mask = mask | other_mask
        except:
            pass

    mask = cv2.dilate(mask, np.ones((3, 3)))
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = mask.astype(np.uint8)

    mask[masks[1]] = 1
    mask[masks[2]] = 2
    mask[masks[3]] = 3
    
    mask = cv2.dilate(mask, np.ones((3, 3)))
    mask = cv2.erode(mask, np.ones((3, 3)))

    mask += 1
    
    tmask = cv2.imread(os.path.join(tr_img_path, 'foreground',
                                    '%02d.jpg' % i), 0)
    tmask = cv2.erode(tmask, np.ones((3, 3)))
    tmask = cv2.dilate(tmask, np.ones((3, 3)))
    mask[tmask == 0] = 0

    mask *= 60

    cv2.imwrite(os.path.join(out_img_path, '%02d.jpg' % i), img)
    cv2.imwrite(os.path.join(out_mask_path, '%02d.bmp' % i), mask)
