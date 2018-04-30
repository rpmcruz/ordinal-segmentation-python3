import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os


def ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path)


in_path = os.sys.argv[1]
out_path = os.sys.argv[2]

disk = pd.read_excel(os.path.join(in_path, 'optic_disk_centers.xls'))


img_path = os.path.join(out_path, 'imgs')
mask_path = os.path.join(out_path, 'masks')

ensuredir(img_path)
ensuredir(mask_path)

for i, f in enumerate(sorted(list(os.walk(os.path.join(in_path,
                                                       'images')))[0][2])):
    next_disk = disk.iloc[i]

    img = cv2.imread(os.path.join(in_path, 'images', f))
    #image     Pap. Center y   vessel orig. x  vessel orig. y  disk diameter
    mask = cv2.imread(os.path.join(in_path, 'mask',
                                   f.split('.')[0] + '_mask.tif'), 0)
    mask = (mask != 0).astype(np.uint8)

    cv2.circle(mask,
               (next_disk['Pap. Center x'], next_disk['Pap. Center y']),
               next_disk['disk diameter'] / 2, 2, -1)

    mask *= 127

    cv2.imwrite(os.path.join(img_path, '%02d.jpg' % i), img)
    cv2.imwrite(os.path.join(mask_path, '%02d.bmp' % i), mask)
