import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def find_image(path, filename):
    ret = []
    for next_path, sub_path, files in os.walk(path):
        if filename in files:
            ret.append(os.path.join(next_path, filename))
    if len(ret) > 0:
        return ret[-1]
    return None


def swap_channels(img):
    ch = cv2.split(img)
    return cv2.merge((ch[2], ch[1], ch[0]))


def read_internal_contours(path):
    f = open(os.path.join(path, "Ctr_in.txt"))
    ctrs = f.readlines()
    f.close()


    ctrs = [l.strip() for l in ctrs]
    ctrs = [l[1:].split(';') for l in ctrs if l]
    ctrs = [[x for x in l[: -1] if x] for l in ctrs]
    ctrs = [(l[0], map(float, l[1:])) for l in ctrs]

    ctrs = [(f, zip(c[: len(c) / 2], c[len(c) / 2:])) for (f, c) in ctrs]

    return dict(ctrs)


def read_external_contours(path):
    f = open(os.path.join(path, "Ctr_out.txt"))
    ctrs = f.readlines()
    f.close()

    ctrs = [l.strip() for l in ctrs]
    ctrs = [l[1:].split('&') for l in ctrs if l]
    ctrs = [[y for x in l[: -1] for y in x.split(';')] for l in ctrs]
    ctrs = [(l[0], map(float, l[1:])) for l in ctrs]

    ctrs = [(f, zip(c[: len(c) / 2], c[len(c) / 2:])) for (f, c) in ctrs]

    return dict(ctrs)


def get_mask(img, contours):
    num_classes = len(contours) + 1

    dst = [np.zeros(img.shape[: 2], dtype=np.uint8) + 255]
    for c in contours:
        prev_mask = np.copy(dst[-1])
        next_mask = np.zeros(img.shape[: 2], dtype=np.uint8)
        cv2.fillPoly(next_mask, [np.array(c, np.int32)], 255, 0)
        next_mask = cv2.min(prev_mask, next_mask)
        dst.append(next_mask)

    drawing = np.zeros(img.shape[: 2], dtype=np.uint8)
    for i, d in enumerate(dst):
        drawing[d == 255] = num_classes - 1 - i

    return drawing

path = os.sys.argv[1]
in_c = read_internal_contours(path)
out_c = read_external_contours(path)
output = os.sys.argv[2]

common_filenames = [f for f in in_c if f in out_c]

plt.figure()

margin = 0.25

for i, f in enumerate(common_filenames):
    filename = find_image(path, f)
    if filename is not None:
        img = cv2.imread(filename)

        eye_mask = cv2.imread(os.path.join(output, 'masks-eye',
                                           '%03d.jpg' % i), 0)
        if eye_mask is None:
            print i, 'error mask'
            continue

        eye_mask = (eye_mask > 50).astype(np.uint8)

        print i

        """
        plt.scatter([x for x, y in in_c[f]],
                    [y for x, y in in_c[f]], color='b')
        plt.scatter([x for x, y in out_c[f]],
                    [y for x, y in out_c[f]], color='r')
        plt.show()
        plt.clf()
        """

        dst = eye_mask.copy()
        cv2.fillPoly(dst, [np.array(out_c[f], np.int32)], 2, 0)
        cv2.fillPoly(dst, [np.array(in_c[f], np.int32)], 3, 0)
        dst[eye_mask == 0] = 0

        dst *= 80

        fg_mask = dst != 0
        rows_with_info = np.arange(fg_mask.shape[0])[np.any(fg_mask, axis=1)]
        cols_with_info = np.arange(fg_mask.shape[1])[np.any(fg_mask, axis=0)]
        
        min_row = rows_with_info[0]
        max_row = rows_with_info[-1]
        min_col = cols_with_info[0]
        max_col = cols_with_info[-1]
        diff = max(int(margin * (max_row - min_row)),
                   int(margin * (max_col - min_col)))
        
        img = img[max(0, min_row - diff): min(img.shape[0], max_row + diff),
                  max(0, min_col - diff): min(img.shape[1], max_col + diff)]
        dst = dst[max(0, min_row - diff): min(dst.shape[0], max_row + diff),
                  max(0, min_col - diff): min(dst.shape[1], max_col + diff)]

        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2,
                         interpolation=cv2.INTER_NEAREST)
        dst = cv2.resize(dst, (0, 0), fx=0.2, fy=0.2,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output, 'imgs', '%03d.jpg' % i), img)
        cv2.imwrite(os.path.join(output, 'masks', '%03d.bmp' % i), dst)

    else:
        print i, 'error filename'