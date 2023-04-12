from sklearn.model_selection import KFold
import numpy as np
import argparse
import shutil
import os


def get_args():
    parser = argparse.ArgumentParser(
                        description="Create train/validation/test partitions.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', metavar="I", nargs='?',
                        default='', help='Path to the input img/masks')
    parser.add_argument('--output', metavar="O", nargs='?',
                        default='', help='Path to the output partitions')
    parser.add_argument('--folds', type=int, default=5)
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


args = get_args()

in_path = args.input
out_path = args.output

ensure_dir(out_path)

img_filenames = set(list(os.walk(os.path.join(in_path, 'imgs')))[0][2])
mask_filenames = set(list(os.walk(os.path.join(in_path, 'masks')))[0][2])

img_prefix = [x.split('.')[0] for x in img_filenames]
mask_prefix = [x.split('.')[0] for x in mask_filenames]

filenames = set(img_prefix) & set(mask_prefix)

img_filenames = np.asarray([x for x in sorted(img_filenames)
                            if x.split('.')[0] in filenames])
mask_filenames = np.asarray([x for x in sorted(mask_filenames)
                             if x.split('.')[0] in filenames])

kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

all_index = [t for _, t in kf.split(np.arange(len(img_filenames)))]
final_partitions = []
for i in range(args.folds):
    vi = (i + 1) % args.folds
    ti = i

    tr_index = [next_part for ci, next_part in enumerate(all_index)
                if ci != ti and ci != vi]
    val_index = all_index[vi]
    ts_index = all_index[ti]
    
    tr_index = np.hstack(tr_index)

    final_partitions.append({'train': tr_index,
                             'validation': val_index,
                             'test': ts_index})

for fold_id, index in enumerate(final_partitions):
    for subset, index in list(index.items()):
        print(fold_id, subset)
        new_img_path = os.path.join(args.output, 'fold%d' % fold_id,
                                    subset, 'imgs', 'seg')
        new_mask_path = os.path.join(args.output, 'fold%d' % fold_id,
                                    subset, 'masks', 'seg')
        
        ensure_dir(new_img_path)
        ensure_dir(new_mask_path)

        for ifs, mfs in zip(img_filenames[index], mask_filenames[index]):
            shutil.copy(os.path.join(in_path, 'imgs', ifs),
                        os.path.join(new_img_path, ifs))
            shutil.copy(os.path.join(in_path, 'masks', mfs),
                        os.path.join(new_mask_path, mfs))
    print()
os.sys.exit(0)