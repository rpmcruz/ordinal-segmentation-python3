# Ordinal Segmentation using Deep Neural Networks

This is Kelwin's code to his paper: https://ieeexplore.ieee.org/document/8489527

**My changes (Ricardo):**
* used the command-line `2to3` to port from Python2 to Python3
* modified the imports to use `tensorflow.compat.v1` API (compatibility API)
* some minor changes to make it work: such as using `multiprocessing=True` since I was getting erros due to thread unsafe generator.
* added a symlink: `data/partitions` to `/data/bioeng/ordinal-segmentation/partitions` (adjust to your datapath)
* I also added a baseline that uses 'softmax' instead of the 'sigmoid' (default)

**How to use?**
* Just run `python3 eval_ordinal_networks.py --dataset /data/bioeng/ordinal-segmentation/teeth.json` for the `teeth` dataset. It will automatically train all models and evaluate them.
* This will run for the first fold only. To run for other folds, you need to use `--first` and `--last`.
