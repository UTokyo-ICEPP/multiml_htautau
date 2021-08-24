# multiml_htautau
*multiml_htautau* is a code for [Event Classification with Multi-step Machine Learning](https://www.epj-conferences.org/articles/epjconf/abs/2021/05/epjconf_chep2021_03036/epjconf_chep2021_03036.html)([arxiv](https://arxiv.org/abs/2106.02301)) in vCHEP 2021.

## Dependency
multiml (https://github.com/UTokyo-ICEPP/multiml)

## Installation
```bash
$ pip install -e .
```

## Run
Scripts for DARTS is in `examples/keras`, and scripts for SPOS-NAS is in `examples/pytorch`.
### Grid search with Keras:
```bash
$ python run_multi_connection_grid.py --tau4vec_weights 0.5 -n 50000 --data_path DATA_PATH
```
### DARTS with Keras:
```bash
$ python run_multi_connection_darts.py --individual_loss_weights 1.0 --tau4vec_weights 0.5 -n 50000 --data_path DATA_PATH
```
### SPOS-NAS with PyTorch:
```bash
$ python run_multi_connection_sposnas.py --weight 0.5 -e 50000 --data_path DATA_PATH
```

## Tests and coding style
Please test and apply yapf before you commit changes.
```bash
$ python setup.py test
```
```bash
$ yapf -i [changed file]
```

