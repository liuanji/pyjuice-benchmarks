# Benchmarks of PyJuice

This repository contains code to benchmark common Probabilistic Circuit structures created using [PyJuice](https://github.com/Juice-jl/pyjuice).

## Usage

Make sure PyJuice is properly installed. You can find more instructions in the repository. The simplest way is to install it with pip:

```
pip install pyjuice
```

If you want to use the latest dev version of PyJuice, please follow the instructions in [the PyJuice repo](https://github.com/Tractables/pyjuice).

Go to the `exps/simple_pcs/` directory:

```
cd exps/simple_pcs
```

Run the `main.py` file with the configurations specified in `configs/`:

```
python main.py --data-config [Dataset configuration file] --model-config [PC model configuration file] --optim-config [Optimizer configuration file]
```

## Dataset structure

ImageNet:

```
ImageNet
├── train
│   ├── filelist.txt
│   ├── n01697457
│   │   ├── n01697457_11482.JPEG
│   │   ├── n01697457_11492.JPEG
│   │   ├── ...
│   ├── n01698640
│   │   ├── ...
│   ├── ...
├── val
│   ├── filelist.txt
│   ├── n01698640
│   │   ├── ILSVRC2012_val_00000090.JPEG
│   │   ├── ILSVRC2012_val_00001338.JPEG
│   │   ├── ...
│   ├── n01704323
│   │   ├── ...
│   ├── ...
```

ImageNet32:

```
ImageNet
├── imagenet32
│   ├── train
│   │   ├── train_data_batch_1.npz
│   │   ├── train_data_batch_2.npz
│   │   ├── ...
│   ├── val
│   │   ├── val_data.npz
```

## Examples

For example, to reproduce the results of PD-mid and PD-large on ImageNet32 (Table 3 in the [PyJuice paper](https://arxiv.org/pdf/2406.00766)), simply run

```bash
cd exps/simple_pcs
python main.py --data-config ../../configs/data/imagenet32.yaml --model-config ../../configs/model/pd_128.yaml # PD-mid
python main.py --data-config ../../configs/data/imagenet32.yaml --model-config ../../configs/model/pd_256.yaml # PD-large
```

Similarly, to use other models such as HCLT-mid and HCLT-large, please run

```bash
cd exps/simple_pcs
python main.py --data-config ../../configs/data/imagenet32.yaml --model-config ../../configs/model/hclt_256.yaml # HCLT-mid
python main.py --data-config ../../configs/data/imagenet32.yaml --model-config ../../configs/model/hclt_512.yaml # HCLT-large
```
