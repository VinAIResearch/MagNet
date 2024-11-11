
<p align="center">	
<img width="150" alt="logo" src="https://i.imgur.com/0OaOlKO.png">
</p>

# Progressive Semantic Segmentation (MagNet)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WTdfIQIEQrnoX40YIzs3HqeIKSZD_iPG?usp=sharing) [![arXiv](https://img.shields.io/badge/arXiv-2104.03778-green.svg)](https://arxiv.org/abs/2104.03778) [![video](https://img.shields.io/badge/video-youtube-red.svg)](https://youtu.be/40Kw8QLym7E)

[**MagNet**](https://github.com/VinAIResearch/MagNet), a multi-scale framework that resolves local ambiguity by looking at the image at multiple magnification levels, has multiple processing stages, where each stage corresponds to a magnification level, and the output of one stage is fed into the next stage for coarse-to-fine information propagation. Experiments on three high-resolution datasets of urban views, aerial scenes, and medical images show that MagNet consistently outperforms the state-of-the-art methods by a significant margin.
![](https://i.imgur.com/fCPhKyX.png)

Details of the MagNet model architecture and experimental results can be found in our [following paper](https://arxiv.org/abs/2104.03778):
```
@inproceedings{m_Huynh-etal-CVPR21,
  author = {Chuong Huynh and Anh Tran and Khoa Luu and Minh Hoai},
  title = {Progressive Semantic Segmentation},
  year = {2021},
  booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
**Please CITE** our paper when MagNet is used to help produce published results or incorporated into other software.

## Datasets

This current code provides configurations to train, evaluate on two datasets: **Cityscapes** and **DeepGlobe**. To prepare the datasets, in the `./data` directory, please do following steps:
### For Cityscapes
1. Register an account on this [page](https://www.cityscapes-dataset.com/) and log in.
2. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`.
3. Run the script below to extract zip files to correct locations:
```bash
sh ./prepare_cityscapes.sh
```

### For DeepGlobe
1. Register an account on this [page](https://competitions.codalab.org/competitions/18468) and log in.
2. Go to this [page]() and download **Starting Kit** of the `#1 Development` Phase.
3. Run the script below to extract zip files to correct locations:
```bash
sh ./prepare_deepglobe.sh
```

If you want to train/evaluate with your dataset, follow the steps in this [document](CUSTOM_DATA.md)

## Getting started
### Requirements
The framework is tested on machines with the following environment:
- Python >= 3.6
- CUDA >= 10.0

To install dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

### Pretrained models
Performance of pre-trained models on datasets:

| Dataset | Backbone | Baseline IoU (%) | MagNet IoU (%) | MagNet-Fast IoU (%) | Download |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Cityscapes | HRNetW18+OCR | 63.24 | 68.20 | 67.37 |[backbone](https://drive.google.com/file/d/1FQghomK8Ssc0zHKrvhEMIJQrbVXiWa8H/view?usp=sharing)<br>[refine_512x256](https://drive.google.com/file/d/1B6NJgi2ujpW7K460vugRhWesK73LJKsb/view?usp=sharing)<br>[refine_1024x512](https://drive.google.com/file/d/1AShNe-1I8wPP_5kHK4G_wO8pSg_d1LKQ/view?usp=sharing)<br>[refine_2048x1024](https://drive.google.com/file/d/1XhVlMN2uzJ3qsrdJ19JvymeYDpvwMD82/view?usp=sharing) |
| DeepGlobe | Resnet50-FPN | 67.22 | 72.10 | 68.22 | [backbone](https://drive.google.com/file/d/1EFj3qNR7Xlp9DqOlKZAsyzkPUSFEY6OP/view?usp=sharing)<br>[refine](https://drive.google.com/file/d/1hybFYyMTCpdojzYPksrw5DdDL9yUH3L-/view?usp=sharing)

Please manually download pre-trained models to `./checkpoints` or run the script below:
```bash
cd checkpoints
sh ./download_cityscapes.sh # for Cityscapes
# or
sh ./download_deepglobe.sh # for DeepGlobe
```

### Usage

You can run this [Google Colab Notebook](https://colab.research.google.com/drive/1WTdfIQIEQrnoX40YIzs3HqeIKSZD_iPG?usp=sharing) to test our pre-trained models with street-view images. Please follow the instructions in the notebook to experience the performance of our network.

If you want to test our framework on your local machine:

1. To test with a Cityscapes image, e.g `data/frankfurt_000001_003056_leftImg8bit.png`:
- With MagNet refinement:
```bash
python demo.py --dataset cityscapes \
               --image data/frankfurt_000001_003056_leftImg8bit.png \
               --scales 256-128,512-256,1024-512,2048-1024 \
               --crop_size 256 128 \
               --input_size 256 128 \
               --model hrnet18+ocr \
               --pretrained checkpoints/cityscapes_hrnet.pth \
               --pretrained_refinement checkpoints/cityscapes_refinement_512.pth checkpoints/cityscapes_refinement_1024.pth checkpoints/cityscapes_refinement_2048.pth \
               --num_classes 19 \
               --n_points 32768 \
               --n_patches -1 \
               --smooth_kernel 5 \
               --save_pred \
               --save_dir test_results/demo

# or in short, you can run
sh scripts/cityscapes/demo_magnet.sh data/frankfurt_000001_003056_leftImg8bit.png
```

- With MagNet-Fast refinement
```bash
python demo.py --dataset cityscapes \
               --image frankfurt_000001_003056_leftImg8bit.png \
               --scales 256-128,512-256,1024-512,2048-1024 \
               --crop_size 256 128 \
               --input_size 256 128 \
               --model hrnet18+ocr \
               --pretrained checkpoints/cityscapes_hrnet.pth \
               --pretrained_refinement checkpoints/cityscapes_refinement_512.pth checkpoints/cityscapes_refinement_1024.pth checkpoints/cityscapes_refinement_2048.pth \
               --num_classes 19 \
               --n_points 0.9 \
               --n_patches 4 \
               --smooth_kernel 5 \
               --save_pred \
               --save_dir test_results/demo

# or in short, you can run
sh scripts/cityscapes/demo_magnet_fast.sh data/frankfurt_000001_003056_leftImg8bit.png
```

All results will be stored at `test_results/demo/frankfurt_000001_003056_leftImg8bit`

2. To test with a Deepglobe image, e.g `data/639004_sat.jpg`:
- With MagNet refinement:
```bash
python demo.py --dataset deepglobe \
               --image data/639004_sat.jpg \
               --scales 612-612,1224-1224,2448-2448 \
               --crop_size 612 612 \
               --input_size 508 508 \
               --model fpn \
               --pretrained checkpoints/deepglobe_fpn.pth \
               --pretrained_refinement checkpoints/deepglobe_refinement.pth \
               --num_classes 7 \
               --n_points 0.75 \
               --n_patches -1 \
               --smooth_kernel 11 \
               --save_pred \
               --save_dir test_results/demo

# or in short, you can run
sh scripts/deepglobe/demo_magnet.sh data/639004_sat.jpg
```
- With MagNet-Fast refinement
```bash
python demo.py --dataset deepglobe \
               --image data/639004_sat.jpg \
               --scales 612-612,1224-1224,2448-2448 \
               --crop_size 612 612 \
               --input_size 508 508 \
               --model fpn \
               --pretrained checkpoints/deepglobe_fpn.pth \
               --pretrained_refinement checkpoints/deepglobe_refinement.pth \
               --num_classes 7 \
               --n_points 0.9 \
               --n_patches 3 \
               --smooth_kernel 11 \
               --save_pred \
               --save_dir test_results/demo

# or in short, you can run
sh scripts/deepglobe/demo_magnet_fast.sh data/639004_sat.jpg
```
All results will be stored at `test_results/demo/639004_sat`

## Training

### Training backbone networks

We customize the training script from [HRNet repository](https://github.com/HRNet/HRNet-Semantic-Segmentation) to train our backbones. Please first go to this directory `./backbone` and run the following scripts:

#### HRNetW18V2+OCR for Cityscapes
Download [pre-trained weights](https://drive.google.com/file/d/1kQ5wnEQgUP9gjwRuhhaBZS2IGFc2ES6_/view?usp=sharing) on ImageNet and put into folder `./pretrained_weights`.

Training the model:
```
# In ./backbone
python train.py --cfg experiments/cityscapes/hrnet_ocr_w18_train_256x128_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
```
The logs of training are stored at `./log/cityscapes/HRNetW18_OCR`.

The checkpoint of backbone after training are stored at `./output/cityscapes/hrnet_ocr_w18_train_256x128_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth`.
This checkpoint is used to train further refinement modules.
#### Resnet50-FPN for Deepglobe
Training the model:
```
# In ./backbone
python train.py --cfg experiments/deepglobe/resnet_fpn_train_612x612_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
```
The logs of training are stored at `./log/deepglobe/ResnetFPN`.

The checkpoint of backbone after training are stored at `./output/deepglobe/resnet_fpn_train_612x612_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth`. This checkpoint is used to train further refinement modules.

### Training refinement modules

Available arguments for training:
```bash
train.py [-h] --dataset DATASET [--root ROOT] [--datalist DATALIST]
                --scales SCALES --crop_size N [N ...] --input_size N [N ...]
                [--num_workers NUM_WORKERS] --model MODEL --num_classes
                NUM_CLASSES --pretrained PRETRAINED
                [--pretrained_refinement PRETRAINED_REFINEMENT [PRETRAINED_REFINEMENT ...]]
                --batch_size BATCH_SIZE [--log_dir LOG_DIR] --task_name
                TASK_NAME [--lr LR] [--momentum MOMENTUM] [--decay DECAY]
                [--gamma GAMMA] [--milestones N [N ...]] [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name: cityscapes, deepglobe (default: None)
  --root ROOT           path to images for training and testing (default: )
  --datalist DATALIST   path to .txt containing image and label path (default:
                        )
  --scales SCALES       scales: w1-h1,w2-h2,... , e.g.
                        512-512,1024-1024,2048-2048 (default: None)
  --crop_size N [N ...]
                        crop size, e.g. 256 128 (default: None)
  --input_size N [N ...]
                        input size, e.g. 256 128 (default: None)
  --num_workers NUM_WORKERS
                        number of workers for dataloader (default: 1)
  --model MODEL         model name. One of: fpn, psp, hrnet18+ocr, hrnet48+ocr
                        (default: None)
  --num_classes NUM_CLASSES
                        number of classes (default: None)
  --pretrained PRETRAINED
                        pretrained weight (default: None)
  --pretrained_refinement PRETRAINED_REFINEMENT [PRETRAINED_REFINEMENT ...]
                        pretrained weight (s) refinement module (default:
                        [''])
  --batch_size BATCH_SIZE
                        batch size for training (default: None)
  --log_dir LOG_DIR     directory to store log file (default: runs)
  --task_name TASK_NAME
                        task name, experiment name. The final path of your
                        logs is <log_dir>/<task_name>/<timestamp> (default:
                        None)
  --lr LR               learning rate (default: 0.001)
  --momentum MOMENTUM   momentum for optimizer (default: 0.9)
  --decay DECAY         weight decay for optimizer (default: 0.0005)
  --gamma GAMMA         gamma for lr scheduler (default: 0.1)
  --milestones N [N ...]
                        milestones to reduce learning rate (default: [10, 20,
                        30, 40, 45])
  --epochs EPOCHS       number of epochs for training (default: 50)
```

#### Cityscapes
To train MagNet with Cityscapes dataset, please run this sample script:
```bash
python train.py --dataset cityscapes \
                --root data/cityscapes \
                --datalist data/list/cityscapes/train.txt \
                --scales 256-128,512-256,1024-512,2048-1024 \
                --crop_size 256 128 \
                --input_size 256 128 \
                --num_workers 8 \
                --model hrnet18+ocr \
                --pretrained checkpoints/cityscapes_hrnet.pth \
                --num_classes 19 \
                --batch_size 8 \
                --task_name cityscapes_refinement \
                --lr 0.001

# or in short, run the script below
sh scripts/cityscapes/train_magnet.sh
```

#### Deepglobe
To train MagNet with Deepglobe dataset, please run this sample script:
```bash
python train.py --dataset deepglobe \
                --root data/deepglobe \
                --datalist data/list/deepglobe/train.txt \
                --scales 612-612,1224-1224,2448-2448 \
                --crop_size 612 612 \
                --input_size 508 508 \
                --num_workers 8 \
                --model fpn \
                --pretrained checkpoints/deepglobe_fpn.pth \
                --num_classes 7 \
                --batch_size 8 \
                --task_name deepglobe_refinement \
                --lr 0.001

# or in short, run the script below
sh scripts/deepglobe/train_magnet.sh
```

## Evaluation

Available arguments for testing:
```bash
test.py [-h] --dataset DATASET [--root ROOT] [--datalist DATALIST]
               --scales SCALES --crop_size N [N ...] --input_size N [N ...]
               [--num_workers NUM_WORKERS] --model MODEL --num_classes
               NUM_CLASSES --pretrained PRETRAINED
               [--pretrained_refinement PRETRAINED_REFINEMENT [PRETRAINED_REFINEMENT ...]]
               [--image IMAGE] --sub_batch_size SUB_BATCH_SIZE
               [--n_patches N_PATCHES] --n_points N_POINTS
               [--smooth_kernel SMOOTH_KERNEL] [--save_pred]
               [--save_dir SAVE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name: cityscapes, deepglobe (default: None)
  --root ROOT           path to images for training and testing (default: )
  --datalist DATALIST   path to .txt containing image and label path (default:
                        )
  --scales SCALES       scales: w1-h1,w2-h2,... , e.g.
                        512-512,1024-1024,2048-2048 (default: None)
  --crop_size N [N ...]
                        crop size, e.g. 256 128 (default: None)
  --input_size N [N ...]
                        input size, e.g. 256 128 (default: None)
  --num_workers NUM_WORKERS
                        number of workers for dataloader (default: 1)
  --model MODEL         model name. One of: fpn, psp, hrnet18+ocr, hrnet48+ocr
                        (default: None)
  --num_classes NUM_CLASSES
                        number of classes (default: None)
  --pretrained PRETRAINED
                        pretrained weight (default: None)
  --pretrained_refinement PRETRAINED_REFINEMENT [PRETRAINED_REFINEMENT ...]
                        pretrained weight (s) refinement module (default:
                        [''])
  --image IMAGE         image path to test (demo only) (default: None)
  --sub_batch_size SUB_BATCH_SIZE
                        batch size for patch processing (default: None)
  --n_patches N_PATCHES
                        number of patches to be refined at each stage. if
                        n_patches=-1, all patches will be refined (default:
                        -1)
  --n_points N_POINTS   number of points to be refined at each stage. If
                        n_points < 1.0, it will be the proportion of total
                        points (default: None)
  --smooth_kernel SMOOTH_KERNEL
                        kernel size of blur operation applied to error scores
                        (default: 16)
  --save_pred           save predictions or not, each image will contains:
                        image, ground-truth, coarse pred, fine pred (default:
                        False)
  --save_dir SAVE_DIR   saved directory (default: test_results)
```

Otherwise, there are sample scripts below to test with our pre-trained models.

### Cityscapes

Full MagNet refinement:
```bash
python test.py --dataset cityscapes \
               --root data/cityscapes \
               --datalist data/list/cityscapes/val.txt \
               --scales 256-128,512-256,1024-512,2048-1024 \
               --crop_size 256 128 \
               --input_size 256 128 \
               --num_workers 8 \
               --model hrnet18+ocr \
               --pretrained checkpoints/cityscapes_hrnet.pth \
               --pretrained_refinement checkpoints/cityscapes_refinement_512.pth checkpoints/cityscapes_refinement_1024.pth checkpoints/cityscapes_refinement_2048.pth \
               --num_classes 19 \
               --sub_batch_size 1 \
               --n_points 32768 \
               --n_patches -1 \
               --smooth_kernel 5 \
               --save_pred \
               --save_dir test_results/cityscapes

# or in short, run the script below
sh scripts/cityscapes/test_magnet.sh
```
MagNet-Fast refinement:
```bash
python test.py --dataset cityscapes \
               --root data/cityscapes \
               --datalist data/list/cityscapes/val.txt \
               --scales 256-128,512-256,1024-512,2048-1024 \
               --crop_size 256 128 \
               --input_size 256 128 \
               --num_workers 8 \
               --model hrnet18+ocr \
               --pretrained checkpoints/cityscapes_hrnet.pth \
               --pretrained_refinement checkpoints/cityscapes_refinement_512.pth checkpoints/cityscapes_refinement_1024.pth checkpoints/cityscapes_refinement_2048.pth \
               --num_classes 19 \
               --sub_batch_size 1 \
               --n_points 0.9 \
               --n_patches 4 \
               --smooth_kernel 5 \
               --save_pred \
               --save_dir test_results/cityscapes_fast

# or in short, run the script below
sh scripts/cityscapes/test_magnet_fast.sh
```

### Deepglobe

Full MagNet refinement:
```bash 
python test.py --dataset deepglobe \
               --root data/deepglobe \
               --datalist data/list/deepglobe/test.txt \
               --scales 612-612,1224-1224,2448-2448 \
               --crop_size 612 612 \
               --input_size 508 508 \
               --num_workers 8 \
               --model fpn \
               --pretrained checkpoints/deepglobe_fpn.pth \
               --pretrained_refinement checkpoints/deepglobe_refinement.pth \
               --num_classes 7 \
               --sub_batch_size 1 \
               --n_points 0.75 \
               --n_patches -1 \
               --smooth_kernel 11 \
               --save_pred \
               --save_dir test_results/deepglobe

# or in short, run the script below
sh scripts/deepglobe/test_magnet.sh
```
MagNet-Fast refinement:
```bash
python test.py --dataset deepglobe \
               --root data/deepglobe \
               --datalist data/list/deepglobe/test.txt \
               --scales 612-612,1224-1224,2448-2448 \
               --crop_size 612 612 \
               --input_size 508 508 \
               --num_workers 8 \
               --model fpn \
               --pretrained checkpoints/deepglobe_fpn.pth \
               --pretrained_refinement checkpoints/deepglobe_refinement.pth \
               --num_classes 7 \
               --sub_batch_size 1 \
               --n_points 0.9 \
               --n_patches 3 \
               --smooth_kernel 11 \
               --save_pred \
               --save_dir test_results/deepglobe_fast

# or in short, run the script below
sh scripts/deepglobe/test_magnet_fast.sh
```

## Acknowledgments
Thanks to [High-resolution networks and Segmentation Transformer for Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) for the backbone training script.

## Contact
If you have any question, please drop an email to [minhchuong.itus@gmail.com](mailto:minhchuong.itus@gmail.com) or create an issue on this repository.
