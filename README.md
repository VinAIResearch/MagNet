
<p align="center">	
<img width="150" alt="logo" src="https://i.imgur.com/0OaOlKO.png">
</p>

# Progressive Semantic Segmentation (MagNet)

[**MagNet**](https://github.com/VinAIResearch/MagNet), a multi-scale framework that resolves local ambiguity by looking at the image at multiple magnification levels, has multiple processing stages, where each stage corresponds to a magnification level, and the output of one stage is fed into the next stage for coarse-to-fine information propagation. Experiments on three high-resolution datasets of urban views, aerial scenes, and medical images show that MagNet consistently outperforms the state-of-the-art methods by a significant margin.
![](https://i.imgur.com/fCPhKyX.png)

Details of the MagNet model architecture and experimental results can be found in our [following paper](https://arxiv.org/abs/2104.03778):

Progressive Semantic Segmentation. \
C. Huynh, A. Tran, K. Luu, M. Hoai (2021) \
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```
@inproceedings{m_Huynh-etal-CVPR21,
  author = {Chuong Huynh and Anh Tran and Khoa Luu and Minh Hoai},
  title = {Progressive Semantic Segmentation},
  year = {2021}, \
  booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
**Please CITE** our paper when MagNet is used to help produce published results or incorporated into other software.

## Quickly run MagNet

Before reading the detailed instruction about training and testing MagNet on your machine, we provide a [Google Colab Notebook](https://colab.research.google.com/drive/1WTdfIQIEQrnoX40YIzs3HqeIKSZD_iPG?usp=sharing) for testing our pre-trained models with street-view images. Please follow the instructions in the notebook to experience the performance of our network.

## Requirements

The framework is tested on machines with the following environment:
- Python >= 3.6
- CUDA >= 10.0

To install dependencies, please run the following command:
```bash
python -m pip install -r requirements.txt
```

## Dataset
Please read this [document](data/README.md) to download and prepare the dataset.

## Pretrained models
Please read this [document](checkpoints/README.md) to download checkpoints.

## Demo

We provide several demo scripts that you can use to test with your images. There are some sample images in `data/` that you can use.

To test with Cityscapes images, please run the script below:
- With MagNet refinement:
```bash
sh scripts/cityscapes/demo_magnet.sh <image_path>
```
- With MagNet-Fast refinement
```bash
sh scripts/cityscapes/demo_magnet_fast.sh <image_path>
```

To test with Deepglobe images, please run the script below:
- With MagNet refinement:
```bash
sh scripts/deepglobe/demo_magnet.sh <image_path>
```
- With MagNet-Fast refinement
```bash
sh scripts/deepglobe/demo_magnet_fast.sh <image_path>
```

## Evaluation

To get the description of evaluation configs, please run the script below:
```bash
python test.py --help
```

Otherwise, there are sample scripts below to test with our pre-trained models.

### Cityscapes

Full MagNet refinement:
```bash 
sh scripts/cityscapes/test_magnet.sh
```
MagNet-Fast refinement:
```bash
sh scripts/cityscapes/test_magnet_fast.sh
```

### Deepglobe

Full MagNet refinement:
```bash 
sh scripts/deepglobe/test_magnet.sh
```
MagNet-Fast refinement:
```bash
sh scripts/deepglobe/test_magnet_fast.sh
```

## Training

### Training backbone networks

Please refer to this [HRNet repository](https://github.com/HRNet/HRNet-Semantic-Segmentation) to train the backbone networks.

### Training refinement modules

To get the description of training configs, please run the script below:
```bash
python train.py --help
```

#### Cityscapes
To train MagNet with Cityscapes dataset, please run this sample script:
```bash
sh scripts/cityscapes/train_magnet.sh
```

#### Deepglobe
To train MagNet with Deepglobe dataset, please run this sample script:
```bash 
sh scripts/deepglobe/train_magnet.sh
```

## Contact
If you have any question, please drop an email to [v.chuonghm@vinai.io](mailto:v.chuonghm@vinai.io) or create an issue on this repository.
