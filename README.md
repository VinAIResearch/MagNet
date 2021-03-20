
<p align="center">	
<img width="150" alt="logo" src="https://i.imgur.com/0OaOlKO.png">
</p>

# Progressive Semantic Segmentation (MagNet)

[**MagNet**](https://github.com/VinAIResearch/MagNet), a multi-scale framework that resolves local ambiguity by looking at the image at multiple magnification levels, has multiple processing stages, where each stage corresponds to a magnification level, and the output of one stage is fed into the next stage for coarse-to-fine information propagation. Experiments on three high-resolution datasets of urban views, aerial scenes, and medical images show that MagNet consistently outperforms the state-of-the-art methods by a significant margin.
![](https://i.imgur.com/fCPhKyX.png)

Details of the MagNet model architecture and experimental results can be found in our [following paper]():
```
@inproceedings{huynh2021magnet,
  title={Progressive Semantic Segmentation},
  author={Chuong Huynh and Anh Tran and Khoa Luu and Minh Hoai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021},
}
```
**Please CITE** our paper when MagNet is used to help produce published results or incorporated into other software.

## Quickly run MagNet

Before reading detail instructions about training and testing MagNet on your own machine, we provide a [Google Colab Notebook](https://colab.research.google.com/drive/1WTdfIQIEQrnoX40YIzs3HqeIKSZD_iPG?usp=sharing) for testing our pretrained models with street-view images. Please following the instructions in the notebook to experience the performance of our network.

## Requirements

The framework is tested on machines with following specifications:
- Python >= 3.5
- CUDA >= 10.0

To install dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

## Dataset

### Cityscapes
Please download two files `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` in this [page](https://www.cityscapes-dataset.com/downloads/) to the directory `data` and run the script below to prepare the data:
```bash
# In root directory
cd data
sh ./prepare_cityscapes.sh
```

### DeepGlobe
Please register [here](https://competitions.codalab.org/competitions/18468) and download **Starting Kit** of the `#1 Development` Phase in this [page](https://competitions.codalab.org/competitions/18468#participate-get_starting_kit) to the directory `data` and run the script below to prepare the data:
```bash
# In root directory
cd data
sh ./prepare_deepglobe.sh
```

## Pretrained models

## Training

## Testing

## Contact
If you have any question, please drop an email to [v.chuonghm@vinai.io](mailto:v.chuonghm@vinai.io) or create an issue on this repository.


