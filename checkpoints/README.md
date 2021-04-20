# PRETRAINED MODELS
To download the pre-trained models, please run the script below:
```bash
sh ./download_checkpoints.sh
```

Performance of pre-trained models on datasets:

| Dataset | Backbone | Baseline IoU (%) | MagNet IoU (%) | MagNet-Fast IoU (%) |
| -------- | -------- | -------- | -------- | -------- |
| Cityscapes | HRNetW18+OCR | 63.24 | 68.20 | 67.37 |
| Deepglobe | Resnet50-FPN | 67.22 | 72.10 | 68.22 |