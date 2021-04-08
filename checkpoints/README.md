# PRETRAINED MODELS
To download the pretrained models, please run the script below:
```bash
sh ./download_checkpoints.sh
```

Performance of pretrained models on tested dataset.

| Dataset | Backbone | Baseline IoU (%) | MagNet IoU (%) | MagNet-Fast IoU (%) |
| -------- | -------- | -------- | -------- | -------- |
| Cityscapes | HRNetW18+OCR | 63.24 | 67.43 | 67.16 |
| Deepglobe | Resnet50-FPN | 67.22 | 72.12 | 71.41 |