# Customized dataset
To make the code run with your new dataset, prepare these components:
1. **Data**: images and corresponding labels stored in `<root_data>`
2. **File list**: `train.txt`, `val.txt`, `test.txt` are files in which each line contains a pair of image path and label path: `<image_path>\t<label_path>`. The absolute path of image and label are `<root_data>/<image_path>` and `<root_data>/<label_path>` respectively.
3. **Data loader**: define class encodings in the label.

The first two components can be prepared on your own. To create a **data loader**, please follow these steps:

1. Create a new sub-class of `BaseDataset` in [base.py](magnet/dataset/base.py). For example, we will create `MyDataset` in `magnet/dataset/mydataset.py`:
```python
from .base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

        self.label2color = {
            <class_id>: <rgb_color>
            ...
        }
        self.ignore_label = <ignore_class_id>
```
2. Define custom information:

Inside the `__init__` function, define the mappings between `<class_id>` and `<rgb_color>` to help transfer color label to segmentation map. 

If your dataset contains ignore class that is not considered in the loss function, please specify it in `<ignore_class_id>`.

If you want to read the label as a grayscale image, please change the reading mode in `__init__` function:
```
self.label_reading_mode = cv2.IMREAD_GRAYSCALE
```
and also overwrite the function `image2class` to process the label correctly:
```python
def image2class(self, label):
    ```
    label: h x w x 1 (grayscale) or h x w x 3 (rgb)
    return: h x w (classmap)
    ```
    # Process here
    return label
```

3. Define dataset name in `NAME2DATASET` in [`magnet/dataset/__init__.py`](magnet/dataset/__init__.py). That name will be used for training/ evaluation

