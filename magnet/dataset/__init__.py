from .cityscapes import Cityscapes
from .deepglobe import Deepglobe


NAME2DATASET = {"deepglobe": Deepglobe, "cityscapes": Cityscapes}


def get_dataset_with_name(dataset_name):
    if dataset_name in NAME2DATASET:
        return NAME2DATASET[dataset_name]
    else:
        raise "Cannot found dataset " + dataset_name
