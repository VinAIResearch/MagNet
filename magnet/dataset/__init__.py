from .cityscapes import Cityscapes
from .deepglobe import Deepglobe


NAME2DATASET = {"deepglobe": Deepglobe, "cityscapes": Cityscapes}


def get_dataset_with_name(dataset_name):
    """Get the dataset class from name

    Args:
        dataset_name (str): defined name of the dataset

    Raises:
        ValueError: when not found the dataset

    Returns:
        nn.Dataset class: class of found dataset
    """
    if dataset_name in NAME2DATASET:
        return NAME2DATASET[dataset_name]
    else:
        raise ValueError("Cannot found dataset " + dataset_name)
