from .deepglobe import Deepglobe
from .gleason import Gleason

NAME2DATASET = {
    "deepglobe": Deepglobe,
    "gleason": Gleason
}

def get_dataset_with_name(dataset_name):
    if dataset_name in NAME2DATASET:
        return NAME2DATASET[dataset_name]
    else:
        raise "Cannot found dataset " + dataset_name