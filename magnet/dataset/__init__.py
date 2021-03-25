from .deepglobe import Deepglobe

def get_dataset_with_name(dataset_name):
    if dataset_name == "deepglobe":
        return Deepglobe
    else:
        raise "Cannot found dataset " + dataset_name