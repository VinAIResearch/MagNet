from .fpn import ResnetFPN

NAME2MODEL = {
    "fpn": ResnetFPN
}

def get_model_with_name(model_name):
    if model_name in NAME2MODEL:
        return NAME2MODEL[model_name]
    else:
        raise "Cannot found the implementation of model " + model_name