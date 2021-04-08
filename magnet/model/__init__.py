from .fpn import ResnetFPN
from .hrnet_ocr import HRNetW18_OCR, HRNetW48_OCR
from .pspnet import PSPNet


NAME2MODEL = {"fpn": ResnetFPN, "psp": PSPNet, "hrnet18+ocr": HRNetW18_OCR, "hrnet48+ocr": HRNetW48_OCR}


def get_model_with_name(model_name):
    """Get model class with defined name

    Args:
        model_name (str): name of model

    Raises:
        ValueError: if not found the model

    Returns:
        torch.nn.Module class: model class
    """
    if model_name in NAME2MODEL:
        return NAME2MODEL[model_name]
    else:
        raise ValueError("Cannot found the implementation of model " + model_name)
