from .fpn import ResnetFPN
from .pspnet import PSPNet
from .hrnet_ocr import HRNetW18_OCR, HRNetW48_OCR

NAME2MODEL = {"fpn": ResnetFPN, "psp": PSPNet, "hrnet18+ocr": HRNetW18_OCR, "hrnet48+ocr": HRNetW48_OCR}


def get_model_with_name(model_name):
    if model_name in NAME2MODEL:
        return NAME2MODEL[model_name]
    else:
        raise "Cannot found the implementation of model " + model_name
