# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Edited by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, "../../..")
add_path(lib_path)


def get_seg_model(cfg, **kwargs):

    from magnet.model.fpn import ResnetFPN
    from magnet.model.hrnet_ocr import HRNetW18_OCR

    if cfg.MODEL.NAME == "ResnetFPN":
        model_class = ResnetFPN
    elif cfg.MODEL.NAME == "HRNetW18_OCR":
        model_class = HRNetW18_OCR
    model = model_class(cfg.DATASET.NUM_CLASSES)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
