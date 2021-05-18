# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Edited by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '../../..')
add_path(lib_path)

from magnet.model.hrnet_ocr import HRNetW18_OCR
from magnet.model.fpn import ResnetFPN

def get_seg_model(cfg, **kwargs):
    model = eval(cfg.MODEL.NAME)(cfg.DATASET.NUM_CLASSES)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model