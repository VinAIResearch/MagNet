# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Editted by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

from .cityscapes import Cityscapes as cityscapes
from .deepglobe import DeepGlobe as deepglobe


__all__ = ["cityscapes", "deepglobe"]
