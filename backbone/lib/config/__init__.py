# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

from .default import _C as config
from .default import update_config


__all__ = ["config", "update_config"]
