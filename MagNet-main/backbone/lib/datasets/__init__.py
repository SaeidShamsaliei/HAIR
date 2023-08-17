# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Editted by Chuong Huynh (v.chuonghm@vinai.io)
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

from .cityscapes import Cityscapes as cityscapes
from .deepglobe import DeepGlobe as deepglobe
from .river_segment import RiverSegment as river_segment
from .river_segment2 import RiverSegment2 as river_segment2
from .deepglobe_river import DeepGlobeRiver as deepglobe_river


__all__ = ["cityscapes", "deepglobe", "river_segment",
           "river_segment2", "deepglobe_river"]
