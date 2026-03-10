# -*- coding: utf-8 -*-
"""NPZ 数据集制作（视频/CSV）。用法: from BaseDT import npz 后使用 npz.make_npz_dataset、npz.NPZGenerator。"""

from .dataset import make_npz_dataset, NPZGenerator

__all__ = ['make_npz_dataset', 'NPZGenerator']
