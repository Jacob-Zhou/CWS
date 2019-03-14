# -*- coding: utf-8 -*-

import time

import torch

padding_str = '<-PAD->'
padding_idx = 0
unknown_str = '<-UNK->'
unknown_idx = 1
eps_ratio = 1e-10


def coarse_equal_to(self, a, b):
    eps = eps_ratio * abs(b)
    return b + eps >= a >= b - eps


def get_time_str():
    return time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time()))
