"""
-*- coding: utf-8 -*-
@Organization : SupaVision
@Author       : 18317
@Date Created : 12/01/2024
@Description  :
"""

import torch

from src.pose_estimation import DEVICE

print(torch.version.cuda)
print(DEVICE)
