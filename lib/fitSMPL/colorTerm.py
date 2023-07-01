import copy
import torch
import scipy
import trimesh
import torch.nn as nn
import cv2
import trimesh
import math
import numpy as np
from icecream import ic

from lib.visualizer.renderer import modelRender
from lib.Utils.depth_utils import depth2PointCloud#(depth_map,fx,fy,cx,cy)

class ColorTerm(nn.Module):
    def __init__(self,depth2color=None,
                 depth2floor=None,
                 cam_intr=None,
                 img_W=1280,img_H=720,
                 dtype=np.float32,
                 device='cpu'):
        super(ColorTerm, self).__init__()

        self.depth2floor = depth2floor
        self.floor2depth = np.linalg.inv(depth2floor)
        self.depth2color = depth2color
        self.cam_intr = cam_intr #fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.dtype=dtype
        self.device=device
