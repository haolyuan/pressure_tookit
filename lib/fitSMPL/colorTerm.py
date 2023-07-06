import copy
import torch
import scipy
import trimesh
import torch.nn as nn
import cv2
import trimesh
import math
import numpy as np
from smplx.lbs import transform_mat
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
        floor2color = self.depth2color @ self.floor2depth
        self.floor2color = torch.tensor(floor2color,dtype=torch.float32,device=device)
        self.cam_intr = cam_intr #fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.dtype=dtype
        self.device=device

        self.camera_mat = torch.zeros([2, 2], dtype=torch.float32, device=device)
        self.camera_mat[0, 0] = self.cam_intr[0]
        self.camera_mat[1, 1] = self.cam_intr[1]
        self.camera_transform = torch.eye(4, dtype=torch.float32, device=device)
        self.center = torch.zeros([1, 2], dtype=torch.float32, device=device)
        self.center[:,0] = self.cam_intr[2]
        self.center[:,1] = self.cam_intr[3]

        self.robustifier = utils.GMoF(rho=rho)


    def projectJoints(self, points):
        device = points.device

        points = (self.floor2color[:3, :3] @ (points.T) +
                   self.floor2color[:3, 3].reshape([3, 1])).T

        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)

        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)
        projected_points = torch.einsum('ki,ji->jk',
                                        [self.camera_transform, points_h])
        img_points = torch.div(projected_points[:, :2],
                               projected_points[:, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('ki,ji->jk', [self.camera_mat, img_points]) \
            + self.center

        return img_points


    def calcColorLoss(self, keypoints=None,points=None):
        projected_joints = self.projectJoints(points)
        ic(projected_joints,projected_joints.shape)
        exit()

        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)
        depth_loss=0
        return depth_loss
