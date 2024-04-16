import cv2
import numpy as np
import torch
import torch.nn as nn

from lib.utils.render_utils import modelRender


class GMoF(nn.Module):

    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual**2
        dist = torch.div(squared_res, squared_res + self.rho**2)
        return self.rho**2 * dist


class ColorTerm(nn.Module):

    def __init__(self,
                 cam_intr,
                 img_W=1280,
                 img_H=720,
                 rho=100,
                 dtype=np.float32,
                 device='cpu'):
        super(ColorTerm, self).__init__()

        self.robustifier = GMoF(rho=rho)

        self.renderer = modelRender(cam_intr, img_W, img_H)

    # TODO: render 放在一个功能中
    def renderMesh(self, mesh, img=None):
        # mesh.apply_transform(self.floor2color_cpu)
        color_render, depth = self.renderer.render(mesh, img=img)
        cv2.imwrite('debug/new_framework/test.png', color_render)
        return color_render, depth

    def forward(self, keypoint_data, projected_joints, joint_weights):

        # Calculate the weights for each joints

        gt_joints = keypoint_data[:, :2]
        joints_conf = keypoint_data[:, 2]

        weights = (joint_weights * joints_conf)
        weights = weights / torch.sum(weights)
        weights = weights.unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = torch.sum(weights**2 * joint_diff)

        return joint_loss
