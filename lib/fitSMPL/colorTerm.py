import torch
import copy
import trimesh
import torch.nn as nn
import cv2
import trimesh
import numpy as np
from icecream import ic

from lib.visualizer.renderer import modelRender
from lib.Utils.smpl2openpose import smpl_to_openpose,JointMapper
from lib.Utils.fileio import saveProjectedJoints
# saveProjectedJoints(filename='debug/2d+.png',img=img,joint_projected=projected_joints)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist

class ColorTerm(nn.Module):
    def __init__(self,depth2color=None,
                 depth2floor=None,
                 cam_intr=None,
                 img_W=1280,img_H=720,
                 rho=100,
                 dtype=np.float32,
                 device='cpu'):
        super(ColorTerm, self).__init__()

        self.depth2floor = depth2floor
        self.floor2depth = np.linalg.inv(depth2floor)
        self.depth2color = depth2color
        self.floor2color_cpu = self.depth2color @ self.floor2depth
        self.floor2color = torch.tensor(self.floor2color_cpu,dtype=torch.float32,device=device)
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

        smpl2kp = smpl_to_openpose(
            model_type='smpl', use_hands=False,
            use_face=False,use_face_contour=False,
            openpose_format='coco25')
        self.smpl2kp = torch.tensor(smpl2kp,device=device).long()
        self.joint_weights = self.get_joint_weights()
        self.joint_mapper = JointMapper(self.smpl2kp)

        self.robustifier = GMoF(rho=rho)

        self.renderer = modelRender(cam_intr, img_W, img_H)


    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(25,dtype=np.float32)
        # These joints are ignored becaue SMPL has no neck.
        optim_weights[1] = 0.
        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[3,6,10,13,4,7]] = 2
        return torch.tensor(optim_weights, dtype=self.dtype,device=self.device)

    def get_model2data(self):
        return self.smpl2kp

    def renderMesh(self, mesh, img=None):
        mesh.apply_transform(self.floor2color_cpu)
        color_render, depth = self.renderer.render(mesh,img=img)
        # cv2.imwrite('debug/test.png',color_render)
        return color_render, depth

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


    def calcColorLoss(self, keypoints=None,points=None,img=None):
        points = self.joint_mapper(points)
        projected_joints = self.projectJoints(points)

        # Calculate the weights for each joints
        keypoint_data = torch.tensor(keypoints, dtype=torch.float32,
                                     device=self.device)
        gt_joints = keypoint_data[:, :2]
        joints_conf = keypoint_data[:, 2]#.reshape(1, -1)
        weights = (self.joint_weights * joints_conf).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        # joint_loss = (torch.sum(weights ** 2 * joint_diff) *
        #               self.data_weight ** 2)
        joint_loss = torch.sum(weights ** 2 * joint_diff)
        return joint_loss