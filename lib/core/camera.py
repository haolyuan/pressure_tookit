import numpy as np
import os.path as osp
import torch
import torch.nn as nn
from xrprimer.utils.path_utils import Existence, check_path_existence

from lib.utils.depth_utils import depth2PointCloud  # (depth_map,fx,fy,cx,cy)


def create_camera(
    basdir,
    dataset_name,
    sub_ids,
    seq_name,
):
    calibration_path = osp.join(basdir, 'images', dataset_name, sub_ids,
                                seq_name, 'calibration.npy')
    floor_path = osp.join(basdir, 'annotations', dataset_name, 'floor_info',
                          'floor_' + sub_ids + '.npy')
    if check_path_existence(calibration_path) == Existence.FileNotExist:
        print(f'{calibration_path} not exist')
        raise FileExistsError
    if check_path_existence(floor_path) == Existence.FileNotExist:
        print(f'{floor_path} not exist')
        raise FileExistsError
    return RgbdCamera(cali_path=calibration_path, floor_path=floor_path)


def loadCalibrationFromNpy(cali_path):
    cali_data = dict(np.load(cali_path, allow_pickle=True).item())
    color_Intr = cali_data['color_Intr']
    depth_Intr = cali_data['depth_Intr']

    d2c = cali_data['d2c']
    d2c[:3, 3] /= 1000.

    return np.array([color_Intr['fx'], color_Intr['fy'],
                     color_Intr['cx'], color_Intr['cy']]), \
        np.array([depth_Intr['fx'], depth_Intr['fy'],
                  depth_Intr['cx'], depth_Intr['cy']]), \
        d2c


def loadFloorFromNpy(floor_path):
    # load floor info
    floor_info = np.load(floor_path, allow_pickle=True).item()
    return floor_info['trans'], floor_info['normal'], floor_info['depth2floor']


class RgbdCamera(nn.Module):

    def __init__(self, cali_path, floor_path, dtype=torch.float32):
        super(RgbdCamera, self).__init__()

        self.dtype = dtype
        # load data for color&depth transfer
        self.cIntr_cpu, self.dIntr_cpu, depth2color_cpu =\
            loadCalibrationFromNpy(cali_path)
        # load data for depth&floor transfer
        floor_trans_cpu, floor_normal_cpu, depth2floor_cpu = loadFloorFromNpy(
            floor_path)

        # intr: fx,fy,cx,cy
        cIntr = torch.tensor(self.cIntr_cpu, dtype=self.dtype)
        self.register_parameter('cIntr',
                                nn.Parameter(cIntr, requires_grad=False))
        dIntr = torch.tensor(self.dIntr_cpu, dtype=self.dtype)
        self.register_parameter('dIntr',
                                nn.Parameter(dIntr, requires_grad=False))

        # floor data
        floor_trans = nn.Parameter(
            torch.tensor(floor_trans_cpu, dtype=self.dtype),
            requires_grad=False)
        self.register_parameter('floor_trans', floor_trans)
        floor_normal = nn.Parameter(
            torch.tensor(floor_normal_cpu, dtype=self.dtype),
            requires_grad=False)
        self.register_parameter('floor_normal', floor_normal)

        # 3d transform
        depth2color = nn.Parameter(
            torch.tensor(depth2color_cpu, dtype=self.dtype),
            requires_grad=False)
        self.register_parameter('depth2color', depth2color)
        depth2floor = nn.Parameter(
            torch.tensor(depth2floor_cpu, dtype=self.dtype),
            requires_grad=False)
        self.register_parameter('depth2floor', depth2floor)
        floor2color_cpu = depth2color_cpu @ np.linalg.inv(depth2floor_cpu)
        floor2color = nn.Parameter(
            torch.tensor(floor2color_cpu, dtype=self.dtype),
            requires_grad=False)
        self.register_parameter('floor2color', floor2color)

        # construct camera matrix
        cIntr_mat = torch.zeros([2, 2], dtype=dtype)
        cIntr_mat[0, 0], cIntr_mat[1, 1] = cIntr[0], cIntr[1]
        cIntr_transform = torch.eye(4, dtype=dtype)
        center = torch.zeros([1, 2], dtype=dtype)
        center[:, 0], center[:, 1] = cIntr[2], cIntr[3]
        cIntr_mat = nn.Parameter(cIntr_mat, requires_grad=False)
        self.register_parameter('cIntr_mat', cIntr_mat)
        cIntr_transform = nn.Parameter(cIntr_transform, requires_grad=False)
        self.register_parameter('cIntr_transform', cIntr_transform)
        center = nn.Parameter(center, requires_grad=False)
        self.register_parameter('center', center)

    def projectJoints(self, points):
        points = self.tranform3d(points=points, type='f2c')

        homog_coord = torch.ones(
            list(points.shape)[:-1] + [1],
            dtype=points.dtype).to(points.device)

        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)
        projected_points = torch.einsum('ki,ji->jk',
                                        [self.cIntr_transform, points_h])
        img_points = torch.div(projected_points[:, :2],
                               projected_points[:, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('ki,ji->jk', [self.cIntr_mat, img_points]) \
            + self.center

        return img_points

    def tranform3d(self, points, type):
        if not torch.is_tensor(points):
            points = torch.tensor(
                points, dtype=self.dtype).to(self.depth2color.device)
        if len(points.shape) == 3:
            points = points.squeeze(0)

        if type == 'd2c':
            transform_mat = self.depth2color
        elif type == 'c2d':
            transform_mat = torch.linalg.inv(self.depth2color)
        elif type == 'd2f':
            transform_mat = self.depth2floor
        elif type == 'f2d':
            transform_mat = torch.linalg.inv(self.depth2floor)
        elif type == 'f2c':
            transform_mat = self.floor2color
        elif type == 'c2f':
            transform_mat = torch.linalg.inv(self.floor2color)
        else:
            print('undefined spatial transform')
            raise TypeError

        return (transform_mat[:3, :3] @ (points.T) +
                transform_mat[:3, 3].reshape([3, 1])).T

    # ~~~ operation related to depth camera  ~~~ #
    def preprocessDepth(self, depth_map, mask_map):
        _mask_map = mask_map.reshape(-1)
        _depth_map = depth_map.reshape(-1)
        idx_valid = np.where((_mask_map > 0.5) & (_depth_map > 1e-6))[0]

        # depth_vraw = self.calcDepth3D(depth_map)
        depth_vraw = depth2PointCloud(depth_map, self.dIntr_cpu[0],
                                      self.dIntr_cpu[1], self.dIntr_cpu[2],
                                      self.dIntr_cpu[3])
        depth_nraw = self.calcDepthRawNormal(depth_vraw)
        depth_vraw = depth_vraw.reshape([-1, 3])
        depth_nraw = depth_nraw.reshape([-1, 3])
        # trimesh.Trimesh(vertices=depth_vraw).export('debug/new_framework/depth_vraw.obj')

        dv_valid_cpu = depth_vraw[idx_valid, :]
        dn_valid_cpu = depth_nraw[idx_valid, :]

        # transform depth data from depth space to floor space
        dv_floor = self.tranform3d(dv_valid_cpu, type='d2f')
        if dn_valid_cpu is not None:
            dn_valid = torch.from_numpy(dn_valid_cpu).to(
                device=dv_floor.device, dtype=self.dtype)
            dn_floor = (self.depth2floor[:3, :3] @ dn_valid.T).T

        # return depth_floor, depth_normal
        return torch.from_numpy(dv_valid_cpu).\
            to(device=dv_floor.device, dtype=self.dtype), \
            torch.from_numpy(dn_valid_cpu).\
            to(device=dv_floor.device, dtype=self.dtype), \
            dv_floor, dn_floor

    def calcDepthRawNormal(self, depth_vraw):
        v0 = depth_vraw[:-2, 1:-1, :]
        v1 = depth_vraw[2:, 1:-1, :]
        v2 = depth_vraw[1:-1, :-2, :]
        v3 = depth_vraw[1:-1, 2:, :]

        E0 = self.matNormalized(v1 - v0)
        E1 = self.matNormalized(v3 - v2)

        nt = np.cross(E0, E1, axisa=-1, axisb=-1)
        n = self.matNormalized(nt)

        depth_nraw = np.zeros([depth_vraw.shape[0], depth_vraw.shape[1], 3])
        depth_nraw[1:-1, 1:-1] = n
        return depth_nraw

    def matNormalized(self, mat):
        mat_normal2 = np.linalg.norm(mat, axis=-1)
        mask_mat_normal2 = np.argwhere(mat_normal2 < 1e-6)
        mat[mask_mat_normal2[:, 0], mask_mat_normal2[:, 1]] = np.nan
        mat_normal2[mask_mat_normal2[:, 0], mask_mat_normal2[:, 1]] = 1e-6
        mat_normal2 = (mat_normal2[..., None]).repeat(3, -1)
        mat = mat / mat_normal2
        mat = np.nan_to_num(mat)
        return mat
