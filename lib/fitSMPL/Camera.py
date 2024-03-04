from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from smplx.lbs import transform_mat
import trimesh, cv2
from lib.Utils.depth_utils import depth2PointCloud#(depth_map,fx,fy,cx,cy)


class RGBDCamera(nn.Module):

    def __init__(self,
                 basdir,
                 dataset_name,
                 sub_ids,
                 seq_name):
        super(RGBDCamera, self).__init__()
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.dtype = torch.float32
        self.basdir = basdir
        self.dataset_name = dataset_name
        self.sub_ids = sub_ids
        self.seq_name = seq_name
        cali_path = osp.join(basdir,self.dataset_name,self.sub_ids,self.seq_name,'calibration.npy')
        # intr: fx,fy,cx,cy,
        self.cIntr_cpu, self.dIntr_cpu, self.d2c_cpu = self.loadCalibrationFromNpy(cali_path) #loadCalibrationFromTxt
        cIntr = nn.Parameter(torch.tensor(self.cIntr_cpu,dtype=self.dtype), requires_grad=False)
        self.register_parameter('cIntr', cIntr)
        dIntr = nn.Parameter(torch.tensor(self.dIntr_cpu, dtype=self.dtype), requires_grad=False)
        self.register_parameter('dIntr', dIntr)
        d2c = nn.Parameter(torch.tensor(self.d2c_cpu, dtype=self.dtype), requires_grad=False)
        self.register_parameter('d2c', d2c)

    def loadCalibrationFromNpy(self, cali_path):
        cali_data = dict(np.load(cali_path, allow_pickle=True).item())
        color_Intr = cali_data['color_Intr']
        depth_Intr = cali_data['depth_Intr']
        
        d2c = cali_data['d2c']
        d2c[:3,3]/=1000.
        
        return np.array([color_Intr['fx'], color_Intr['fy'], color_Intr['cx'], color_Intr['cy']]),\
            np.array([depth_Intr['fx'], depth_Intr['fy'], depth_Intr['cx'], depth_Intr['cy']]),\
            d2c
        
    def loadCalibrationFromTxt(self,cali_path):
        fp = open(cali_path)
        lines = fp.readlines()

        cIntr = (lines[0]).split(' ')[:-1]
        cIntr = [float(x) for x in cIntr]
        dIntr = (lines[1]).split(' ')[:-1]
        dIntr = [float(x) for x in dIntr]

        d2c = []
        for i in range(2,len(lines)):
            d2c_line = (lines[i]).split(' ')
            d2c_line = [float(x) for x in d2c_line]
            d2c+=d2c_line
        d2c+=[0,0,0,1]
        d2c = np.array(d2c).reshape([4,4])
        d2c[:3,3]/=1000.
        return np.array([cIntr[2],cIntr[3],cIntr[0],cIntr[1]]),\
               np.array([dIntr[2],dIntr[3],dIntr[0],dIntr[1]]),\
               d2c

    def matNormalized(self,mat):
        mat_normal2 = np.linalg.norm(mat, axis=-1)
        mask_mat_normal2 = np.argwhere(mat_normal2 < 1e-6)
        mat[mask_mat_normal2[:, 0], mask_mat_normal2[:, 1]] = np.nan
        mat_normal2[mask_mat_normal2[:, 0], mask_mat_normal2[:, 1]] = 1e-6
        mat_normal2 = (mat_normal2[..., None]).repeat(3, -1)
        mat = mat / mat_normal2
        mat = np.nan_to_num(mat)
        return mat

    # def calcDepth3D(self,depth_map):
    #     z = depth_map
    #     r = np.arange(depth_map.shape[0])
    #     c = np.arange(depth_map.shape[1])
    #     x, y = np.meshgrid(c, r)
    #     x = (x - self.dIntr_cpu[0]) / self.dIntr_cpu[2] * z
    #     y = (y - self.dIntr_cpu[1]) / self.dIntr_cpu[3] * z
    #     pointCloud = np.dstack([x, y, z])
    #     return pointCloud

    def calcDepthRawNormal(self,depth_vraw):
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

    def preprocessDepth(self,depth_map,mask_map):

        _mask_map = mask_map.reshape(-1)
        _depth_map = depth_map.reshape(-1)
        idx_valid = np.where((_mask_map > 0.5) & (_depth_map > 1e-6))[0]

        # depth_vraw = self.calcDepth3D(depth_map)
        depth_vraw = depth2PointCloud(depth_map,self.dIntr_cpu[0],self.dIntr_cpu[1],self.dIntr_cpu[2],self.dIntr_cpu[3])
        depth_nraw = self.calcDepthRawNormal(depth_vraw)
        depth_vraw = depth_vraw.reshape([-1,3])
        depth_nraw = depth_nraw.reshape([-1,3])

        dv_valid = depth_vraw[idx_valid,:]
        dn_valid = depth_nraw[idx_valid,:]
        return dv_valid,dn_valid