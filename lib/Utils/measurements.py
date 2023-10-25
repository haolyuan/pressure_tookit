# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Author: Vasileios Choutas, vassilis.choutas@tuebingen.mpg.de
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import yaml

import torch
import torch.nn as nn
import numpy as np
from smplx import SMPLX

class MeasurementsLoss(nn.Module):

    # The density of the human body is 985 kg / m^3
    DENSITY = 985

    def __init__(self, faces, **kwargs):
        ''' Loss that penalizes deviations in weight and height
        '''
        super(MeasurementsLoss, self).__init__()
        
        self.faces = torch.tensor(faces.astype(np.int32))

    def compute_height(self, live_verts):
        ''' Compute the height using the heel and the top of the head
        '''
        live_y = live_verts[:, :, 1]
        y_max = torch.max(live_y)

        return torch.abs(y_max)

    def compute_mass(self, tris):
        ''' Computes the mass from volume and average body density
        '''
        x = tris[:, :, :, 0]
        y = tris[:, :, :, 1]
        z = tris[:, :, :, 2]
        volume = (
            -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
            x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
            x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
            x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
            x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
            x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
        ).sum(dim=1).abs() / 6.0
        return volume * self.DENSITY

    def forward(self, v, **kwargs):
        batch_size = v.shape[0]


        mesh_height = self.compute_height(v)
        
        self.faces = self.faces.to(v.device)
        v_triangles = torch.index_select(
            v, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)
        mesh_mass = self.compute_mass(v_triangles)

        measurements = {'mass': mesh_mass, 'height': mesh_height}

        return measurements
