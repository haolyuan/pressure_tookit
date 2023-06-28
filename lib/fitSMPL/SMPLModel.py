from typing import Optional, Dict, Union
import pickle
import torch
import torch.nn as nn
import os.path as osp
import trimesh
from smplx.utils import Struct,to_np, to_tensor
from smplx.lbs import batch_rodrigues,batch_rigid_transform
from icecream import ic

from lib.Utils.fileio import saveJointsAsOBJ

class SMPLModel(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300
    def __init__(self,
                 model_path: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 dtype=torch.float32,):
        super(SMPLModel, self).__init__()

        if osp.isdir(model_path):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_path, model_fn)
        with open(smpl_path, 'rb') as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file,
                                               encoding='latin1'))

        self.dtype = dtype
        self.num_betas = num_betas
        self.faces = data_struct.f

        shapedirs = data_struct.shapedirs
        shapedirs = shapedirs[:, :, :num_betas]
        self.register_buffer('shapedirs',to_tensor(to_np(shapedirs), dtype=dtype))

        v_template = to_tensor(to_np(data_struct.v_template), dtype=dtype)
        self.register_buffer('v_template', v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

        #SMPL Parameter
        betas = torch.zeros([1, self.num_betas+1], dtype=dtype)# scale and betas
        betas[0] = 1
        self.register_parameter('betas',nn.Parameter(betas, requires_grad=False))
        body_pose = torch.zeros([1,self.NUM_JOINTS*3], dtype=dtype)
        self.register_parameter('body_pose', nn.Parameter(body_pose, requires_grad=True))
        transl = torch.zeros([1,3],dtype=dtype)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        global_orient = torch.zeros([1,3],dtype=dtype)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

    def updateShape(self, betas):
        blend_shape = torch.einsum('bl,mkl->bmk', [betas[:,1:], self.shapedirs])
        v_shaped = self.v_template + blend_shape
        v_shaped *= betas[:,0]
        J_shaped = self.vertices2joints(self.J_regressor, v_shaped)
        return v_shaped,J_shaped

    def vertices2joints(self, J_regressor, vertices):
        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

    def updatePose(self,v_shaped,J,body_pose,global_orient,batch_size=1):
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        device = v_shaped.device

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        J_transformed, A = batch_rigid_transform(rot_mats, J,
                                                 self.parents, dtype=self.dtype)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        num_joints = self.NUM_BODY_JOINTS+1

        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_shaped.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_shaped, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]
        return verts, J_transformed

    def initShape(self,depth_scan,keypoints):
        v_shaped, J_shaped = m_smpl.updateShape(betas=beta)
        verts, _ = self.updatePose(v_shaped, J_shaped, body_pose, global_orient)

        # trimesh.Trimesh(vertices=v_shaped.detach().cpu().numpy()[0],
        #                 faces=self.faces, process=False).export('debug/v_shaped.obj')


