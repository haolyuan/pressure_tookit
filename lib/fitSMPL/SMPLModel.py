from typing import Optional, Dict, Union
import pickle
import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import trimesh
from smplx.utils import Struct,to_np, to_tensor
from smplx.lbs import batch_rodrigues,batch_rigid_transform
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.vertex_joint_selector import VertexJointSelector

from icecream import ic


from lib.Utils.fileio import saveJointsAsOBJ

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


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
        betas[:,0] = 1
        self.register_parameter('betas',nn.Parameter(betas, requires_grad=False))
        body_pose = torch.zeros([1,self.NUM_JOINTS*3], dtype=dtype)
        self.register_parameter('body_pose', nn.Parameter(body_pose, requires_grad=False))
        body_poseZ = torch.zeros([1,32], dtype=dtype)
        self.register_parameter('body_poseZ', nn.Parameter(body_poseZ, requires_grad=True))
        transl = torch.zeros([1,3],dtype=dtype)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        global_orient = torch.zeros([1,3],dtype=dtype)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.J_shaped,self.v_shaped = None,None

        """ Extension of the official SMPL implementation to support more joints """
        # load extra 9 joints
        J_regressor_extra = np.load('../../bodyModels/smpl/J_regressor_extra.npy')
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        
        # load joints mapping to remap smpl joints
        joints_ids = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.joint_map = torch.tensor(joints_ids, dtype=torch.long)
        
        # load smpl extra 21 joints according to official code, 24+21=45
        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids)
        
        
    def updateShape(self):
        blend_shape = torch.einsum('bl,mkl->bmk', [self.betas[:,1:], self.shapedirs])
        v_shaped = self.v_template + blend_shape
        self.v_shaped = v_shaped * self.betas[:,0]
        self.J_shaped = self.vertices2joints(self.J_regressor, self.v_shaped)
        # return v_shaped,J_shaped

    def vertices2joints(self, J_regressor, vertices):
        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

    def updatePose(self,body_pose=None,batch_size=1):#v_shaped,J,body_pose,global_orient,
        if body_pose == None:
            body_pose = self.body_pose
        full_pose = torch.cat([self.global_orient, body_pose], dim=1)
        device = self.v_shaped.device

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        J_transformed, A = batch_rigid_transform(rot_mats, self.J_shaped,
                                                 self.parents, dtype=self.dtype)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        num_joints = self.NUM_BODY_JOINTS+1

        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, self.v_shaped.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([self.v_shaped, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]
        
        # add extra joints in origin smpl
        joints = self.vertex_joint_selector(verts, J_transformed)
        # apply extra joints in SPIN
        extra_joints = self.vertices2joints(self.J_regressor_extra, verts)
        joints_54 = torch.cat([joints, extra_joints], dim=1)
        joints_54 = joints_54[:, self.joint_map, :]
        
        # apply trans
        verts += self.transl.unsqueeze(dim=1)
        joints_54 += self.transl.unsqueeze(dim=1)
        
        return verts, joints_54

    @torch.no_grad()
    def setPose(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])





