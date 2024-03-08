import pickle
import trimesh
import os.path as osp

from typing import NewType, Union, Optional
import numpy as np
import torch.nn as nn
import torch
from dataclasses import dataclass
from trimesh import load_mesh

from smplx.utils import Struct,to_np, to_tensor
from smplx.vertex_joint_selector import VertexJointSelector
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from smplx.lbs import batch_rodrigues,batch_rigid_transform

from smplx.utils import ModelOutput

from lib.data.constants_spin import JOINT_NAMES, JOINT_MAP, FOOT_IDS_SMPLL, FOOT_IDS_SMPLR

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


@dataclass
class SMPLMMVPOuput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    foot_plane: Optional[Tensor] = None
    model_scale_opt : Optional[Tensor] = None
    

class SMPL_MMVP(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(self,
                 essential_root: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 stage: str = 'init_shape',
                 bs: int = 1,
                 dtype=torch.float32):
        super(SMPL_MMVP, self).__init__()
        
        self.essential_root = essential_root
        
        model_root = osp.join(essential_root, 'bodyModels', 'smpl')

        if osp.isdir(model_root):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_root, model_fn)
            if not osp.exists(smpl_path):
                print(f'{smpl_path} for smpl not exist!')
                raise FileNotFoundError
        else:
            print(f'{model_root} for smpl not exist!')
            raise IsADirectoryError
        with open(smpl_path, 'rb') as smpl_file:
            data_struct = Struct(**pickle.load(smpl_file,
                                               encoding='latin1'))

        self.dtype = dtype
        self.num_betas = num_betas
        self.bs = bs
        
        self.faces = data_struct.f
        
        # init smpl essentials
        shapedirs = data_struct.shapedirs
        shapedirs = shapedirs[:, :, :num_betas]
        self.register_buffer('shapedirs',to_tensor(to_np(shapedirs), dtype=self.dtype))

        v_template = to_tensor(to_np(data_struct.v_template), dtype=self.dtype)
        self.register_buffer('smpl_v_template', v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=self.dtype)
        self.register_buffer('lbs_weights', lbs_weights)

        self.J_shaped, self.v_shaped = None, None

        # init smpl baby essentials
        baby_mesh = load_mesh(f'{essential_root}/smplify_essential/smil_template_fingers_fix.ply', process=False)
        smil_v_template = to_tensor(baby_mesh.vertices, dtype=self.dtype)
        self.register_buffer('smil_v_template', smil_v_template)

        """ Extension of the official SMPL implementation to support more joints """
        # load extra 9 joints

        J_regressor_extra = np.load(f'{model_root}/J_regressor_extra.npy')
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=self.dtype))
        
        # load joints mapping to remap smpl joints
        joints_ids = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.register_buffer('joint_map', torch.tensor(joints_ids, dtype=torch.long))

        # load smpl extra 21 joints according to official code, 24+21=45
        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids)
        
        """ load foot data for dense contact control """
        self.foot_ids_smplL, self.foot_ids_smplR = FOOT_IDS_SMPLL, FOOT_IDS_SMPLL
        # TODO: load faces data for norm calculation in future version
        
        # init smpl param
        self.init_param(stage)
            
        
    @torch.no_grad()
    def setPose(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = params_dict[param_name].clone().detach()    

        
    def init_param(self, stage):
        # SMPL parameter
        
        # Add extra shape param and weight to control body scale for different ages
        betas = torch.zeros([1, self.num_betas+1], dtype=self.dtype)# scale and betas
        betas[:,0] = 1
        self.register_parameter('betas',nn.Parameter(betas))
        self.betas.requires_grad = True if stage=='init_shape' else False
        
        model_scale_opt = torch.tensor([1.0], dtype=self.dtype)
        self.register_parameter('model_scale_opt',nn.Parameter(model_scale_opt))
        self.model_scale_opt.requires_grad = True if stage=='init_shape' else False
        
        # init body pose, global rot and transl        
        body_pose = torch.zeros([1,self.NUM_JOINTS*3], dtype=self.dtype)
        self.register_parameter('body_pose', nn.Parameter(body_pose, requires_grad=True))
        transl = torch.zeros([self.bs, 3],dtype=self.dtype)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        global_orient = torch.zeros([self.bs, 3],dtype=self.dtype)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))
    
    def update_shape(self):
        # update v_template
        self.v_template = ((self.smpl_v_template * self.model_scale_opt) +\
            (1. - self.model_scale_opt) * self.smil_v_template).to(self.shapedirs.device)
        
        blend_shape = torch.einsum('bl,mkl->bmk', [self.betas[:,1:], self.shapedirs])
        v_shaped = self.v_template + blend_shape
        self.v_shaped = v_shaped * self.betas[:,0]
        self.J_shaped = self.vertices2joints(self.J_regressor, self.v_shaped)
        # return v_shaped,J_shaped
        
    def vertices2joints(self, J_regressor, vertices):
        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


    def init_plane(self):
        assert self.v_shaped != None, "must init smpl shape first"

        # seperate foots into two part
        self.foot_front_r_ids = [i for i in range(36)] + [36, 41, 42, 47, 48, 49]
        self.foot_back_r_ids = [60, 61, 62, 69, 70, 71] + [i for i in range(72, 96)]
        self.foot_front_l_ids = self.foot_front_r_ids
        self.foot_back_l_ids = self.foot_back_r_ids
        
        # construct y offset in smpl foots
        v_smpl_foot_L, v_smpl_foot_R =\
            self.v_shaped[0][self.foot_ids_smplL, :] ,self.v_shaped[0][self.foot_ids_smplR, :]
    
        y_smpl_foot_L = v_smpl_foot_L[:, 1]
        y_smpl_foot_R = v_smpl_foot_R[:, 1]            
        floor_L, floor_R = torch.min(y_smpl_foot_L), torch.min(y_smpl_foot_R)
        # offset_L, offset_R = (y_smpl_foot_L - floor_L).tolist(), (y_smpl_foot_R - floor_R).tolist()
        self.plane_L_init, self.plane_R_init =\
            torch.zeros_like(v_smpl_foot_L, device=self.v_shaped.device),\
            torch.zeros_like(v_smpl_foot_R, device=self.v_shaped.device)

        self.plane_L_init[:, 0], self.plane_L_init[:, 2] = v_smpl_foot_L[:, 0], v_smpl_foot_L[:, 2]
        temp = [floor_L.unsqueeze(0) for i in range(self.plane_L_init.shape[0])]
        self.plane_L_init[:, 1] = torch.concat(temp)
        
        self.plane_R_init[:, 0], self.plane_R_init[:, 2] = v_smpl_foot_R[:, 0], v_smpl_foot_R[:, 2]
        temp = [floor_R.unsqueeze(0) for i in range(self.plane_R_init.shape[0])]
        self.plane_R_init[:, 1] = torch.concat(temp)

        self.foot_ids_back_smplL =  np.array(self.foot_ids_smplL)[self.foot_back_l_ids].tolist()
        self.foot_ids_back_smplR = np.array(self.foot_ids_smplR)[self.foot_back_r_ids].tolist()
        self.foot_ids_front_smplL = np.array(self.foot_ids_smplL)[self.foot_front_l_ids].tolist()
        self.foot_ids_front_smplR = np.array(self.foot_ids_smplR)[self.foot_front_r_ids].tolist()

        self.plane_back_L_init = self.plane_L_init[self.foot_back_l_ids, :]
        self.plane_back_R_init = self.plane_R_init[self.foot_back_r_ids, :]
        self.plane_front_L_init = self.plane_L_init[self.foot_front_l_ids, :]
        self.plane_front_R_init = self.plane_R_init[self.foot_front_r_ids, :]

    def update_pose(self,body_pose=None):
        if body_pose == None:
            body_pose = self.body_pose
            
        full_pose = torch.cat([self.global_orient, body_pose], dim=1)

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
            [self.bs, -1, 3, 3])
        J_transformed, A = batch_rigid_transform(rot_mats, self.J_shaped,
                                                 self.parents, dtype=self.dtype)    

        W = self.lbs_weights.unsqueeze(dim=0).expand([self.bs, -1, -1])
        num_joints = self.NUM_BODY_JOINTS+1
        
        T = torch.matmul(W, A.view(self.bs, num_joints, 16)) \
            .view(self.bs, -1, 4, 4)

        homogen_coord = torch.ones([self.bs, self.v_shaped.shape[1], 1],
                                   dtype=self.dtype).to(body_pose.device)
        v_posed_homo = torch.cat([self.v_shaped, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        
        verts = v_homo[:, :, :3, 0]

        # add extra joints in origin smpl
        joints = self.vertex_joint_selector(verts, J_transformed)
        # apply extra joints in SPIN
        extra_joints = self.vertices2joints(self.J_regressor_extra, verts)        
        
        joints_54 = torch.cat([joints, extra_joints], dim=1)
        joints_49 = joints_54[:, self.joint_map, :]
        
        # plane transform
        v_plane = self.update_plane(T=T)         
        
        # apply trans
        verts += self.transl.unsqueeze(dim=1)
        joints_49 += self.transl.unsqueeze(dim=1)
        v_plane += self.transl.unsqueeze(dim=1)
        joints_54 += self.transl.unsqueeze(dim=1)
        return SMPLMMVPOuput(
                             joints=joints_49,
                             vertices=verts,
                             body_pose=self.body_pose,
                             foot_plane=v_plane,
                             betas=self.betas,
                             model_scale_opt=self.model_scale_opt
                             )


    def update_plane(self, T):
        device = T.device
        
        T_back_l = T[:, self.foot_ids_back_smplL, :, :]
        T_back_r = T[:, self.foot_ids_back_smplR, :, :]
        T_front_l = T[:, self.foot_ids_front_smplL, :, :]
        T_front_r = T[:, self.foot_ids_front_smplR, :, :] 

        # back left foot plane
        homogen_coord_back_l = torch.ones([1, self.plane_back_L_init.shape[0], 1],
                                          dtype=self.dtype).to(device)
        
        v_plane_homo_back_l = torch.cat([self.plane_back_L_init.expand(1, -1, -1),
                                               homogen_coord_back_l], dim=2)
        v_plane_back_l = torch.matmul(T_back_l, v_plane_homo_back_l.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        # back right foot plane
        homogen_coord_back_r = torch.ones([1, self.plane_back_R_init.shape[0], 1],
                                          dtype=self.dtype).to(device)
        v_plane_homo_back_r = torch.cat([self.plane_back_R_init.expand(1, -1, -1),
                                               homogen_coord_back_r], dim=2)
        v_plane_back_r = torch.matmul(T_back_r, v_plane_homo_back_r.unsqueeze(-1)).squeeze(-1)[:, :, :3]

        # front right foot plane
        homogen_coord_front_r = torch.ones([1, self.plane_front_R_init.shape[0], 1],
                                           dtype=self.dtype).to(device)
        v_plane_homo_front_r = torch.cat([self.plane_front_R_init.expand(1, -1, -1),
                                               homogen_coord_front_r], dim=2)
        v_plane_front_r = torch.matmul(T_front_r, v_plane_homo_front_r.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        # front left foot plane
        homogen_coord_front_l = torch.ones([1, self.plane_front_L_init.shape[0], 1],
                                           dtype=self.dtype).to(device)
        v_plane_homo_front_l = torch.cat([self.plane_front_L_init.expand(1, -1, -1),
                                               homogen_coord_front_l], dim=2)
        v_plane_front_l = torch.matmul(T_front_l, v_plane_homo_front_l.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        v_plane = torch.concat([v_plane_back_l, v_plane_front_l, v_plane_back_r, v_plane_front_r], dim = 1)
        
        return v_plane