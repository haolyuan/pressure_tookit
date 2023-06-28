from typing import Optional, Dict, Union
import pickle
import torch
import torch.nn as nn
import os.path as osp
from smplx.utils import Struct,to_np, to_tensor
from icecream import ic

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

        self._num_betas = num_betas
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
        betas = torch.zeros([self.num_betas], dtype=dtype)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))
        body_pose = torch.zeros([self.NUM_JOINTS*3], dtype=dtype)
        self.register_parameter('body_pose', nn.Parameter(body_pose, requires_grad=True))
        transl = torch.zeros([4,],dtype=dtype,requires_grad=True)#transl+scale
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        global_orient = torch.zeros([3,],dtype=dtype,requires_grad=True)
        self.register_parameter('global_orient', global_orient)


    def initShape(self):
        ic()
