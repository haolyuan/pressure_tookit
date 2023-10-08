import pickle
import torch
import torch.nn as nn
import os.path as osp
import numpy as np
import trimesh

from smplx.utils import Struct,to_np, to_tensor
from smplx.lbs import batch_rodrigues,batch_rigid_transform

NUM_JOINTS = 23
NUM_BODY_JOINTS = 23
SHAPE_SPACE_DIM = 300

class smpl_model_refined(nn.Module):

    def __init__(self,
                 model_root: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 bs: int = 1,
                 dtype=torch.float32):
        super(smpl_model_refined, self).__init__()

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
        self.faces = data_struct.f
        
        self.bs = bs
        
        # init foot plane vertices
        init_plane_back_l = trimesh.load('essentials/foot_related/init_plane_back_l.obj')
        init_plane_back_l_np = np.array([v for v in init_plane_back_l])
        self.register_buffer('init_plane_back_l',to_tensor(to_np(init_plane_back_l_np), dtype=dtype).expand(self.bs, -1, -1))
        init_plane_back_r = trimesh.load('essentials/foot_related/init_plane_back_r.obj')
        init_plane_back_r_np = np.array([v for v in init_plane_back_r])
        self.register_buffer('init_plane_back_r',to_tensor(to_np(init_plane_back_r_np), dtype=dtype).expand(self.bs, -1, -1))
        init_plane_front_l = trimesh.load('essentials/foot_related/init_plane_front_l.obj')
        init_plane_front_l_np = np.array([v for v in init_plane_front_l])
        self.register_buffer('init_plane_front_l',to_tensor(to_np(init_plane_front_l_np), dtype=dtype).expand(self.bs, -1, -1))        
        init_plane_front_r = trimesh.load('essentials/foot_related/init_plane_front_r.obj')
        init_plane_front_r_np = np.array([v for v in init_plane_front_r])
        self.register_buffer('init_plane_front_r',to_tensor(to_np(init_plane_front_r_np), dtype=dtype).expand(self.bs, -1, -1))        
        
        # init foot plane ids
        foot_back_l_ids = np.loadtxt('essentials/foot_related/planeL_ids_back.txt')
        self.register_buffer('foot_back_l_ids',to_tensor(foot_back_l_ids).to(torch.long))        
        foot_back_r_ids = np.loadtxt('essentials/foot_related/planeR_ids_back.txt')
        self.register_buffer('foot_back_r_ids',to_tensor(foot_back_r_ids).to(torch.long))        
        foot_front_l_ids = np.loadtxt('essentials/foot_related/planeL_ids_front.txt')
        self.register_buffer('foot_front_l_ids',to_tensor(foot_front_l_ids).to(torch.long))        
        foot_front_r_ids = np.loadtxt('essentials/foot_related/planeR_ids_front.txt')
        self.register_buffer('foot_front_r_ids',to_tensor(foot_front_r_ids).to(torch.long))        
        
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
        
        body_pose_wo_foot = torch.zeros([self.bs, (NUM_JOINTS - 4)*3], dtype=dtype)
        self.register_parameter('body_pose_wo_foot', nn.Parameter(body_pose_wo_foot, requires_grad=True))
        
        # 这里没有改
        r_back_foot_pose = torch.zeros([self.bs, 1* 3], dtype=dtype)
        self.register_parameter('r_back_foot_pose', nn.Parameter(r_back_foot_pose, requires_grad=True))
        
        r_front_foot_pose = torch.zeros([self.bs], dtype=dtype)
        self.register_parameter('r_front_foot_pose', nn.Parameter(r_front_foot_pose, requires_grad=True)) 
               
        l_back_foot_pose = torch.zeros([self.bs, 1* 3], dtype=dtype)
        self.register_parameter('l_back_foot_pose', nn.Parameter(l_back_foot_pose, requires_grad=True))
        
        l_front_foot_pose = torch.zeros([self.bs], dtype=dtype)
        self.register_parameter('l_front_foot_pose', nn.Parameter(l_front_foot_pose, requires_grad=True))         
        
        transl = torch.zeros([self.bs, 3],dtype=dtype)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))
        
        global_orient = torch.zeros([self.bs, 3],dtype=dtype)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.J_shaped, self.v_shaped = None, None

    def update_shape(self):
        blend_shape = torch.einsum('bl,mkl->bmk', [self.betas[:,1:], self.shapedirs])
        v_shaped = self.v_template + blend_shape
        

        self.v_shaped = v_shaped * self.betas[:,0]
        self.J_shaped = self.vertices2joints(self.J_regressor, self.v_shaped)
        # TODO: 脚平面实际上会根据shape的不同变化，当前的脚平面是加载的固定文件，即全零betas下的脚平面位置，不够准确
        # return v_shaped,J_shaped

    def vertices2joints(self, J_regressor, vertices):
        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

    def update_pose(self,
                   body_pose_wo_foot=None,
                   r_back_foot_pose=None,
                   r_front_foot_pose=None,
                   l_back_foot_pose=None,
                   l_front_foot_pose=None,
                   transl=None,
                   global_orient=None
                   ):#v_shaped,J,body_pose,global_orient,
        if body_pose_wo_foot == None:
            body_pose_wo_foot = self.body_pose_wo_foot
        if r_back_foot_pose == None:
            r_back_foot_pose = self.r_back_foot_pose
        if r_front_foot_pose == None:
            r_front_foot_pose = self.r_front_foot_pose
        if l_back_foot_pose == None:     
            l_back_foot_pose = self.l_back_foot_pose
        if l_front_foot_pose == None:     
            l_front_foot_pose = self.l_front_foot_pose
        if global_orient == None:
            global_orient = self.global_orient
        if transl == None:
            transl = self.transl
        
        body_pose = torch.zeros([self.bs, 23* 3], dtype=self.dtype).to(body_pose_wo_foot.device)
        
        
        # 把脚上的点额外从smpl的joints中分出来，所以要重新赋值
        body_pose[:, :6* 3] = body_pose_wo_foot[:, :6* 3]
        body_pose[:, 6* 3: 7* 3] = l_back_foot_pose
        body_pose[:, 7* 3: 8* 3] = r_back_foot_pose
        body_pose[:, 8* 3: 9* 3] = body_pose_wo_foot[:, 6* 3: 7* 3]

        body_pose[:, 9* 3] = l_front_foot_pose # : 10* 3
        body_pose[:, 10* 3] = r_front_foot_pose # : 11* 3
        body_pose[:, 11* 3:] = body_pose_wo_foot[:,  7* 3:]        
        
        full_pose = torch.cat([global_orient, body_pose], dim=1)
            
        device = self.v_shaped.device

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
            [self.bs, -1, 3, 3])

        J_transformed, A = batch_rigid_transform(rot_mats, self.J_shaped.expand(self.bs, -1, -1), self.parents, dtype=self.dtype)
        W = self.lbs_weights.unsqueeze(dim=0).expand([self.bs, -1, -1])
        num_joints = NUM_BODY_JOINTS+1

        T = torch.matmul(W, A.view(self.bs, num_joints, 16)) \
            .view(self.bs, -1, 4, 4) # (6890, 4, 4)
        
        homogen_coord = torch.ones([self.bs, self.v_shaped.shape[1], 1],
                                   dtype=self.dtype, device=device)

        v_posed_homo = torch.cat([self.v_shaped.expand(self.bs, -1, -1), homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]
        verts += transl.unsqueeze(dim=1)
        
        v_plane_back_l, v_plane_front_l, v_plane_back_r, v_plane_front_r\
            = self.update_plane(T)
        
        v_plane = torch.concat([v_plane_back_l, v_plane_front_l, v_plane_back_r, v_plane_front_r], dim = 1)
        v_plane += transl.unsqueeze(dim=1)
        # import pdb; pdb.set_trace()
        
        return verts, J_transformed, v_plane

    # update foot plane vertices
    def update_plane(self, T_homo):
        T_back_l = T_homo[:, self.foot_back_l_ids, :, :]
        T_back_r = T_homo[:, self.foot_back_r_ids, :, :]
        T_front_l = T_homo[:, self.foot_front_l_ids, :, :]
        T_front_r = T_homo[:, self.foot_front_r_ids, :, :]
        
        # back left foot plane
        homogen_coord_back_l = torch.ones([self.bs, self.init_plane_back_l.shape[1], 1], dtype=self.dtype,
                                   device=self.init_plane_back_l.device)
        v_plane_posed_homo_back_l = torch.cat([self.init_plane_back_l.expand(self.bs, -1, -1),
                                               homogen_coord_back_l], dim=2)
        v_plane_back_l = torch.matmul(T_back_l, v_plane_posed_homo_back_l.unsqueeze(-1)).squeeze(-1)
        
        # back right foot plane
        homogen_coord_back_r = torch.ones([self.bs, self.init_plane_back_r.shape[1], 1], dtype=self.dtype,
                                   device=self.init_plane_back_r.device)
        v_plane_posed_homo_back_r = torch.cat([self.init_plane_back_r.expand(self.bs, -1, -1),
                                               homogen_coord_back_r], dim=2)
        v_plane_back_r = torch.matmul(T_back_r, v_plane_posed_homo_back_r.unsqueeze(-1)).squeeze(-1)      

        # front right foot plane
        homogen_coord_front_r = torch.ones([self.bs, self.init_plane_front_r.shape[1], 1], dtype=self.dtype,
                                   device=self.init_plane_front_r.device)
        v_plane_posed_homo_front_r = torch.cat([self.init_plane_front_r.expand(self.bs, -1, -1),
                                               homogen_coord_front_r], dim=2)
        v_plane_front_r = torch.matmul(T_front_r, v_plane_posed_homo_front_r.unsqueeze(-1)).squeeze(-1)     
        
        # front left foot plane
        homogen_coord_front_l = torch.ones([self.bs, self.init_plane_front_l.shape[1], 1], dtype=self.dtype,
                                   device=self.init_plane_front_l.device)
        v_plane_posed_homo_front_l = torch.cat([self.init_plane_front_l.expand(self.bs, -1, -1),
                                               homogen_coord_front_l], dim=2)
        v_plane_front_l = torch.matmul(T_front_l, v_plane_posed_homo_front_l.unsqueeze(-1)).squeeze(-1)              
        
        return v_plane_back_l[:, :, :3], v_plane_front_l[:, :, :3],\
            v_plane_back_r[:, :, :3], v_plane_front_r[:, :, :3]
            

    @torch.no_grad()
    def setPose(self,
                **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
    
    def param_setting(self,
                    dtype=torch.float32,
                    device='cpu'):
        # init smpl pose
        betas = np.zeros([1, 11])
        # TODO: 当betas的缩小参数不是1时，不可以直接使用smpl_official出前向动作，需要使用自己构造的前向
        betas[:,0] = 1 # 0.78
        betas = torch.tensor(betas,dtype=dtype,device=device)
        
        transl = np.zeros([self.bs, 3])
        transl[:, 1] = np.array([0.9144] * self.bs)
        transl = torch.tensor(transl, dtype=dtype, device=device)
        
        body_pose_wo_foot = np.zeros([self.bs, (23- 4)* 3])
        body_pose_wo_foot = torch.tensor(body_pose_wo_foot, dtype=dtype, device=device)
        
        # 优化的过程发现global transl和脚部动作的变化冲突
        r_back_foot_pose = np.zeros([self.bs, 1* 3])
        r_back_foot_pose = torch.tensor(r_back_foot_pose, dtype=dtype, device=device)
        r_front_foot_pose = np.zeros([self.bs])
        r_front_foot_pose = torch.tensor(r_front_foot_pose, dtype=dtype, device=device)
    
        l_back_foot_pose = np.zeros([self.bs, 1* 3])
        l_back_foot_pose = torch.tensor(l_back_foot_pose, dtype=dtype, device=device)
        l_front_foot_pose = np.zeros([self.bs])
        l_front_foot_pose = torch.tensor(l_front_foot_pose, dtype=dtype, device=device)
        
        global_rot = np.zeros([self.bs, 3])
        global_rot = torch.tensor(global_rot, dtype=dtype, device=device)
        
        params_dict = {
                'betas': betas,
                'transl': transl,
                'body_pose_wo_foot': body_pose_wo_foot,
                'r_back_foot_pose': r_back_foot_pose,
                'r_front_foot_pose': r_front_foot_pose,
                'l_back_foot_pose': l_back_foot_pose,
                'l_front_foot_pose': l_front_foot_pose,
                'global_orient':global_rot
            }
        
        return params_dict
    
    # 计算当前body-pose下的脚面法线方向(脚是rest-pose)
    def construct_foot_n_T(self):
        device = self.v_shaped.device
        
        # assume only optim ankle and foot, so knee is fixed
        body_pose = torch.zeros([self.bs, 23* 3],
                                dtype=self.dtype).to(device)
        body_pose[:, :6* 3] = self.body_pose_wo_foot[:, :6* 3]
        body_pose[:, 8* 3: 9* 3] = self.body_pose_wo_foot[:, 6* 3: 7* 3]
        body_pose[:, 11* 3:] = self.body_pose_wo_foot[:,  7* 3:]        
        
        full_pose = torch.cat([self.global_orient, body_pose], dim=1)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(
            [self.bs, -1, 3, 3])

        J_transformed, A = batch_rigid_transform(rot_mats, self.J_shaped.expand(self.bs, -1, -1), self.parents, dtype=self.dtype)
        W = self.lbs_weights.unsqueeze(dim=0).expand([self.bs, -1, -1])
        num_joints = NUM_BODY_JOINTS+1

        T = torch.matmul(W, A.view(self.bs, num_joints, 16)) \
            .view(self.bs, -1, 4, 4) # (6890, 4, 4)

        T_back_l = T[:, self.foot_back_l_ids, :, :]
        T_back_r = T[:, self.foot_back_r_ids, :, :]
        T_front_l = T[:, self.foot_front_l_ids, :, :]
        T_front_r = T[:, self.foot_front_r_ids, :, :]
        
        T_n_back_l = torch.mean(T_back_l, dim=1)
        T_n_back_r = torch.mean(T_back_r, dim=1)
        T_n_front_l = torch.mean(T_front_l, dim=1)
        T_n_front_r = torch.mean(T_front_r, dim=1)
        
        return T_n_back_l, T_n_back_r, T_n_front_l, T_n_front_r