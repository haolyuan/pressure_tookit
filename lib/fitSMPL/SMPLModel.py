from typing import Optional, Dict, Union
import pickle
import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import trimesh
from trimesh import load_mesh
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
        
        baby_mesh = load_mesh('essentials/smplify_essential/smil_template_fingers_fix.ply', process=False)
        smil_v_template = torch.tensor(baby_mesh.vertices, dtype=self.dtype, requires_grad=False)
        smpl_v_template = torch.tensor(data_struct.v_template, dtype=self.dtype, requires_grad=False)
        self.register_buffer('smil_v_template', smil_v_template)
        self.register_buffer('smpl_v_template', smpl_v_template)

        j_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)
        self.register_buffer('J_regressor', j_regressor)

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)
        self.register_buffer('lbs_weights', lbs_weights)

        #SMPL Parameter
        model_scale_opt = torch.tensor([1], dtype=self.dtype)
        self.register_parameter('model_scale_opt',nn.Parameter(model_scale_opt, requires_grad=False))
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
        
        # add foot plane
        self.plane_L_init, self.plane_R_init = None, None
        self.foot_front_r_ids, self.foot_back_r_ids, self.foot_front_l_ids, \
            self.foot_back_l_ids = None, None, None, None
        self.foot_ids_smplL, self.foot_ids_smplR = None, None

        self.foot_ids_smplL = [3237, 3239, 3238, 3240, 3293, 3295, 3297, 3221, 3220, 3254, 3251, 3253,
                                3250, 3296, 3300, 3261, 3263, 3264, 3262, 3306, 3305, 3228, 3229, 3276,
                                3278, 3275, 3277, 3307, 3310, 3315, 3287, 3289, 3288, 3290, 3224, 3225,
                                3352, 3353, 3406, 3437, 3355, 3354, 3358, 3359, 3438, 3439, 3361, 3360,
                                3362, 3357, 3363, 3356, 3440, 3419, 3407, 3444, 3443, 3408, 3448, 3447,
                                3430, 3449, 3450, 3442, 3441, 3420, 3446, 3445, 3421, 3451, 3452, 3422,
                                3431, 3456, 3455, 3429, 3462, 3464, 3428, 3461, 3463, 3467, 3460, 3427,
                                3454, 3453, 3423, 3465, 3457, 3424, 3466, 3459, 3425, 3468, 3458, 3426]
        self.foot_ids_smplR = [6637, 6640, 6636, 6639, 6693, 6695, 6697, 6621, 6622, 6654, 6651, 6653, 
                                6652, 6696, 6701, 6661, 6663, 6664, 6660, 6706, 6705, 6630, 6629, 6674, 
                                6677, 6675, 6678, 6707, 6711, 6716, 6687, 6690, 6686, 6689, 6626, 6625, 
                                6752, 6753, 6806, 6837, 6755, 6754, 6758, 6759, 6838, 6839, 6761, 6760, 
                                6762, 6757, 6763, 6756, 6840, 6819, 6807, 6844, 6843, 6808, 6848, 6847, 
                                6830, 6849, 6850, 6842, 6841, 6820, 6846, 6845, 6821, 6851, 6852, 6822, 
                                6831, 6856, 6855, 6829, 6862, 6863, 6828, 6861, 6864, 6867, 6860, 6827, 
                                6854, 6853, 6823, 6865, 6857, 6824, 6866, 6859, 6826, 6868, 6858, 6825]

        back_L_faces = np.loadtxt('essentials/foot_related/back_L_faces.txt', dtype=int).tolist()
        self.back_L_faces = torch.tensor(back_L_faces).unsqueeze(0)
        front_L_faces = np.loadtxt('essentials/foot_related/front_L_faces.txt', dtype=int).tolist()
        self.front_L_faces = torch.tensor(front_L_faces).unsqueeze(0)
        front_R_faces = np.loadtxt('essentials/foot_related/front_R_faces.txt', dtype=int).tolist()
        self.front_R_faces = torch.tensor(front_R_faces).unsqueeze(0)        
        back_R_faces = np.loadtxt('essentials/foot_related/back_R_faces.txt', dtype=int).tolist()
        self.back_R_faces = torch.tensor(back_R_faces).unsqueeze(0)        
        
                            
    def updateShape(self):
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
        trimesh.Trimesh(vertices=verts[0].detach().cpu().numpy(), faces=self.faces).export('debug/smpl.obj')
        trimesh.Trimesh(vertices=joints[0].detach().cpu().numpy()).export('debug/smpl_jts.obj')
        # apply extra joints in SPIN
        extra_joints = self.vertices2joints(self.J_regressor_extra, verts)
        joints_54 = torch.cat([joints, extra_joints], dim=1)
        joints_49 = joints_54[:, self.joint_map, :]
        
        # plane transform
        v_plane = self.updatePlane(T=T)
        
        # apply trans
        verts += self.transl.unsqueeze(dim=1)
        joints_49 += self.transl.unsqueeze(dim=1)
        v_plane += self.transl.unsqueeze(dim=1)
        joints_54 += self.transl.unsqueeze(dim=1)
        return verts, joints_49, joints_54, v_plane

    @torch.no_grad()
    def setPose(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])

    def initPlane(self):
        assert self.v_shaped != None, "must init smpl shape first"
        
        # seperate foots into two part
        self.foot_front_r_ids = [i for i in range(36)] + [36, 41, 42, 47, 48, 49]
        self.foot_back_r_ids = [60, 61, 62, 69, 70, 71] + [i for i in range(72, 96)]
        self.foot_front_l_ids = self.foot_front_r_ids
        self.foot_back_l_ids = self.foot_back_r_ids
        
        # construct y offset in smpl foots
        v_smpl_foot_L, v_smpl_foot_R =\
            self.v_shaped[0][self.foot_ids_smplL, :] ,self.v_shaped[0][self.foot_ids_smplR, :]
            # .detach().cpu().numpy()
            # .detach().cpu().numpy()        
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
        
        
        
        # trimesh.Trimesh(vertices=self.v_shaped[0].detach().cpu().numpy(), faces=self.faces).\
        #     export('debug/plane_visual/smpl_mesh.obj')
        # trimesh.Trimesh(vertices=self.plane_R_init[self.foot_front_r_ids, :]).\
        #     export('debug/plane_visual/plane_front_R_init.obj')
        # trimesh.Trimesh(vertices=self.plane_L_init[self.foot_front_l_ids, :]).\
        #     export('debug/plane_visual/plane_front_L_init.obj')
        # trimesh.Trimesh(self.plane_R_init[self.foot_back_r_ids, :]).\
        #     export('debug/plane_visual/plane_back_R_init.obj')
        # trimesh.Trimesh(self.plane_L_init[self.foot_back_l_ids, :]).\
        #     export('debug/plane_visual/plane_back_L_init.obj')
            
    def updatePlane(self, T):
        T_back_l = T[:, self.foot_ids_back_smplL, :, :]
        T_back_r = T[:, self.foot_ids_back_smplR, :, :]
        T_front_l = T[:, self.foot_ids_front_smplL, :, :]
        T_front_r = T[:, self.foot_ids_front_smplR, :, :] 

        # back left foot plane
        homogen_coord_back_l = torch.ones([1, self.plane_back_L_init.shape[0], 1], dtype=self.dtype,
                                   device=self.plane_back_L_init.device)
        
        v_plane_homo_back_l = torch.cat([self.plane_back_L_init.expand(1, -1, -1),
                                               homogen_coord_back_l], dim=2)
        v_plane_back_l = torch.matmul(T_back_l, v_plane_homo_back_l.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        # back right foot plane
        homogen_coord_back_r = torch.ones([1, self.plane_back_R_init.shape[0], 1], dtype=self.dtype,
                                   device=self.plane_back_R_init.device)
        v_plane_homo_back_r = torch.cat([self.plane_back_R_init.expand(1, -1, -1),
                                               homogen_coord_back_r], dim=2)
        v_plane_back_r = torch.matmul(T_back_r, v_plane_homo_back_r.unsqueeze(-1)).squeeze(-1)[:, :, :3]

        # front right foot plane
        homogen_coord_front_r = torch.ones([1, self.plane_front_R_init.shape[0], 1], dtype=self.dtype,
                                   device=self.plane_front_R_init.device)
        v_plane_homo_front_r = torch.cat([self.plane_front_R_init.expand(1, -1, -1),
                                               homogen_coord_front_r], dim=2)
        v_plane_front_r = torch.matmul(T_front_r, v_plane_homo_front_r.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        # front left foot plane
        homogen_coord_front_l = torch.ones([1, self.plane_front_L_init.shape[0], 1], dtype=self.dtype,
                                   device=self.plane_front_L_init.device)
        v_plane_homo_front_l = torch.cat([self.plane_front_L_init.expand(1, -1, -1),
                                               homogen_coord_front_l], dim=2)
        v_plane_front_l = torch.matmul(T_front_l, v_plane_homo_front_l.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        
        v_plane = torch.concat([v_plane_back_l, v_plane_front_l, v_plane_back_r, v_plane_front_r], dim = 1)
        
        return v_plane

        

