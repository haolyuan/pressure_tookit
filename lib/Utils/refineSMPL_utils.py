import numpy as np
import torch
from smplx import SMPL
import trimesh
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('D:/utils/BetterFootContact_new') 

def smpl_forward_official(device, 
                          dtype,
                          body_pose = np.zeros([1, 69]),
                          betas = np.zeros([1, 10]),
                          transl = np.zeros([1, 3]),
                          global_rot = np.zeros([1, 3]),
                          result_dir=None,
                          result_name=None,
                          save_data=True):
    smpl_model = SMPL('bodymodels/smpl/SMPL_NEUTRAL.pkl').to(device)

    transl[:, 1] = np.array([0.9144] * 1)
    
    # rest pose
    betas = torch.tensor(betas,dtype=dtype,device=device)        
    transl = torch.tensor(transl,dtype=dtype,device=device)
    body_pose = torch.tensor(body_pose, dtype=dtype, device=device)
    global_rot = torch.tensor(global_rot, dtype=dtype, device=device)
    output = smpl_model.forward(betas=betas,
                                body_pose=body_pose,
                                global_orient=global_rot,
                                transl=transl)
    
    return output.vertices, output.joints

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


def compute_normal_batch(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    vert_norm = torch.zeros(bs * nv, 3).type_as(vertices)
    tris = face_vertices(vertices, faces)
    face_norm = F.normalize(
        torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]), dim=-1
    )

    faces = (faces + (torch.arange(bs).type_as(faces) * nv)[:, None, None]).view(-1, 3)

    vert_norm[faces[:, 0]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 1]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 2]] += face_norm.view(-1, 3)

    vert_norm = F.normalize(vert_norm, dim=-1).view(bs, nv, 3)

    return vert_norm


def body_pose_descompose(full_pose, device, dtype):
    """descompose smpl pose into pose format using for smpl_model_refine

    Args:
        full_pose (_type_): torch.tensor([bs, 24, 3, 3])
        device (_type_): _description_
        dtype (_type_): _description_

    Returns:
        dict:  {
            'body_pose_wo_foot': torch.tensor([bs, 19, 3]),
            'r_back_foot_pose': torch.tensor([bs, 3]),
            'r_front_foot_pose': torch.tensor([bs, 1]),
            'l_back_foot_pose': torch.tensor([bs, 3]),
            'l_front_foot_pose': torch.tensor([bs, 1]),
            'global_orient': torch.tensor([bs, 3])
            }
    """    
    full_pose_aa_cpu = np.zeros([full_pose.shape[0], full_pose.shape[1], 3, 1])
    for bs_idx in range(full_pose.shape[0]):
        for joints_idx in range(full_pose.shape[1]):
            joints_pose_cpu = full_pose[bs_idx][joints_idx].cpu().numpy()
            full_pose_aa_cpu[bs_idx, joints_idx] = cv2.Rodrigues(joints_pose_cpu)[0]
            
    full_pose_aa = torch.tensor(full_pose_aa_cpu, dtype=dtype, device=device).squeeze(-1).view((1, -1))
    body_pose_aa = full_pose_aa[:, 3:]

    # [23, 3, 3] → 身体 + 左/右ankle + 左/右foot
    body_pose_wo_foot = torch.zeros([body_pose_aa.shape[0], 19* 3], dtype=dtype, device=device)
    
    body_pose_wo_foot[:, :6* 3] = body_pose_aa[:, :6* 3]
    body_pose_wo_foot[:, 6* 3: 7* 3] = body_pose_aa[:, 8* 3: 9* 3]
    body_pose_wo_foot[:,  7* 3:] = body_pose_aa[:, 11* 3:]
    
    output_param_dict = dict()
    output_param_dict['body_pose_wo_foot'] = body_pose_wo_foot
    output_param_dict['r_back_foot_pose'] = body_pose_aa[:, 7* 3: 8* 3]
    output_param_dict['l_back_foot_pose'] = body_pose_aa[:, 6* 3: 7* 3]
    output_param_dict['l_front_foot_pose'] = body_pose_aa[:, 9* 3]
    output_param_dict['r_front_foot_pose'] = body_pose_aa[:, 10* 3]
    output_param_dict['global_orient'] = full_pose_aa[:, :3]
    
    return output_param_dict

def body_pose_compose(pose_dict,
                      device=torch.device('cuda'),
                      dtype=torch.float32):
    """compose smpl_model_refine pose format to smpl fullbody pose

    Args:
        pose_dict (dict): {
            'body_pose_wo_foot': torch.tensor([bs, 19, 3]),
            'r_back_foot_pose': torch.tensor([bs, 3]),
            'r_front_foot_pose': torch.tensor([bs, 1]),
            'l_back_foot_pose': torch.tensor([bs, 3]),
            'l_front_foot_pose': torch.tensor([bs, 1]),
            'global_orient': torch.tensor([bs, 3])
            }
        device (_type_): _description_
        dtype (_type_): _description_
    Returns:
        _type_: [bs, 24, 3, 3]
    """    
    body_pose_wo_foot = pose_dict['body_pose_wo_foot']
    l_back_foot_pose = pose_dict['l_back_foot_pose']
    r_back_foot_pose = pose_dict['r_back_foot_pose']
    l_front_foot_pose = pose_dict['l_front_foot_pose']
    r_front_foot_pose = pose_dict['r_front_foot_pose']
    global_orient = pose_dict['global_orient']
    
    full_pose = torch.zeros([body_pose_wo_foot.shape[0], 24* 3],
                            dtype=dtype).to(device)

    full_pose[:, :1* 3] = global_orient
    full_pose[:, 1* 3:7* 3] = body_pose_wo_foot[:, :6* 3]
    full_pose[:, 7* 3: 8* 3] = l_back_foot_pose
    full_pose[:, 8* 3: 9* 3] = r_back_foot_pose
    full_pose[:, 9* 3: 10* 3] = body_pose_wo_foot[:, 6* 3: 7* 3]

    full_pose[:, 10* 3] = l_front_foot_pose # : 10* 3
    full_pose[:, 11* 3] = r_front_foot_pose # : 11* 3
    full_pose[:, 12* 3:] = body_pose_wo_foot[:,  7* 3:]
    
    # aa to matrix
    output_pose = np.zeros((full_pose.shape[0], 24, 3, 3))
    for i in range(output_pose.shape[0]):
        for j in range(output_pose.shape[1]):
            output_pose[i, j, :3, :3] = R.from_rotvec(full_pose[i, j * 3: j * 3 + 3].cpu().numpy()).as_matrix()
    
    return torch.tensor(output_pose, device=device, dtype=dtype)