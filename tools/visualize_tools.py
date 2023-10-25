import hydra
import torch
import numpy as np
import trimesh
import os
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('D:/utils/pressure_toolkit') 
from lib.visualizer.open3d_visualizer import Visualizer
from lib.fitSMPL.SMPLModelRefine import smpl_model_refined
from lib.fitSMPL.SMPLModel import SMPLModel
from lib.Utils.refineSMPL_utils import compute_normal_batch,\
    body_pose_descompose, body_pose_compose
    
# @hydra.main(version_base=None, config_path="../configs", config_name="smpl_visualization")
# def main(cfg):
#     dtype = torch.float32
#     device = torch.device('cuda')
    
#     motion_type = cfg.task['motion_type']
    
#     # init visualizer
#     time_str = time.strftime("%m%d%H%M", time.localtime())
#     visualizer = Visualizer(cfg, time_str)
    

#     # init smpl result
#     init_smpl_data = torch.load(f'data/sh_result/opt_result_{motion_type}.pth')
#     transl_init = init_smpl_data['trans'] # [seq_len, bs, 3]
#     pose_init = init_smpl_data['pose'] # [seq_len, bs, 24, 3, 3]
#     beta_init = init_smpl_data['beta'] # [seq_len, bs, 10]
    
#     # init optim result
#     optim_smpl_data = torch.load(f'data/yhl_result/{motion_type}_optim_transl_pose.pth')
#     transl_optim = optim_smpl_data['trans'] # [seq_len, bs, 3]
#     pose_optim = optim_smpl_data['pose'] # [seq_len, bs, 24, 3, 3]
#     beta_optim = optim_smpl_data['beta'] # [seq_len, bs, 10]

#     # init smpl model
#     m_smpl = smpl_model_refined(
#             model_root='bodymodels/smpl',
#             num_betas=10,
#             bs=1,
#             gender='neutral')
#     m_smpl.to(device)
#     init_param = m_smpl.param_setting()
#     m_smpl.setPose(**init_param)
#     m_smpl.update_shape()
    
#     source_verts_list, optim_verts_list = [], []
#     for frame_idx in range(len(transl_optim)):
        
#         frame_pose_init = pose_init[frame_idx]
#         frame_trans_init = transl_init[frame_idx]

#         m_smpl_pose_dict = body_pose_descompose(full_pose=frame_pose_init,
#                         dtype=dtype, device=device)        
#         m_smpl_pose_dict['transl'] = frame_trans_init.to(device)
#         m_smpl.setPose(**m_smpl_pose_dict)
#         live_verts, _, live_plane = m_smpl.update_pose()
#         source_verts = live_verts
#         source_verts[:, :, 1] += 1.5
#         source_verts_list.append(live_verts)
        
#         # import pdb;pdb.set_trace()
#         frame_pose_optim = pose_optim[frame_idx]
#         frame_trans_optim = transl_optim[frame_idx]
        
#         m_smpl_pose_dict = body_pose_descompose(full_pose=frame_pose_optim,
#                         dtype=dtype, device=device)        
#         m_smpl_pose_dict['transl'] = frame_trans_optim.to(device)
#         m_smpl.setPose(**m_smpl_pose_dict)
#         live_verts, _, live_plane = m_smpl.update_pose()
#         optim_verts = live_verts
#         optim_verts_list.append(optim_verts)        

#     source_verts4view = torch.concat(source_verts_list, dim= 0)
#     optim_verts4view = torch.concat(optim_verts_list, dim= 0)
#     verts4view_all = [source_verts4view, optim_verts4view]
#     visualizer.render_two_sequence_3d(verts=verts4view_all,
#                                         faces=m_smpl.faces,
#                                         save_video=True,
#                                         visible=True,
#                                         need_norm=False, 
#                                         view='side') 


# single visualize
@hydra.main(version_base=None, config_path="../configs", config_name="smpl_visualize")
def main(cfg):
    dtype = torch.float32
    device = torch.device('cuda')
    
    motion_type = cfg.task['motion_type']
    sub_ids = cfg.task['sub_ids']
    seq_name = cfg.task['seq_name']
    # init visualizer
    time_str = time.strftime("%m%d%H%M", time.localtime())
    visualizer = Visualizer(cfg, time_str, fps=15)

    # init smpl result
    init_smpl_data = torch.load(f'debug/{sub_ids}/{seq_name}/tracking_result_{seq_name}.pth')
    transl_init = init_smpl_data['trans'] # [seq_len, bs, 3]
    pose_init = init_smpl_data['pose'] # [seq_len, bs, 24, 3, 3]
    beta_init = init_smpl_data['beta'] # [seq_len, bs, 10]
    model_scale_opt = init_smpl_data['model_scale_opt']
    # import pdb;pdb.set_trace()
    # import pdb; pdb.set_trace()
    
    # init smpl model
    # m_smpl = smpl_model_refined(
    #         model_root='../../bodyModels/smpl',
    #         num_betas=10,
    #         gender='male')
    m_smpl = SMPLModel(
            model_path='../../bodyModels/smpl',
            gender='male')
    m_smpl.to(device)
    
    # not use default betas, but betas optimized
    # init_param = m_smpl.param_setting()
    # m_smpl.setPose(**init_param)
    # m_smpl.update_shape()
    
    source_verts_list = []
    for frame_idx in range(0, len(transl_init)):
        
        # smpl_data = torch.load(f'debug/frame_debug/{sub_ids}/{seq_name}/{frame_idx}.pth')
        # import pdb;pdb.set_trace()
        frame_pose_init = pose_init[frame_idx]
        frame_trans_init = transl_init[frame_idx]
        frame_betas_init = beta_init
        frame_model_scale_opt = model_scale_opt
        
        # set wrist pose to zero. vposer cannot handle wrist rot
        frame_pose_init[:, 20, :, :] = torch.eye(3)
        frame_pose_init[:, 21, :, :] = torch.eye(3)


        global_rot_mat = frame_pose_init[:, 0, :, :]
        body_pose_mat = frame_pose_init[:, 1:, :, :]
        body_pose = np.zeros((1, 69))
        global_rot_aa = R.from_matrix(global_rot_mat.cpu().numpy()).as_rotvec()
        for i in range(23):
            body_pose[:, i*3:i*3+3] = R.from_matrix(body_pose_mat[:, i, :, :].cpu().numpy()).as_rotvec()
        
        
        # m_smpl_pose_dict = body_pose_descompose(full_pose=frame_pose_init,
        #                 dtype=dtype, device=device)     
        m_smpl_pose_dict = {}
        m_smpl_pose_dict['transl'] = frame_trans_init.to(device)
        m_smpl_pose_dict['betas'] = frame_betas_init.to(device)
        m_smpl_pose_dict['global_orient'] = torch.tensor(global_rot_aa, device=device, dtype=dtype)
        m_smpl_pose_dict['body_pose'] = torch.tensor(body_pose, device=device, dtype=dtype)
        m_smpl_pose_dict['model_scale_opt'] = torch.tensor(frame_model_scale_opt, device=device, dtype=dtype)
        
        m_smpl.setPose(**m_smpl_pose_dict)
        m_smpl.updateShape()
        m_smpl.initPlane()
        
        live_verts, _, _, _ = m_smpl.updatePose(m_smpl.body_pose)
        # trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=m_smpl.faces).export('debug/visual_smpl.obj')
        # import pdb;pdb.set_trace()
        source_verts = live_verts

        source_verts_list.append(source_verts)   

    source_verts4view = torch.concat(source_verts_list, dim= 0)

    visualizer.render_sequence_3d(verts=source_verts4view.detach().cpu().numpy(),
                                  faces=m_smpl.faces,
                                  save_video=True,
                                  visible=True,
                                  need_norm=False, 
                                  view='side')

    visualizer.render_sequence_3d(verts=source_verts4view.detach().cpu().numpy(),
                                  faces=m_smpl.faces,
                                  save_video=True,
                                  visible=True,
                                  need_norm=False, 
                                  view='front')

if __name__ == "__main__":
    main()