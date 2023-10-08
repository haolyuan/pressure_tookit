import hydra
import torch
import numpy as np
import trimesh
import os
import time
from tqdm import tqdm
import sys
sys.path.append('D:/utils/pressure_toolkit')

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.SMPLModelRefine import smpl_model_refined

from lib.Utils.refineSMPL_utils import compute_normal_batch,\
    body_pose_descompose, body_pose_compose, smpl_forward_official
    
    
@hydra.main(version_base=None, config_path="../configs", config_name="smpl_visualize")
def main(cfg):
    dtype = torch.float32
    device = torch.device('cuda')
    
    motion_type = cfg.task['motion_type']
    seq_name = cfg.task['seq_name']

    # init smpl result
    init_smpl_data = torch.load(f'debug/{seq_name}/tracking_result_{seq_name}.pth')
    beta_init = init_smpl_data['beta'] # [seq_len, bs, 10]
    
    # init smpl model
    m_smpl_refined = smpl_model_refined(
            model_root='../../bodyModels/smpl',
            num_betas=10,
            bs=1,
            gender='male')
    m_smpl = SMPLModel(
        model_path='../../bodyModels/smpl',
        num_betas=10,
        gender='male')
    m_smpl_refined
    frame_idx = 10
    frame_betas_init = beta_init[frame_idx]
    
    m_smpl_refined_pose_dict = {}        

    m_smpl_refined_pose_dict['betas'] = frame_betas_init.to(device)
    m_smpl_refined.setPose(**m_smpl_refined_pose_dict)
    m_smpl_refined.update_shape()

    live_verts_refined, _, _ = m_smpl_refined.update_pose()

    params_path = 'debug/init_param100.npy'
    init_params = np.load(params_path,allow_pickle=True).item()

    betas = torch.tensor(init_params['betas'],dtype=dtype,device=device)

    params_dict = {
        'betas':betas,

    }
    m_smpl.setPose(**params_dict)

    m_smpl.updateShape()
    live_verts, _ = m_smpl.updatePose()

    trimesh.Trimesh(vertices=live_verts_refined[0].detach().cpu().numpy(), faces = m_smpl.faces).\
        export(f'debug/live_verts_refined.obj')    
    trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces = m_smpl.faces).\
        export(f'debug/live_verts.obj') 
    import pdb;pdb.set_trace()
        
    official_verts, _ = smpl_forward_official(gender='MALE',
                                              betas=init_params['betas'][:, 1:])    
    trimesh.Trimesh(vertices=official_verts[0].detach().cpu().numpy(), faces = m_smpl.faces).\
        export(f'debug/official_verts.obj')   
          
if __name__ == "__main__":
    main()