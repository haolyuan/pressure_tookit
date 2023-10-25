import hydra
import torch
import numpy as np
import trimesh
import os
import time
from tqdm import tqdm
import pickle as pkl
from smplx import SMPL, SMPLX
import yaml
import cv2, imageio
import sys
sys.path.append('D:/utils/pressure_toolkit')

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.SMPLModelRefine import smpl_model_refined
from lib.fitSMPL.contactTerm import ContactTerm

from lib.Utils.refineSMPL_utils import compute_normal_batch,\
    body_pose_descompose, body_pose_compose, smpl_forward_official
from lib.Utils.measurements import MeasurementsLoss
    
# @hydra.main(version_base=None, config_path="../configs", config_name="smpl_visualize")
# def main(cfg):
#     dtype = torch.float32
#     device = torch.device('cuda')
    
#     motion_type = cfg.task['motion_type']
#     seq_name = cfg.task['seq_name']

#     # init smpl result
#     init_smpl_data = torch.load(f'debug/{seq_name}/tracking_result_{seq_name}.pth')
#     beta_init = init_smpl_data['beta'] # [seq_len, bs, 10]
    
#     # init smpl model
#     m_smpl_refined = smpl_model_refined(
#             model_root='../../bodyModels/smpl',
#             num_betas=10,
#             bs=1,
#             gender='male')
#     m_smpl = SMPLModel(
#         model_path='../../bodyModels/smpl',
#         num_betas=10,
#         gender='male')
#     m_smpl_refined
#     frame_idx = 10
#     frame_betas_init = beta_init[frame_idx]
    
#     m_smpl_refined_pose_dict = {}        

#     m_smpl_refined_pose_dict['betas'] = frame_betas_init.to(device)
#     m_smpl_refined.setPose(**m_smpl_refined_pose_dict)
#     m_smpl_refined.update_shape()

#     live_verts_refined, _, _ = m_smpl_refined.update_pose()

#     params_path = 'debug/init_param100.npy'
#     init_params = np.load(params_path,allow_pickle=True).item()

#     betas = torch.tensor(init_params['betas'],dtype=dtype,device=device)

#     params_dict = {
#         'betas':betas,

#     }
#     m_smpl.setPose(**params_dict)

#     m_smpl.updateShape()
#     live_verts, _ = m_smpl.updatePose()

#     trimesh.Trimesh(vertices=live_verts_refined[0].detach().cpu().numpy(), faces = m_smpl.faces).\
#         export(f'debug/live_verts_refined.obj')    
#     trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces = m_smpl.faces).\
#         export(f'debug/live_verts.obj') 
#     import pdb;pdb.set_trace()
        
#     official_verts, _ = smpl_forward_official(gender='MALE',
#                                               betas=init_params['betas'][:, 1:])    
#     trimesh.Trimesh(vertices=official_verts[0].detach().cpu().numpy(), faces = m_smpl.faces).\
#         export(f'debug/official_verts.obj')   
          

def get_pic():
    root_dir = 'D:/temp/debug'
    sub_ids_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12']

    for sub_ids in sub_ids_list:
        seq_name_list = os.listdir(os.path.join(root_dir, sub_ids)) 
        for seq_name in seq_name_list:
            img_list = os.listdir(os.path.join(root_dir, sub_ids, f'{seq_name}', 'gt_visual'))
            for img_name in img_list:
                
                img = cv2.imread(os.path.join(root_dir, sub_ids, f'{seq_name}', 'gt_visual', img_name))
                cv2.imshow('img', img)
                cv2.waitKey(-1)
    
def get_smpl_height():
    root = 'D:\dataset\PressureDataset/20230422\S01\MoCap_20230422_094530\insole'
    for idx in range(0, 13):
        insole_path = os.path.join(root, f'{idx:03d}.npy')
        print(insole_path)
        insole_data = np.load(insole_path, allow_pickle=True).item()
        print(insole_data['contact_label'])

    import pdb;pdb.set_trace()

    smpl_mesh = trimesh.load_mesh('essentials/smpl_uv/smpl_template.obj')
    smpl_v = smpl_mesh.vertices
    y_min = min(smpl_v[:, 1])
    foot_ids = []
    for idx in range(smpl_v.shape[0]):
        if abs(smpl_v[idx][1] - y_min) <=0.05:
            foot_ids.append(idx)
    foot_1_ids = foot_ids[:len(foot_ids)//2] 
    foot_2_ids = foot_ids[len(foot_ids)//2:] 
    np.save('essentials/foot_related/foot_ids_surfaces.npy', np.array(foot_ids))
    import pdb;pdb.set_trace()
    
    
    depth_img = cv2.imread('D:/dataset/mkv_data/20230422/S01/MoCap_20230422_092117/transform_depth2color/transform_depth2color_1.png')
    color_img = cv2.imread('D:\dataset\mkv_data/20230422\S01\MoCap_20230422_092117\color/001.png')
    
    depth_img = np.sum(depth_img, axis=2)
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i][j]<10:
                color_img[i][j][0]=0
                color_img[i][j][1]=1
                color_img[i][j][2]=2
    # output_img = depth_img + color_img
    cv2.imshow('temp', color_img)
    cv2.waitKey(-1)
    cv2.imwrite('D:\dataset\mkv_data/20230422\S01\MoCap_20230422_092117/w_dist.png', color_img)
    exit()
    import pdb;pdb.set_trace()
    
    
    cam_data = np.load('D:/dataset/PressureDataset/20230422/S01/MoCap_20230422_092117/calibration.npy',
                       allow_pickle=True)
    import pdb;pdb.set_trace()
    
    smpl_model = SMPL(model_path='../../bodyModels/smpl/SMPL_MALE.pkl')
    t_smpl = smpl_model.forward()
    smplx_model = SMPLX(model_path='D:/BaiduNetdiskDownload/For_Study/SMPL相关/model_files/smplx/SMPLX_MALE.npz')
    t_smplx = smplx_model.forward()
    
    with open('D:/utils/pressure_toolkit/essentials/smplify_essential/smplx_measurements.yaml', 'r') as f:
        meas_data = yaml.safe_load(f)
    head_top = meas_data['HeadTop']
    left_heel = meas_data['HeelLeft']

    left_heel_bc = left_heel['bc']
    left_heel_face_idx = left_heel['face_idx']
    head_top_bc = head_top['bc']
    head_top_face_idx = head_top['face_idx']

    v_shaped_triangles = torch.index_select(
            t_smplx.vertices, 1, torch.tensor(smplx_model.faces.astype(np.int32)).view(-1)).reshape(1, -1, 3, 3)
    head_top_tri = v_shaped_triangles[:, head_top_face_idx]
    head_top = (head_top_tri[:, 0, :] * head_top_bc[0] +
                head_top_tri[:, 1, :] * head_top_bc[1] +
                head_top_tri[:, 2, :] * head_top_bc[2])
    left_heel_tri = v_shaped_triangles[:, left_heel_face_idx]
    left_heel = (left_heel_tri[:, 0, :] * left_heel_bc[0] +
                    left_heel_tri[:, 1, :] * left_heel_bc[1] +
                    left_heel_tri[:, 2, :] * left_heel_bc[2])
    
    v_array = torch.concat([head_top, left_heel, head_top_tri[0], left_heel_tri[0]], dim=0)
    trimesh.Trimesh(vertices=v_array.detach().cpu().numpy()).export('debug/smplx_top_floor.obj')

    
    trimesh.Trimesh(vertices=t_smpl.vertices[0].detach().cpu().numpy(),
                    faces=smpl_model.faces).export('debug/smpl_template.obj')
    
    trimesh.Trimesh(vertices=t_smplx.vertices[0].detach().cpu().numpy(),
                    faces=smplx_model.faces).export('debug/smplx_template.obj')
          
def main():
    # cali_path = 'D:/dataset/PressureDataset/20230422/S01/MoCap_20230422_092117/calibration.npy'
    # cali_data = dict(np.load(cali_path, allow_pickle=True).item())
    
    # get_pic()
    # exit()
        
    get_smpl_height()
    exit()
    import pdb;pdb.set_trace()
    

    contact_path = 'D:\dataset\PressureDataset/20230422\S01\MoCap_20230422_094530\insole/013.npy'
    contact_data = dict(np.load(contact_path, allow_pickle=True).item())
    
    contact_term = ContactTerm(device=torch.device('cuda'))
    
    full_foot_contacts = [[1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1]]
    contact_ids_smpl, contact_ids_smplL, contact_ids_smplR =\
        contact_term.contact2smpl(np.array(full_foot_contacts))
    import pdb;pdb.set_trace()
    J_regressor_extra = np.load('D:/BaiduNetdiskDownload/For_Study/SMPL相关/model_files/J_regressor_extra.npy')
    
    with open('D:/bodyModels\smpl/SMPL_MALE.pkl', 'rb') as f:
        smpl_file = pkl.load(f, encoding='latin1')
    
    smpl_file_1 = np.load('D:\BaiduNetdiskDownload\For_Study\SMPL相关\data/smpl_mean_params.npz')
    data_a = dict(smpl_file_1)
    
    
    vertices, joints = smpl_forward_official()
    trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy()).export('D:/utils/pressure_toolkit/debug/smpl_template.obj')
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    main()