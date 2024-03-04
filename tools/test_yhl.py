# import hydra
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
import subprocess
import sys
sys.path.append('/home/yuanhaolei/Document/code/pressure_toolkit')

from lib.fitSMPL.SMPLModel import SMPLModel


# from lib.fitSMPL.SMPLModel import SMPLModel
# from lib.fitSMPL.SMPLModelRefine import smpl_model_refined
# from lib.fitSMPL.contactTerm import ContactTerm

# from lib.Utils.refineSMPL_utils import compute_normal_batch,\
#     body_pose_descompose, body_pose_compose, smpl_forward_official
# from lib.Utils.measurements import MeasurementsLoss
    
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

def test_kp():
    sub_ids_list = ['S03', 'S04']
    for sub_ids in sub_ids_list:
        sub_ids_dir = f'/data/PressureDataset/20230611/{sub_ids}'
        for seq_name in os.listdir(sub_ids_dir):
            seq_kp_dir = f'/data/PressureDataset/20230611/{sub_ids}/{seq_name}/insole'
            if not os.path.exists(seq_kp_dir):
                continue
            for insole_npy in os.listdir(seq_kp_dir):
                kp_name = insole_npy.split('.')[0]
                kp_path = f'{sub_ids_dir}/{seq_name}/keypoints/{kp_name}.npy'
                kp_data = np.load(kp_path, allow_pickle=True)
                if kp_data.shape!=():
                    try:
                        kp_dict = dict(kp_data.item())
                    except:
                        print(kp_data)
                        import pdb;pdb.set_trace()
                        
                    print(kp_path)
                    # temp_data = kp_data[0]
                    # os.makedirs(f'/home/yuanhaolei/temp_linux/{sub_ids}/{seq_name}/keypoints', exist_ok=True)
                    # np.save(f'/home/yuanhaolei/temp_linux/{sub_ids}/{seq_name}/keypoints/{kp_name}', temp_data)
                    
                    
                    # command = ['sudo', 'cp', f'/home/yuanhaolei/temp_linux/{sub_ids}/{seq_name}/keypoints/{kp_name}.npy',
                    #            f'/data/PressureDataset/20230611/{sub_ids}/{seq_name}/keypoints/{kp_name}.npy']
                    # proc = subprocess.Popen(command)
                    # proc.wait()

    # os.makedirs('/home/yuanhaolei/temp_linux/S03/S3-B-PAOBU-3/keypoints', exist_ok=True)
    # np.save('/home/yuanhaolei/temp_linux/S03/S3-B-PAOBU-3/keypoints/418', save_data)

def load_openpose_kp():
    import json
    kp_path = "/home/yuanhaolei/Document/code/prox-master/prox_dataset/quantitative/keypoints/vicon_03301_01/s001_frame_00001__00.00.00.023_keypoints.json"
    with open(kp_path, 'rb')as f:
        data = json.load(f)
    import pdb;pdb.set_trace()


def visSMPLFootModel(self,contact_label):
    ## Ground truths
    if contact_label.shape[0]!=6890:
        contact = np.zeros(6890)
        contact[self.footIdsL] = contact_label[0]
        contact[self.footIdsR] = contact_label[1]
    else:
        contact = contact_label
        hit_id = (contact == 1).nonzero()[0]

    _mesh = trimesh.Trimesh(vertices=self.v_template, faces=self.faces, process=False)
    _mesh.visual.vertex_colors = (191, 191, 191, 255)
    _mesh.visual.vertex_colors[hit_id, :] = (0, 255, 0, 255)

    return _mesh

def A2_refine():
    smpl_path = '/data/yuanhaolei/PressureDataset_label/smpl_pose/S01/A2/127_0100.npz'
    smpl_data = np.load(smpl_path, allow_pickle=True)['arr_0'].item()
    m_smpl = SMPLModel(
        model_path='/home/yuanhaolei/Document/code/pressure_toolkit/essentials/bodyModels/smpl',
        num_betas=10,
        gender='female')
    
    init_param_path = '/data/yuanhaolei/PressureDataset_label/20230713/S01/init_param_S01.npy'
    shape_data = dict(np.load(init_param_path, allow_pickle=True).item())

    device = torch.device('cuda')
    dtype=torch.float32
    betas = torch.tensor(shape_data['betas'],dtype=dtype,device=device)
    transl = torch.tensor(smpl_data['transl'],dtype=dtype,device=device)
    body_pose = torch.tensor(smpl_data['body_pose'], dtype=dtype, device=device)
    global_rot = torch.tensor(smpl_data['global_orient'], dtype=dtype, device=device)
    model_scale_opt = torch.tensor(smpl_data['model_scale_opt'], dtype=dtype, device=device)
    betas[0][1:] = 0
    body_pose[0][3*3+2] = 0
    body_pose[0][4*3+2] = 0.03
    body_pose[0][1*3+2] = 0.02
    body_pose[0][1*3] += 0.03
    
    body_pose[0][0] = 0.02
    body_pose[0][3*3] = 0.05
    
    transl[0][1] += 0.032
    
    # body_pose_zero = torch.zeros_like(body_pose)
    # global_rot_zero = torch.zeros_like(global_rot)
    params_dict = {
        'betas':betas,
        'transl':transl,
        'body_pose':body_pose,
        'global_orient':global_rot,
        'model_scale_opt':model_scale_opt
    }
    m_smpl.setPose(**params_dict)
    m_smpl.updateShape()
    
    # update plane with init shape
    m_smpl.initPlane()
    live_verts, live_joints, _, live_plane = m_smpl.updatePose(body_pose=m_smpl.body_pose)
    _verts = live_verts.detach().cpu().numpy()[0]
    output_mesh = trimesh.Trimesh(vertices=_verts,faces=m_smpl.faces,process=False)
    
    ref_mesh= trimesh.load('/home/yuanhaolei/Document/code/pressure_toolkit/127.obj')
    # # load contact label
    # footIdsL, footIdsR =\
    #     np.loadtxt('/home/yuanhaolei/Document/code/pressure_toolkit/essentials/footL_ids.txt'),\
    #     np.loadtxt('/home/yuanhaolei/Document/code/pressure_toolkit/essentials/footR_ids.txt')
    # contact_label = dict(np.load('/data/PressureDataset/20230713/S01/A2/insole/135.npy', allow_pickle=True).item())['insole']
    
    # if contact_label.shape[0]!=6890:
    #     contact = np.zeros(6890)
    #     contact[footIdsL] = contact_label[0]
    #     contact[footIdsR] = contact_label[1]
    # else:
    #     contact = contact_label
    #     hit_id = (contact == 1).nonzero()[0]
        
    output_mesh.visual.vertex_colors = (191, 191, 191, 255)
    # output_mesh.visual.vertex_colors[hit_id, :] = (0, 255, 0, 255)
    
    # output_mesh.visual.vertex_colors = (191, 191, 191, 255)

    for v_id in range(output_mesh.visual.vertex_colors.shape[0]):
        if ref_mesh.visual.vertex_colors[v_id, :][1] == 255:
            output_mesh.visual.vertex_colors[v_id, :] = (0, 255, 0, 255)
    
    output_mesh.export(f'/home/yuanhaolei/temp_linux/test_A2.obj')
    
    import pdb;pdb.set_trace()

def main():
    A2_refine()
    # load_openpose_kp()
    
    # cali_path = 'D:/dataset/PressureDataset/20230422/S01/MoCap_20230422_092117/calibration.npy'
    # cali_data = dict(np.load(cali_path, allow_pickle=True).item())
    
    # get_pic()
    # exit()
    
    # test_kp()
    
    # get_smpl_height()
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