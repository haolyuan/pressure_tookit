import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import os.path as osp
import numpy as np
import smplx
import torch
import trimesh
import cv2
from icecream import ic

import sys
sys.path.append('D:/utils/pressure_toolkit')

from lib.config.config import parse_config
from lib.dataset.PressureDataset import PressureDataset
from lib.fitSMPL.Camera import RGBDCamera
from lib.fitSMPL.SMPLSolver import SMPLSolver
from lib.Utils.fileio import saveNormalsAsOBJ

def main(**args):
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')
             

    basdir=args.get('basdir')
    dataset_name=args.get('dataset')
    seq_name=args.get('seq_name')
    sub_ids=args.get('sub_ids')
    label_output_dir=args.get('label_output_dir')

    os.makedirs(os.path.join(label_output_dir, 'mesh', f'{sub_ids}', f'{seq_name}'), exist_ok=True)
    os.makedirs(os.path.join(label_output_dir, 'smpl_pose', f'{sub_ids}', f'{seq_name}'), exist_ok=True)

    m_data = PressureDataset(
        basdir=args.get('basdir'),
        dataset_name=args.get('dataset'),
        sub_ids=args.get('sub_ids'),
        seq_name=args.get('seq_name'),
        label_output_dir=label_output_dir
    )
    
    m_cam = RGBDCamera(
        basdir=basdir,
        dataset_name=dataset_name,
        sub_ids=sub_ids,
        seq_name=seq_name,
    )

    m_solver = SMPLSolver(
        model_path=args.get('model_folder'),
        num_betas=args.get('num_shape_comps'),
        gender=args.get('model_gender'),
        color_size=args.get('color_size'), depth_size=args.get('depth_size'),
        cIntr=m_cam.cIntr_cpu, dIntr=m_cam.dIntr_cpu,
        depth2floor=m_data.depth2floor,
        depth2color=m_cam.d2c_cpu,
        w_verts3d=args.get('depth_weights'),
        w_betas=args.get('shape_weights'),
        w_joint2d=args.get('keypoint_weights'),
        w_penetrate=args.get('penetrate_weights'),
        w_contact=args.get('contact_weights'),
        label_output_dir=label_output_dir,
        sub_ids=sub_ids,
        seq_name=seq_name,
        device=device
    )
    
    # load init shape
    init_A_data_path = f'{label_output_dir}/{dataset_name}/{sub_ids}/init_param_{sub_ids}.npy'
    init_A_data = np.load(init_A_data_path, allow_pickle=True).item()
    # import pdb;pdb.set_trace()
    if args.get('init_model'):
        select_idx = int(args.get('init_idx_start'))
        frame_data = m_data.getFrameData(ids=select_idx,
                                         init_shape=False,
                                         tracking=False)
        dv_valid,dn_valid = m_cam.preprocessDepth(frame_data['depth_map'],frame_data['mask'])
        dv_floor,dn_normal = m_data.mapDepth2Floor(dv_valid,dn_valid)
        annot = m_solver.initPose(init_shape=[init_A_data['betas'], init_A_data['model_scale_opt']],
                                depth_vmap=dv_floor,
                                depth_nmap=dn_normal,
                                color_img=frame_data['img'],
                                keypoints=frame_data['kp'],
                                contact_data=frame_data['contact'],
                                cliff_pose=frame_data['cliff_pose'],
                                frame_idx=select_idx,
                                max_iter=args.get('maxiters'))
        
    else:

        frame_start = int(args.get('init_idx_start'))
        frame_end = int(args.get('init_idx_end'))
        

        step = 1
        for ids in range(frame_start + step, frame_end + step, step):
            
            try:
                params_path = f'{label_output_dir}/smpl_pose/{sub_ids}/{seq_name}/init_{ids-step:03d}_0100.npz'
                init_params = dict(np.load(params_path, allow_pickle=True))
            except:
                params_path = f'{label_output_dir}/smpl_pose/{sub_ids}/{seq_name}/{ids-step:03d}_0100.npz'
                init_params = dict(np.load(params_path, allow_pickle=True))
            print(f'load data from {params_path}')
            
            m_solver.setInitPose(init_params=dict(init_params['arr_0'].item()))

            frame_data = m_data.getFrameData(ids=ids,
                                             init_shape=False,
                                             tracking=True)
            dv_valid, dn_valid = m_cam.preprocessDepth(frame_data['depth_map'], frame_data['mask'])
            dv_floor, dn_normal = m_data.mapDepth2Floor(dv_valid, dn_valid)
            # frame_trans, frame_pose, frame_betas =\
            #     m_solver.modelTracking(
            #     frame_ids=ids,
            #     depth_vmap=dv_floor,
            #     depth_nmap=dn_normal,
            #     color_img=frame_data['img'],
            #     keypoints=frame_data['kp'],
            #     contact_data=frame_data['contact'],
            #     max_iter=args.get('maxiters'))
            frame_trans, frame_pose, frame_betas =\
                m_solver.modelTracking_single_frame(
                frame_ids=ids,
                depth_vmap=dv_floor,
                depth_nmap=dn_normal,
                color_img=frame_data['img'],
                keypoints=frame_data['kp'],
                contact_data=frame_data['contact'],
                max_iter=args.get('maxiters'))

if __name__ == "__main__":
    args = parse_config()
    ic(args)
    main(**args)
