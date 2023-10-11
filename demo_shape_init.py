import os
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

    m_data = PressureDataset(
        basdir=args.get('basdir'),
        dataset_name=args.get('dataset'),
        sub_ids=args.get('sub_ids'),
        seq_name=args.get('seq_name'),
    )

    # from lib.fitSMPL.pressureTerm import PressureTerm
    # m_pt = PressureTerm()
    # frame_range = args.get('frame_range')
    # for ids in range(frame_range[0], frame_range[1] + 1):
    #     frame_data = m_data.getFrameData(ids=ids)
    #     m_pt.insole2smpl(ids,frame_data['insole'])
    # exit()

    
    basdir=args.get('basdir')
    dataset_name=args.get('dataset')
    seq_name=args.get('seq_name')
    sub_ids=args.get('sub_ids')
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
        seq_name=args.get('seq_name'),
        device=device
    )

    # import pdb;pdb.set_trace()
    select_idx = int(args.get('init_idx_start'))
    frame_data = m_data.getFrameData(ids=select_idx)
    dv_valid,dn_valid = m_cam.preprocessDepth(frame_data['depth_map'],frame_data['mask'])
    dv_floor,dn_normal = m_data.mapDepth2Floor(dv_valid,dn_valid)
    annot = m_solver.initShape(depth_vmap=dv_floor,depth_nmap=dn_normal,
                        color_img=frame_data['img'],
                        keypoints=frame_data['kp'],
                        max_iter=args.get('maxiters'))
    np.save(f'{basdir}/{dataset_name}/{sub_ids}/init_param_{sub_ids}', annot)

if __name__ == "__main__":
    args = parse_config()
    ic(args)
    main(**args)
