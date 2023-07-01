import os
import os.path as osp
import numpy as np
import smplx
import torch
import trimesh
from icecream import ic

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

    m_cam = RGBDCamera(
        basdir=args.get('basdir'),
        dataset_name=args.get('dataset'),
        sub_ids=args.get('sub_ids'),
        seq_name=args.get('seq_name'),
    )

    m_solver = SMPLSolver(
        model_path=args.get('model_folder'),
        num_betas=args.get('num_shape_comps'),
        gender=args.get('model_gender'),
        color_size=args.get('color_size'), depth_size=args.get('depth_size'),
        cIntr=m_cam.cIntr_cpu, dIntr=m_cam.dIntr_cpu,
        device=device
    )

    frame_data = m_data.getFrameData(ids=13)
    dv_valid,dn_valid = m_cam.preprocessDepth(frame_data['depth_map'],frame_data['mask'])
    dv_floor,dn_normal = m_data.mapDepth2Floor(dv_valid,dn_valid)
    m_solver.initShape(depth_vmap=dv_floor,depth_nmap=dn_normal,
                       depth2floor=m_data.depth2floor,
                       depth2color=m_cam.d2c_cpu,
                       color_img=frame_data['img'],
                       keypoints=None)


if __name__ == "__main__":
    args = parse_config()
    ic(args)
    main(**args)
