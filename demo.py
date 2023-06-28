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
from lib.fitSMPL.SMPLModel import SMPLModel

def main(**args):
    m_smpl = SMPLModel(
        model_path=args.get('model_folder'),
         num_betas=args.get('num_shape_comps'),
         gender=args.get('model_gender'),
    )

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

    frame_data = m_data.getFrameData(ids=13)
    pointCloud = (m_cam.calcDepth3D(frame_data['depth_map'])).reshape([-1,3])
    trimesh.Trimesh(vertices=pointCloud, process=False).export('debug/pointCloud.obj')
    exit()

    dv_valid,dn_valid = m_cam.preprocessDepth(frame_data['depth_map'],frame_data['mask'])
    trimesh.Trimesh(vertices=dv_valid,process=False).export('debug/depth.obj')


if __name__ == "__main__":
    args = parse_config()
    ic(args)
    main(**args)
