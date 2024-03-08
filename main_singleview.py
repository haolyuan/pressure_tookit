import sys
import os
import glob

# for cluster rendering
import os.path as osp

import time
import yaml
import torch
import pickle
import numpy as np

from icecream import ic

from lib.dataextra.data_loader import create_dataset
from lib.core.camera import create_camera
from lib.core.fit_single_frame import fit_single_frame
from lib.core.smpl_mmvp import SMPL_MMVP
from lib.config.config import parse_config


def main(**args):
    start = time.time()
    
    # create output folders
    output_folder = osp.expandvars(args.pop('output_dir'))
    if not osp.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for f in ['results', 'meshes']:
            os.makedirs(osp.join(output_folder, f), exist_ok=True)
            print(osp.join(output_folder, f))

    # save arguments of current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # fitting stage
    stage = args.pop('fitting_stage')
    
    # init common param
    demo_basdir = args.pop('basdir')
    demo_dsname = args.pop('dataset')
    demo_subids = args.pop('sub_ids')
    demo_seqname = args.pop('seq_name')
    demo_essential_root = args.pop('essential_root')
    
    # create Dataset from folders
    dataset_obj = create_dataset(
                    basdir=demo_basdir,
                    dataset_name=demo_dsname,
                    sub_ids=demo_subids,
                    seq_name=demo_seqname,
                    start_img_idx=args.pop('start_idx'),
                    end_img_idx=args.pop('end_idx'),
                    stage=stage
    )
    
    # read gender and select model
    body_model = SMPL_MMVP(essential_root=demo_essential_root,
                           gender=args.pop('model_gender'),
                           stage=stage,
                           dtype=dtype).to(device)
    
    
    # Create the camera object
    rgbd_cam = create_camera(
        basdir=demo_basdir,
        dataset_name=demo_dsname,
        sub_ids=demo_subids,
        seq_name=demo_seqname,
    )
    rgbd_cam.to(device=device)
    
    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights()\
                        .to(device=device, dtype=dtype)    
    
    # create subdir to write output
    for f in ['results', 'meshes']:
        curr_output_folder = osp.join(output_folder, f, demo_dsname, demo_subids, demo_seqname)
        os.makedirs(curr_output_folder, exist_ok=True)
        print(f'{f} will be saved in {curr_output_folder}')
    
    
    depth_size, color_size = args.pop('depth_size'), args.pop('color_size')

    for idx in range(len(dataset_obj)):
        # read data
        data = dataset_obj[idx]
        print('Processing: {}'.format(data['root_path'])) 
        
        img = data['img']
        depth_mask = data['depth_mask']
        keypoints = data['kp']
        depth_map = data['depth_map']
        
        # optional data
        contact_label = data['contact_label']
        init_pose = data['init_pose']
        
        fit_single_frame(img=img,
                         depth_mask=depth_mask,
                         keypoints=keypoints,
                         depth_map=depth_map,
                         contact_label=contact_label,
                         init_pose=init_pose,
                         essential_root=demo_essential_root,
                         body_model=body_model,
                         camera=rgbd_cam,
                         depth_size=depth_size,
                         color_size=color_size,
                         joint_weights=joint_weights,
                         joint_mapper=dataset_obj.joint_mapper,
                         stage=stage
                         )

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))



if __name__ == "__main__":
    args = parse_config()
    ic(args)
    main(**args)