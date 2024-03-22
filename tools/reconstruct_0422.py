import os
import torch
import os.path as osp
import glob
import numpy as np
import trimesh
from tqdm import tqdm

import sys
sys.path.append('/home/yuanhaolei/Document/code/pressure_toolkit')

from lib.core.smpl_mmvp import SMPL_MMVP
# reconstruct 0422 data structure corresponding to 0611 and 0713

def main():
    root = '/data/yuanhaolei/PressureDataset_label/smpl_pose/'
    output_root = '/data/yuanhaolei/PressureDataset_annotations'
    ref_path = '/data/yuanhaolei/PressureDataset_annotations/results/20230611/S03/S3-A-PAOBU-2/smpl_349.npz'
    ref_data = dict(np.load(ref_path, allow_pickle= True))

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    body_model = SMPL_MMVP(essential_root='/data/nas_data/yuanhaolei/essential_files/essentials',
                           gender='male',
                           stage='init_shape',
                           dtype=dtype).to(device)
    
    
    for sub_idx in os.listdir(root):
        sub_idx = 'S10'
        sub_path = osp.join(root, sub_idx)
        for seq_idx in os.listdir(sub_path):
            seq_idx = 'MoCap_20230422_171506'

            pose_path = sorted(glob.glob(osp.join(sub_path, seq_idx, '**'), recursive=True))
            pose_path = [x for x in filter(lambda x: x.split('.')[-1] == 'npz', pose_path)]
            print(f'process {osp.join(sub_path, seq_idx)}')
            if len(pose_path) == 0:
                print(f'{sub_path}/{seq_idx} have no data, pass')
                continue
            
            assert pose_path[-1].rsplit('/', 1)[-1].startswith('init'), print(pose_path)
            
            os.makedirs(f'{output_root}/meshes/20230422/{sub_idx}/{seq_idx}', exist_ok=True)
            os.makedirs(f'{output_root}/results/20230422/{sub_idx}/{seq_idx}', exist_ok=True)
            
            for frame_path in tqdm(pose_path):
                if frame_path.rsplit('/', 1)[-1].startswith('init'):
                    init_idx = frame_path.rsplit('/')[-1].split('_')[1]
                else:
                    init_idx = frame_path.rsplit('/')[-1].split('_')[0]
                    
                output_results_path = f'{output_root}/results/20230422/{sub_idx}/{seq_idx}/smpl_{init_idx}.npz'
                output_mesh_path = f'{output_root}/meshes/20230422/{sub_idx}/{seq_idx}/smpl_{init_idx}.obj'
                
                data = dict(np.load(frame_path, allow_pickle=True))
                data = dict(data['arr_0'].item())
                # for key in data.keys():
                #     print(data[key].shape, key)
                # print('===========')
                # for key in ref_data.keys():
                #     print(ref_data[key].shape, key)
                # save pose
                np.savez(output_results_path, body_pose=data['body_pose'],
                        shape=data['betas'], global_rot=data['global_orient'],
                        transl=data['transl'], model_scale_opt=data['model_scale_opt'])

                # save mesh
                params_dict = {}
                params_dict['body_pose'] = torch.from_numpy(data['body_pose']).to(dtype=dtype, device=device)
                params_dict['betas'] = torch.from_numpy(data['betas']).to(dtype=dtype, device=device)
                params_dict['model_scale_opt'] = torch.from_numpy(data['model_scale_opt']).to(dtype=dtype, device=device)
                params_dict['global_orient'] = torch.from_numpy(data['global_orient']).to(dtype=dtype, device=device)
                params_dict['transl'] = torch.from_numpy(data['transl']).to(dtype=dtype, device=device)
                
                body_model.setPose(**params_dict)
                body_model.update_shape()
                body_model.init_plane()
                model_output = body_model.update_pose()                

                vertices = model_output.vertices.detach().cpu().numpy().squeeze(0)
                mesh = trimesh.Trimesh(vertices=vertices,
                                        faces=body_model.faces
                                        )
                mesh.export(output_mesh_path)
                
        import pdb;pdb.set_trace()
        
        # import pdb;pdb.set_trace()        
        # save shape for each sub idx
        output_shape_path = osp.join(
            output_root, 'results/20230422', sub_idx, f'init_shape_{sub_idx}.npz')
        np.savez(output_shape_path, shape=data['betas'], model_scale_opt=data['model_scale_opt']) 


def refine_single_frame()   :
    dataset_name = '20230422'
    sub_ids = 'S12'
    seq_name = 'MoCap_20230422_150210'
    frama_idx = 63
    temp_frame_idx = 62

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    ref_path = f'/data/yuanhaolei/PressureDataset_annotations/results/{dataset_name}/{sub_ids}/{seq_name}/smpl_{frama_idx:03d}.npz'
    output_mesh_path = f'/data/yuanhaolei/PressureDataset_annotations/meshes/{dataset_name}/{sub_ids}/{seq_name}/smpl_{frama_idx:03d}.obj'
    
    data = dict(np.load(ref_path, allow_pickle=True))
    
    temp_path = f'/data/yuanhaolei/PressureDataset_annotations/results/{dataset_name}/{sub_ids}/{seq_name}/smpl_{temp_frame_idx:03d}.npz'
    temp_data = dict(np.load(temp_path, allow_pickle=True))
    # import pdb;pdb.set_trace()
    
    
    # refine data
    # data['body_pose'][:, 15*3:15*3+3] = temp_data['body_pose'][:, 15*3:15*3+3]
    # data['body_pose'][:, 12*3:12*3+3] = temp_data['body_pose'][:, 12*3:12*3+3]
    # data['body_pose'][:, 16*3:16*3+3] = temp_data['body_pose'][:, 16*3:16*3+3]
    # data['body_pose'][:, 13*3:13*3+3] = temp_data['body_pose'][:, 13*3:13*3+3]
    # data['body_pose'][:, 11*3:11*3+3] = temp_data['body_pose'][:, 11*3:11*3+3]
    data['body_pose'][:, 17*3:17*3+3] = 0

    
    
    
    
    np.savez(ref_path, body_pose=data['body_pose'],
            shape=data['shape'], global_rot=data['global_rot'],
            transl=data['transl'], model_scale_opt=data['model_scale_opt'])    

    body_model = SMPL_MMVP(essential_root='/data/nas_data/yuanhaolei/essential_files/essentials',
                           gender='male',
                           stage='init_shape',
                           dtype=dtype).to(device)

    # save mesh
    params_dict = {}
    params_dict['body_pose'] = torch.from_numpy(data['body_pose']).to(dtype=dtype, device=device)
    params_dict['betas'] = torch.from_numpy(data['shape']).to(dtype=dtype, device=device)
    params_dict['model_scale_opt'] = torch.from_numpy(data['model_scale_opt']).to(dtype=dtype, device=device)
    params_dict['global_orient'] = torch.from_numpy(data['global_rot']).to(dtype=dtype, device=device)
    params_dict['transl'] = torch.from_numpy(data['transl']).to(dtype=dtype, device=device)
    
    body_model.setPose(**params_dict)
    body_model.update_shape()
    body_model.init_plane()
    model_output = body_model.update_pose()                

    vertices = model_output.vertices.detach().cpu().numpy().squeeze(0)
    mesh = trimesh.Trimesh(vertices=vertices,
                            faces=body_model.faces
                            )
    mesh.export(output_mesh_path)


if __name__ == "__main__":
    main()
    # refine_single_frame()