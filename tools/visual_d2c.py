import subprocess
import os

if __name__ == "__main__":
    dataset_name = '20230713'
    root_dir = f'/data/PressureDataset/{dataset_name}'
    
    for sub_ids in os.listdir(root_dir):
        for seq_name in os.listdir(os.path.join(root_dir, sub_ids)):
            depth_dir = os.path.join(root_dir, sub_ids, seq_name, 'depth')
            seq_length = len(os.listdir(depth_dir))
    
    
            command = ['python',
                    'demo_shape_init.py',
                    '-c',
                    'configs/init_smpl_rgbd.yaml',
                    '--init_idx_start',
                    f'{seq_length-1}',
                    '--sub_ids',
                    f'{sub_ids}',
                    '--seq_name',
                    f'{seq_name}',
                    '--dataset',
                    f'{dataset_name}'
                    ]
            proc = subprocess.Popen(command)
            proc.wait()