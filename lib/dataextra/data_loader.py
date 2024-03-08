import torch
import glob
import os.path as osp
import numpy as np
import cv2

from torch.utils.data import Dataset


def create_dataset(
        basdir,
        dataset_name=None,
        sub_ids=None,
        seq_name=None,
        start_img_idx=0,
        end_img_idx=-1,
        stage='init_shape', # init_pose, tracking
        ):
    
    return Pressure_Dataset(
                basdir,
                dataset_name,
                sub_ids,
                seq_name, 
                start_img_idx=start_img_idx,
                end_img_idx=end_img_idx   ,
                stage=stage
    )

def read_rtm_kpts(keypoint_fn):
    # data = read_json(keypoint_fn)
    # target_halpe = np.array([data['instance_info'][i]['instances'][0]['keypoints']\
    #     for i in range(len(data['instance_info']))])
    # target_score = np.array([data['instance_info'][i]['instances'][0]['keypoint_scores']\
    #     for i in range(len(data['instance_info']))])
    # target_score = target_score[:, :, np.newaxis]
    
    # target_keypoints = np.concatenate([target_halpe, target_score], axis=2)

    # try:
    #     data = np.load(keypoint_fn, allow_pickle=True)[0] # select first people to tracking
    # except:
    #     data = dict(np.load(keypoint_fn, allow_pickle=True).item())# only one people
    
    data = dict(np.load(keypoint_fn, allow_pickle=True).item())# only one people

    points = np.array(data['keypoints'])
    points_score = np.array(data['keypoint_scores'])
    points_score = points_score[:, np.newaxis]
    target_keypoints = np.concatenate([points, points_score], axis=1)
    
    return target_keypoints

def load_contact(contact_fn):
    insole_data = dict(np.load(contact_fn, allow_pickle=True).item())
    region_l, region_r = insole_data['insole'][0], insole_data['insole'][1]
    contact_label = [[0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]]
    
    row_range = [range(0, 8), range(8, 15), range(15, 23), range(23, 31)]
    
    # 左脚
    for row in range(region_l.shape[0]):
        for col in range(region_l.shape[1]):
            if region_l[row][col] != 0:    
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 7):
                        contact_label[0][1] = 1
                    if col in range(7, 11):
                        contact_label[0][0] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[0][4] = 1
                    if col in range(4, 8):
                        contact_label[0][3] = 1
                    if col in range(8, 11):
                        contact_label[0][2] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 4):
                        contact_label[0][6] = 1
                    if col in range(4, 11):
                        contact_label[0][5] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 4):
                        contact_label[0][8] = 1
                    if col in range(4, 11):
                        contact_label[0][7] = 1
    # 右脚
    for row in range(region_r.shape[0]):
        for col in range(region_r.shape[1]):
            if region_r[row][col] != 0:    
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 4):
                        contact_label[1][0] = 1
                    if col in range(4, 11):
                        contact_label[1][1] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[1][2] = 1
                    if col in range(4, 8):
                        contact_label[1][3] = 1
                    if col in range(8, 11):
                        contact_label[1][4] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 8):
                        contact_label[1][5] = 1
                    if col in range(8, 11):
                        contact_label[1][6] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 8):
                        contact_label[1][7] = 1
                    if col in range(8, 11):
                        contact_label[1][8] = 1
        return contact_label

def load_cliff(cliff_fn):
    cliff_data = dict(np.load(cliff_fn).items())
    if cliff_data['pose'].shape[0] > 1:
        init_pose = np.expand_dims(cliff_data['pose'][0], 0) 
    else:
        init_pose = cliff_data['pose']    
    return init_pose


class Pressure_Dataset(Dataset):
    def __init__(self,
                 basdir,
                 dataset_name,
                 sub_ids,
                 seq_name,
                 start_img_idx=0,
                 end_img_idx=-1,
                 dtype=torch.float32,
                 stage='init_shape', # init_pose, tracking
                 ):
        super(Pressure_Dataset, self).__init__()
        
        self.dtype = dtype
        
        self.basdir = basdir
        self.dataset_name = dataset_name
        self.sub_ids = sub_ids
        self.seq_name = seq_name
        
        self.start_idx = int(start_img_idx)
        self.end_idx = int(end_img_idx)
        
        self.stage = stage

        self.cnt = 0

        self.rgbd_path = osp.join(basdir, 'images', self.dataset_name, self.sub_ids, self.seq_name)
        # rgb
        self.img_paths = sorted(glob.glob(osp.join(self.rgbd_path, 'color', '**'), recursive=True))
        self.img_paths = [x for x in filter(lambda x: x.split('.')[-1] == 'png', self.img_paths)][self.start_idx:self.end_idx]
        # depth
        self.depth_paths = sorted(glob.glob(osp.join(self.rgbd_path, 'depth', '**'), recursive=True))
        self.depth_paths = [x for x in filter(lambda x: x.split('.')[-1] == 'png', self.depth_paths)][self.start_idx:self.end_idx]    
        # depth_mask
        self.dmask_paths = sorted(glob.glob(osp.join(self.rgbd_path, 'depth_mask', '**'), recursive=True))# mask
        self.dmask_paths = [x for x in filter(lambda x: x.split('.')[-1] == 'png', self.dmask_paths)][self.start_idx:self.end_idx]
        # keypoints
        self.kp_paths = sorted(glob.glob(osp.join(self.rgbd_path, 'keypoints', '**'), recursive=True))
        self.kp_paths = [x for x in filter(lambda x: x.split('.')[-1] == 'npy', self.kp_paths)][self.start_idx:self.end_idx]
        # pressure data, A-pose has no pressure data
        self.pressure_paths = sorted(glob.glob(osp.join(self.rgbd_path, 'insole', '**'), recursive=True))
        self.pressure_paths = [x for x in filter(lambda x: x.split('.')[-1] == 'npy', self.pressure_paths)][self.start_idx:self.end_idx]
        
        self.joint_mapper = self.init_joint_mapper()
        # check 
        
    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(25,dtype=np.float32)        
        # These joints are ignored becaue SMPL has no neck.
        optim_weights[1] = 0        
        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[3,6,10,13,4,7]] = 2

        optim_weights[17] = 0
        optim_weights[18] = 0

        return torch.tensor(optim_weights)

    def init_joint_mapper(self):
        openposemap = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])[np.newaxis, :] 
        halpemap = np.array([0,18,6,8,10,5,7,9,19,12,14,16,11,13,15,2,1,4,3,20,22,24,21,23,25])[np.newaxis, :]  
        return np.concatenate([openposemap, halpemap], axis=0).tolist()
        
    def __iter__(self):
        return self        

    def __next__(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        self.cnt += 1

        return self.read_item(self.cnt - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.read_item(idx)


    def read_item(self, idx):
        # load data
        
        # rgb
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path).astype(np.float32)# [:, :, ::-1] / 255.0
        # depth
        depth_path = self.depth_paths[idx]
        depth_map = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_mask
        dmask_path = self.dmask_paths[idx]
        mask_ori = cv2.imread(dmask_path)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask_ori, kernel, 1)
        dmask = np.mean(mask,axis=-1)
        # keypoints
        kp_path = self.kp_paths[idx]
        keypoints = read_rtm_kpts(kp_path)
        frame_kp = torch.from_numpy(keypoints).float()
        # insole pressure
        if self.stage != 'init_shape':
            pressure_path = self.pressure_paths[idx]
            contact_label = load_contact(pressure_path)
        else:
            contact_label = None
        
        # load cliff initial pose data
        if self.stage == 'init_pose':
            cliff_path = osp.join(self.basdir, 'annotations', self.dataset_name, 'init_pose_cliff', f'{self.seq_name}_cliff_hr48.npz')
            init_pose = load_cliff(cliff_path)
        else:
            init_pose = None
            
        output_dict = {
                    'root_path':self.rgbd_path,
                    'depth_map':depth_map,
                    'img': img,
                    'depth_mask': dmask,
                    'kp':frame_kp,
                    'contact_label':contact_label,
                    'init_pose':init_pose
                    }
        
        
        return output_dict
