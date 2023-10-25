import os
import cv2,imageio
import trimesh
import json
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import glob
from icecream import ic
from lib.Utils.fileio import read_json
# from lib.dataset.insole_sync import insole_sync

class PressureDataset(Dataset):
    def __init__(self,
                 basdir,
                 dataset_name,
                 sub_ids,
                 seq_name):
        super(PressureDataset, self).__init__()
        self.dtype = torch.float32
        self.basdir = basdir
        self.dataset_name = dataset_name
        self.sub_ids = sub_ids
        self.seq_name = seq_name

        self.rgbddir = osp.join(basdir, self.dataset_name, self.sub_ids, self.seq_name)

        #read floor
        floor_path = osp.join(self.basdir, self.dataset_name, self.sub_ids, 'floor_'+self.sub_ids+'.npy')
        floor_info = np.load(floor_path, allow_pickle=True).item()
        self.floor_trans = floor_info['trans']
        self.floor_normal = floor_info['normal']
        self.depth2floor = floor_info['depth2floor']
        
        self.visual_path = f'debug/mesh_visual/{sub_ids}/{seq_name}'
        os.makedirs(self.visual_path, exist_ok=True)


    def mapDepth2Floor(self,pointCloud,depth_normal):
        pointCloud = pointCloud.reshape([-1,3])
        
        depth_floor = (self.depth2floor[:3, :3] @ pointCloud.T + self.depth2floor[:3, 3].reshape([3, 1])).T
        trimesh.Trimesh(vertices=depth_floor, process=False).export(f'{self.visual_path}/depth_gt.obj')
        # trimesh.Trimesh(vertices=depth_floor, process=False).export(f'{self.visual_path}/depth_pointcloud.obj')
        # import pdb;pdb.set_trace()
        if depth_normal is not None:
            depth_normal = (self.depth2floor[:3, :3] @ depth_normal.T).T
        return depth_floor, depth_normal

    def read_rtm_kpts(self, keypoint_fn):
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

    def load_contact(self, contact_fn):
        insole_data = dict(np.load(contact_fn, allow_pickle=True).item())
        contact_data = insole_data['contact_label']
        return contact_data
    
    def show_insole(self,idx):
        data = self.insole_data[idx]
        press_dim,rows,cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0] * 5)
        imgR = np.uint8(data[1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        imgColor = cv2.applyColorMap(imgLarge, cv2.COLORMAP_HOT)
        # imgColor = cv2.applyColorMap(imgLarge, cv2.COLORMAP_JET)
        cv2.imshow("img", imgColor)
        cv2.waitKey(1)
        return img

    def load_cliff(self, cliff_fn):
        cliff_data = dict(np.load(cliff_fn).items())
        init_pose = cliff_data['pose']
        return init_pose

    def getFrameData(self, ids, init_shape=True):
        """_summary_

        Args:
            ids (_type_): _description_
            tracking (bool, optional): use frame data to init shape or not. Defaults to False.

        Returns:
            _type_: _description_
        """        
        # read rgbd
        depth_path = osp.join(self.rgbddir,'depth/%03d.png'%ids)
        depth_map = imageio.imread(depth_path).astype(np.float32) / 1000.
        img_path = osp.join(self.rgbddir,'color/%03d.png'%ids)
        img = cv2.imread(img_path)
        mask_path = osp.join(self.rgbddir,'depth_mask/%03d.png'%ids)
        mask_ori = cv2.imread(mask_path)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask_ori, kernel, 1)
        mask = np.mean(mask,axis=-1)

        # read keypoints
        keypoint_fn = osp.join(self.rgbddir,'keypoints', f'{ids:03d}.npy')
        keypoints = self.read_rtm_kpts(keypoint_fn)
        frame_kp = keypoints
        # keypoint_fn = osp.join(self.rgbddir, 'results_color.json')
        # keypoints = self.read_rtm_kpts(keypoint_fn)
        # frame_kp = keypoints[ids]

        # read contact data
        contact_fn = osp.join(self.rgbddir, 'insole', f'{ids:03d}.npy')
        contact_data = None if init_shape else self.load_contact(contact_fn)

        # load cliff initial pose data
        cliff_fn = osp.join(self.rgbddir, f'{self.seq_name}_cliff_hr48.npz')
        init_pose = None if init_shape else self.load_cliff(cliff_fn)

        
        # load weight info
        # if init_shape:
        #     insole_sync = np.load(osp.join('D:/dataset/tebu_contact','Sync-list-total.npy'), allow_pickle=True).item()
            
        #     ids_list = list(self.sub_ids)
        #     ids_list.remove('0')
            
        #     temp_ids = ('').join(ids_list)
        #     # A-pose has no insole data, we would get data from other seq from the same sub_ids data
        #     insole_name = insole_sync[temp_ids]['MoCap_20230422_092324']
        #     insole_path = osp.join('D:/dataset/tebu_contact', 'insole_pkl', insole_name + '.pkl')
        #     with open(insole_path, "rb") as f:
        #         insole_data = pickle.load(f)
        #     indice_name = osp.join('D:/dataset/tebu_contact', 'Synced_indice', insole_name + '*')
        #     synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
        #     insole_data = insole_data[synced_indice]
        #     press_data = insole_data[0]
        #     weight_data = np.sum(press_data[0] + press_data[1])
        # else:
        
        weight_data = None
        output_dict = {'depth_map':depth_map,
                       'img': img,
                       'mask': mask,
                       'kp':frame_kp,
                       'contact':contact_data,
                       'cliff_pose':init_pose,
                       'weight_data': weight_data
                       }
        

        
        return output_dict

