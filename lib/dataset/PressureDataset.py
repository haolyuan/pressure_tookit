import os
import cv2,imageio
import trimesh
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import glob
from icecream import ic
from lib.Utils.fileio import read_json
from lib.dataset.insole_sync import insole_sync

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

        self.rgbddir = osp.join(basdir,self.dataset_name,self.sub_ids,'RGBD',self.seq_name)

        #read floor
        floor_path = osp.join(self.basdir, self.dataset_name, self.sub_ids, 'MoCap/Floor_'+self.seq_name[6:]+'.npy')
        floor_info = np.load(floor_path, allow_pickle=True).item()
        self.floor_trans = floor_info['trans']
        self.floor_normal = floor_info['normal']
        self.depth2floor = floor_info['depth2floor']

        insole_name = insole_sync[self.sub_ids][seq_name]
        insole_path = osp.join(self.basdir,self.dataset_name,self.sub_ids,'insole_pkl',insole_name+'.pkl')
        with open(insole_path, "rb") as f:
            insole_data = pickle.load(f)
        # with open(insole_path.replace(".pkl", ".timestamps"), "rb") as f:
        #     insole_timestamps = pickle.load(f)
        indice_name = osp.join(self.basdir,self.dataset_name,self.sub_ids,'Synced_indice',insole_name+'*')
        synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
        self.insole_data = insole_data[synced_indice]

    def mapDepth2Floor(self,pointCloud,depth_normal):
        pointCloud = pointCloud.reshape([-1,3])
        depth_floor = (self.depth2floor[:3, :3] @ pointCloud.T + self.depth2floor[:3, 3].reshape([3, 1])).T
        if depth_normal is not None:
            depth_normal = (self.depth2floor[:3, :3] @ depth_normal.T).T
        return depth_floor, depth_normal

    def read_keypoints(self,keypoint_fn):
        data = read_json(keypoint_fn)

        keypoints = []
        for idx, person_data in enumerate(data['people']):
            body_keypoints = np.array(
                person_data['pose_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate([body_keypoints, left_hand_keyp,
                right_hand_keyp, face_keypoints, contour_keyps], axis=0)

            keypoints.append(body_keypoints)
        return keypoints

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
        cv2.imshow("img", imgColor)
        cv2.waitKey(1)

    def getFrameData(self,ids):
        # read rgbd
        depth_path = osp.join(self.rgbddir,'depth/frame_%d.png'%ids)
        depth_map = imageio.imread(depth_path).astype(np.float32) / 1000.
        img_path = osp.join(self.rgbddir,'color/frame_%d.png'%ids)
        img = cv2.imread(img_path)
        mask_path = osp.join(self.rgbddir,'mask/frame_%d.png'%ids)
        mask = cv2.imread(mask_path)
        mask = np.mean(mask,axis=-1)

        # read keypoints
        keypoint_fn = osp.join(self.rgbddir,'openpose','frame_%d_keypoints.json'%ids)
        keypoints = self.read_keypoints(keypoint_fn)
        body_kp = keypoints[0][:25,:]

        # read insole data
        insole_data = self.insole_data[ids-1] #2,31,11

        output_dict = {'depth_map':depth_map,
                       'img': img,
                       'mask': mask,
                       'kp':body_kp,
                       'insole':insole_data,
                       }
        return output_dict

