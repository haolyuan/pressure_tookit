import os

import cv2,imageio
import trimesh
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from icecream import ic


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

    def mapDepth2Floor(self,pointCloud):
        pointCloud = pointCloud.reshape([-1,3])
        depth_slices_RT = (self.depth2floor[:3, :3] @ pointCloud.T + self.depth2floor.reshape([3, 1])).T
        return depth_slices_RT

    def getFrameData(self,ids):
        #read rgbd
        depth_path = osp.join(self.rgbddir,'depth/frame_%d.png'%ids)
        depth_map = imageio.imread(depth_path).astype(np.float32) / 1000.
        img_path = osp.join(self.rgbddir,'color/frame_%d.png'%ids)
        img = cv2.imread(img_path)
        mask_path = osp.join(self.rgbddir,'mask/frame_%d.png'%ids)
        mask = cv2.imread(mask_path)
        mask = np.mean(mask,axis=-1)

        output_dict = {'depth_map':depth_map,
                       'img': img,
                       'mask': mask,
                       }
        return output_dict

