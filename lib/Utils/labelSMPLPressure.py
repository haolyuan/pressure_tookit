import os,sys
import os.path as osp
import pickle
import glob
import cv2
import trimesh
from tqdm import tqdm,trange
import numpy as np
from icecream import ic
sys.path.append('E:/projects/pressure_toolkit')
from lib.dataset.insole_sync import insole_sync
from color_utils import rgb_code
from vis_foot import visFootImage
from fileio import saveImgSeqAsvideo

class SMPLFootLabel():
    def __init__(self,basedir,sub_ids,seq_name):
        self.basedir = basedir
        self.sub_ids = sub_ids
        self.seq_name = seq_name

        self.footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
        self.footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
        self.footLR = np.loadtxt('essentials/footLR_ids.txt').astype(np.int32)
        self.insole2smplR = np.load('essentials/pressure/insole2smplR.npy', allow_pickle=True).item()
        self.insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()

        self.m_fimg = visFootImage()

        insole_name = insole_sync[self.sub_ids][self.seq_name]
        insole_path = osp.join(self.basedir, 'insole_pkl', insole_name + '.pkl')
        with open(insole_path, "rb") as f:
            insole_data = pickle.load(f)
        indice_name = osp.join(self.basedir, 'Synced_indice', insole_name + '*')
        synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
        self.insole_data = insole_data[synced_indice]

    def getVertsPress(self, press_data,footIds,insole2smpl):
        smpl_press = np.zeros([footIds.shape[0]],dtype=np.float32)
        for i in range(footIds.shape[0]):
            ids = footIds[i]
            if str(ids) in insole2smpl.keys():
                tmp = insole2smpl[str(ids)]
                _data = press_data[tmp[0],tmp[1]]
                if _data.shape[0] != 0:
                    smpl_press[i] = np.sum(_data,axis=0)
        return smpl_press

    def show_insole(self,data):
        press_dim,rows,cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0] * 5)
        imgR = np.uint8(data[1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        # imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        return imgColor

    def visVertslabel(self,img_H=512,save_dir=None):
        for frame_ids in trange(self.insole_data.shape[0]):
            img = cv2.imread(osp.join(self.basedir,'RGBD', self.sub_ids,self.seq_name,'color/%03d.png'%frame_ids))
            press_data = self.insole_data[frame_ids]
            press_img = self.show_insole(press_data)
            smpl_pressL = self.getVertsPress(press_data[0],self.footIdsL,self.insole2smplL)
            smpl_pressR = self.getVertsPress(press_data[1],self.footIdsR,self.insole2smplR)
            smpl_press = np.concatenate([smpl_pressL,smpl_pressR])
            contact_labels = np.zeros(smpl_press.shape[0])
            contact_labels[smpl_press>10] = 1

            cont_img = self.m_fimg.drawContactSMPL(contact_labels)
            smpl_img = self.m_fimg.drawPressureSMPL(press_img)
            img = cv2.resize(img, (int(img.shape[1]/(img.shape[0]/img_H)), img_H))
            press_img = cv2.resize(press_img, (int(press_img.shape[1]/(press_img.shape[0]/img_H)), img_H))
            smpl_img = cv2.resize(smpl_img, (int(smpl_img.shape[1]/(smpl_img.shape[0]/img_H)), img_H))
            cont_img = cv2.resize(cont_img, (int(cont_img.shape[1]/(cont_img.shape[0]/img_H)), img_H))
            save_img = np.concatenate([img,press_img,smpl_img,cont_img],axis=1)
            cv2.imwrite(osp.join(save_dir,'%03d.png'%frame_ids),save_img)


if __name__ == "__main__":
    m_sfl = SMPLFootLabel(
        basedir='E:/dataset/PressureDataset/20230422',
        sub_ids='S12',
        seq_name = 'MoCap_20230422_145422')
    m_sfl.visVertslabel(save_dir='debug/insole2smpl/pressure')
