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
    def __init__(self,basedir):
        self.basedir = basedir

        self.m_fimg = visFootImage()

        self.footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
        self.footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
        self.footLR = np.loadtxt('essentials/footLR_ids.txt').astype(np.int32)
        self.insole2smplR = np.load('essentials/pressure/insole2smplR.npy', allow_pickle=True).item()
        self.insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()
        self.RegionInsole2SMPLL = np.load('essentials/pressure/RegionInsole2SMPLL.npy', allow_pickle=True).item()
        self.RegionInsole2SMPLR = np.load('essentials/pressure/RegionInsole2SMPLR.npy', allow_pickle=True).item()

        if self.footIdsL.shape[0] == self.footIdsR.shape[0]:
            self.footVertsNum = self.footIdsL.shape[0]


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

    def visVertslabel(self,insole_data,sub_ids,seq_name,img_H=512,save_dir=None):
        for frame_ids in trange(insole_data.shape[0]):
            img = cv2.imread(osp.join(self.basedir,'RGBD', sub_ids,seq_name,'color/%03d.png'%frame_ids))
            press_data = insole_data[frame_ids]
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

    def catRegion(self,is_cat=True):
        if is_cat:
            insoleRegionL, smplRegionL = self.RegionInsole2SMPLL['insole'], self.RegionInsole2SMPLL['smpl']
            insoleRegionR, smplRegionR = self.RegionInsole2SMPLR['insole'], self.RegionInsole2SMPLR['smpl']
            ic(insoleRegionL.keys())
            _insoleRegionL={}
            _insoleRegionL['Toe'] = np.concatenate([insoleRegionL['BigToe'],insoleRegionL['MidToe'],
                                                    insoleRegionL['SmallToe']], axis=1)
            _insoleRegionL['Front'] = np.concatenate([insoleRegionL['Front0'],insoleRegionL['Front1'],
                                                    insoleRegionL['Front2']], axis=1)
            _insoleRegionL['Mid'] = np.concatenate([insoleRegionL['Mid0'],insoleRegionL['Mid1']], axis=1)
            _insoleRegionL['Back'] = np.concatenate([insoleRegionL['Back0'],insoleRegionL['Back1']], axis=1)

            _smplRegionL = {}
            _smplRegionL['Toe'] = np.array(list(set(smplRegionL['BigToe'].tolist() +
                                smplRegionL['MidToe'].tolist() + smplRegionL['SmallToe'].tolist())))
            _smplRegionL['Front'] = np.array(list(set(smplRegionL['Front0'].tolist() +
                                smplRegionL['Front1'].tolist() + smplRegionL['Front2'].tolist())))
            _smplRegionL['Mid'] = np.array(list(set(smplRegionL['Mid0'].tolist() + smplRegionL['Mid1'].tolist())))
            _smplRegionL['Back'] = np.array(list(set(smplRegionL['Back0'].tolist() + smplRegionL['Back1'].tolist())))

            _insoleRegionR={}
            _insoleRegionR['Toe'] = np.concatenate([insoleRegionR['BigToe'],insoleRegionR['MidToe'],
                                                    insoleRegionR['SmallToe']], axis=1)
            _insoleRegionR['Front'] = np.concatenate([insoleRegionR['Front0'],insoleRegionR['Front1'],
                                                    insoleRegionR['Front2']], axis=1)
            _insoleRegionR['Mid'] = np.concatenate([insoleRegionR['Mid0'],insoleRegionR['Mid1']], axis=1)
            _insoleRegionR['Back'] = np.concatenate([insoleRegionR['Back0'],insoleRegionR['Back1']], axis=1)

            _smplRegionR = {}
            _smplRegionR['Toe'] = np.array(list(set(smplRegionR['BigToe'].tolist() +
                                smplRegionR['MidToe'].tolist() + smplRegionR['SmallToe'].tolist())))
            _smplRegionR['Front'] = np.array(list(set(smplRegionR['Front0'].tolist() +
                                smplRegionR['Front1'].tolist() + smplRegionR['Front2'].tolist())))
            _smplRegionR['Mid'] = np.array(list(set(smplRegionR['Mid0'].tolist() + smplRegionR['Mid1'].tolist())))
            _smplRegionR['Back'] = np.array(list(set(smplRegionR['Back0'].tolist() + smplRegionR['Back1'].tolist())))
        else:
            _insoleRegionL, _smplRegionL = self.RegionInsole2SMPLL['insole'],self.RegionInsole2SMPLL['smpl']
            _insoleRegionR, _smplRegionR = self.RegionInsole2SMPLR['insole'],self.RegionInsole2SMPLR['smpl']
        return _insoleRegionL, _smplRegionL, _insoleRegionR, _smplRegionR

    def press2contact(self,press_data,footIds,insoleRegion,smplRegion,th=50.):
        _label = np.zeros(6890)
        for keyl in insoleRegion.keys():
            insolel = insoleRegion[keyl]
            sum_press = np.sum(press_data[insolel[0, :], insolel[1, :]])
            if sum_press>th:
                _label[smplRegion[keyl]] = 1
        _label = _label[footIds]
        return _label

    def labelContact(self,sub_ids,seq_name):
        insoleRegionL, smplRegionL, insoleRegionR, smplRegionR = self.catRegion(is_cat=True)

        '''load insole data'''
        insole_name = insole_sync[sub_ids][seq_name]
        insole_path = osp.join(self.basedir, 'insole_pkl', insole_name + '.pkl')
        with open(insole_path, "rb") as f:
            insole_data = pickle.load(f)
        indice_name = osp.join(self.basedir, 'Synced_indice', insole_name + '*')
        synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
        insole_data = insole_data[synced_indice]

        contact_label = []
        for frame_ids in trange(insole_data.shape[0]):
            press_data = insole_data[frame_ids]
            left_label = self.press2contact(press_data[0,...], self.footIdsL, insoleRegionL,smplRegionL)
            right_label = self.press2contact(press_data[1,...], self.footIdsR, insoleRegionR,smplRegionR)
            cont_label = np.stack([left_label,right_label])
            contact_label.append(cont_label)
        contact_label = np.stack(contact_label)
        np.save(osp.join(self.basedir,'RGBD',sub_ids,seq_name,'contact_label.npy'),contact_label)



if __name__ == "__main__":
    m_sfl = SMPLFootLabel(basedir='E:/dataset/PressureDataset/20230422')
    m_sfl.labelContact(
        sub_ids='S12',
        seq_name='MoCap_20230422_145422')
