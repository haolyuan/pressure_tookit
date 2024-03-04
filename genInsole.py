import numpy as np
import os,sys
import os.path as osp
import cv2,csv
import torch
from tqdm import tqdm,trange
from icecream import ic

from lib.Dataset.InsoleModule import InsoleModule
from lib.Utils.fileio import saveImgSeqAsvideo#(basdir,fps=30,ratio=1.0,color=[0,0,255]):

def drawPseudoColo():
    img = np.ones([10,255], dtype=np.uint8)#*51
    for i in range(255):
        img[:,i] = img[:,i]*i
    imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite('debug/COLORMAP_JET.png',imgColor)

def saveInsole():
    ic()
    # basdir = '/data/PressureDataset/20230713'
    # sub_ls = sorted([x for x in os.listdir(basdir) if len(x)<4])
    # for sub_ids in sub_ls:
    #     sub_dir = osp.join(basdir,sub_ids)
    #     seq_ls = sorted(os.listdir(sub_dir))
    #     for seq_name in tqdm(seq_ls):
    #         seq_dir = osp.join(sub_dir,seq_name,'region_contact')
    #         if osp.exists(seq_dir):
    #             frame_ls = sorted(os.listdir(seq_dir))
    #             save_dir = osp.join(sub_dir,seq_name,'insole')
    #             if not osp.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             for fram_fn in frame_ls:
    #                 fn = osp.join(seq_dir,fram_fn)
    #                 insole_data = np.load(fn,allow_pickle=True).item()
    #                 time_stamp = insole_data['time_stamp']
    #                 insole_press = insole_data['insole']
    #                 region_contact = insole_data['contact_label']
    #                 for lfi in range(2):
    #                     sum_region_cont = np.sum(region_contact[lfi])
    #                     if sum_region_cont<0.5:
    #                         insole_press[lfi] = 0
    #                 resultes = {
    #                     'time_stamp':time_stamp,
    #                     'insole':insole_press,
    #                     'visual':True
    #                 }
    #                 np.save(osp.join(save_dir,fram_fn),resultes)

def csv2npy(csv_file_path,frame_range):
    contact_flag=np.ones([frame_range,2])
    with open(csv_file_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f)
        for r, row in enumerate(reader):
            if r>0:
                idx = int(row[0][:3])
                R_label = int(row[1])
                if R_label ==1:
                    contact_flag[idx,1] = 0
                L_label = int(row[2])
                if L_label ==1:
                    contact_flag[idx,0] = 0
            else:
                print('RL:',row)
    return contact_flag

def savePressure(seq_name=None,start_frame=-1,end_frame=-1,frame_offset=0,ratio=10):
    data_dir = '/data/PressureDataset/20230713'
    src_dir = "/data/20230713_rtm/"
    sub_ids = 'S01'

    debug_dir = 'debug/Sync'
    sub_weight = 50*9.8#sub_info[sub_ids]['weight']

    # load insole
    insoleFile = src_dir + "insole_pkl/%s.pkl"%(seq_name)
    insole_data = np.load(insoleFile,allow_pickle=True)
    indiceFile = src_dir + "Synced_indice/%s.txt"%(seq_name)
    syncIndice = np.loadtxt(indiceFile).astype(np.int32)
    syncIndice = syncIndice+frame_offset

    # load csv
    contact_labels = csv2npy('debug/%s.csv'%seq_name, end_frame+1)

    m_insole = InsoleModule('/data/PressureDataset')

    save_dir = osp.join(data_dir,sub_ids,seq_name,'insole')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for frame_ids in trange(start_frame,end_frame+1):
        live_indice = syncIndice[frame_ids]
        insole_frame = insole_data[live_indice]
        contact_flag = contact_labels[frame_ids,:]
        if contact_flag[0]==0:#left
            insole_frame[0,:]=0
        if contact_flag[1]==0:#left
            insole_frame[1,:]=0
        resultes = {
                'time_stamp':live_indice,
                'insole':insole_frame,
                'visual':True
                }
        np.save(osp.join(save_dir,'%03d.npy'%frame_ids),resultes)

    insole_ls = sorted(os.listdir(save_dir))
    for insole_fn in tqdm(insole_ls):
        insole_results = np.load(osp.join(save_dir,insole_fn),allow_pickle=True).item()
        insole_frame = insole_results['insole']

        imgColor = m_insole.show_insole(insole_frame)
        imgColor = cv2.resize(imgColor,(int(imgColor.shape[1]*ratio),int(imgColor.shape[0]*ratio)))

        insole = m_insole.sigmoidNorm(insole_frame,sub_weight)
        img_norm = m_insole.showNormalizedInsole(insole)
        img_norm = cv2.resize(img_norm,(int(img_norm.shape[1]*ratio),int(img_norm.shape[0]*ratio)))

        contact_map = m_insole.press2Cont(insole_frame,sub_weight,th=0.5)
        cont_img = m_insole.showContact(contact_map)
        cont_img = cv2.resize(cont_img,(int(cont_img.shape[1]*ratio),int(cont_img.shape[0]*ratio)))

        img_press = np.concatenate([imgColor,img_norm,cont_img],axis=1)

        img_path = src_dir+'color/%s/%s/%s.png'%(sub_ids,seq_name,insole_fn[:3])
        img_ori = cv2.imread(img_path)
        img_ori = cv2.resize(img_ori,(img_press.shape[1],int(img_ori.shape[0]/(img_ori.shape[1]/img_press.shape[1]))))
        img = np.concatenate([img_ori,img_press],axis=0)
        cv2.imwrite(osp.join(debug_dir,'%s.png'%insole_fn[:3]),img)

if __name__ == '__main__':
    savePressure(seq_name='A2',frame_offset=0,
                start_frame = 40,end_frame = 338)
