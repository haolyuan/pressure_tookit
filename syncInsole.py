import numpy as np
import os
import os.path as osp
import pickle
import torch
import cv2
from tqdm import tqdm,trange
from icecream import ic

import csv
import json
import subprocess

def parseInsoleTimestamp(lines):
    timestamps = []
    for line in lines:
        if len(line) < 5: continue
        line = line.strip()
        hour = float(line[0:2])
        min = float(line[3:5])
        sec = float(line[6:len(line)])
        timestamp = sec + min * 60 + hour * 60 * 60
        timestamps.append(timestamp)
    return np.array(timestamps)

def drawPseudoColo():
    img = np.ones([10,255], dtype=np.uint8)#*51
    for i in range(255):
        img[:,i] = img[:,i]*i
    imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite('debug/COLORMAP_JET.png',imgColor)


def syncTwoSequences(seq1, seq2):
    indice = []
    start_j = 0
    for i in range(len(seq1)):
        min_dist = 999999999.9
        matched_j = 0
        for j in range(len(seq2)):
            dist = abs(seq1[i] - seq2[j])
            if (dist <= min_dist):
                min_dist = dist
                matched_j = j
            if dist > min_dist: break
        start_j = matched_j
        indice.append(matched_j)

    return indice

class InsoleModule():
    def __init__(self, basdir):
        self.basdir = basdir
        self.maskL = np.loadtxt(osp.join(self.basdir,'insole_mask/insoleMaskL.txt')).astype(np.int32)
        self.maskR = np.loadtxt(osp.join(self.basdir,'insole_mask/insoleMaskR.txt')).astype(np.int32)
        self.pixel_num = np.sum(self.maskL) + np.sum(self.maskR)
        self.maskImg = np.concatenate([self.maskL,self.maskR],axis=1)>0.5

    def show_insole(self,data):
        press_dim,rows,cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0] * 5)
        imgR = np.uint8(data[1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        # imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        imgColor[~self.maskImg,:] = [100,100,100]
        return imgColor
    
    def sigmoidNorm(self,insole,pixel_weight,avg=False):
        if avg == False:
            pixel_weight = pixel_weight/self.pixel_num
        insole_norm = (insole-pixel_weight)/pixel_weight
        insole_norm = torch.sigmoid(torch.from_numpy(insole_norm)).detach().cpu().numpy()
        return insole_norm

    def showNormalizedInsole(self,data):
        '''
            vis pressure infered from pressureNet
            input:
                data: 2*31*11 pressure
            return 
                imgColor: 31*22
        '''
        data = (data*255).astype(np.uint8)
        press_dim,rows,cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0])
        imgR = np.uint8(data[1])
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        # imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        imgColor[~self.maskImg,:] = [0,0,0]
        return imgColor
    def showContact(self,contact_label):
        rows,cols = contact_label.shape
        img_cont = np.zeros([rows,cols,3])
        img_cont[self.maskImg,:] = [255,255,255]
        img_cont[contact_label>0.5,:] = [0,0,255]
        return img_cont
    def press2Cont(self,insole,pixel_weight,th=0.7,avg=False):
        ''' vis pressure infered from pressureNet
            input:
                data: 2*31*11
                    pressure data
                pixel_weight: float
                    weight or avg weight
                avg: bool
                    is avg weight or not
            return 
                imgColor
        '''
        press_sigmoid = self.sigmoidNorm(insole,pixel_weight,avg=avg)
        cv2.imwrite('/home/yuanhaolei/temp_linux/press2Cont/sigmoid.png',self.showNormalizedInsole(press_sigmoid))
        contact_label = np.zeros_like(press_sigmoid)
        contact_label[press_sigmoid>th] = 1
        contact_label = np.concatenate([contact_label[0],contact_label[1]],axis=1)
        return contact_label

def load_csv(csv_file_path):
    key_frame_list = []
    with open(csv_file_path, 'rt', encoding='utf8') as f:
        reader = csv.DictReader(f)
        key_list = reader.fieldnames
        data_list = list(reader)
    for idx in range(len(data_list)):
        key_frame_list.append(data_list[idx]['img'])
    return data_list, key_frame_list, key_list

def syncInsole(basdir,
               sub_ids,
               seq_name,
               ratio=10,
               insole_offset=0,
               save_dir=None):
    
    src_dir = '/data/20230611_rtm/'
    basdir = src_dir
    insoleFile = basdir + "insole_pkl/%s/%s.pkl"%(sub_ids,seq_name)
    mocapFile = basdir + "keypoints/%s/%s"%(sub_ids,seq_name)
    offsetFile = basdir + "time_offset/%s/%s.txt"%(sub_ids,seq_name)
    # import pdb;pdb.set_trace()

    # load insole
    insole_data = np.load(insoleFile,allow_pickle=True)
    # load time stamps
    with open(insoleFile.replace(".pkl", ".timestamps"), "rb") as f:
        insoleTimestamps_lines = pickle.load(f)
    insoleTimestamps = parseInsoleTimestamp(insoleTimestamps_lines)
    # frame number to frame time stamps
    mocapFrames = [x for x in os.listdir(mocapFile) if x.endswith('.png')]
    mocapNframe = len(mocapFrames)
    mocapTimestamps = np.arange(mocapNframe) * 0.03333

    offset = np.loadtxt(offsetFile)
    syncIndice = syncTwoSequences(mocapTimestamps + insoleTimestamps[0] + offset, insoleTimestamps)

    m_insole = InsoleModule('/data/PressureDataset')
    sub_info = np.load('/data/PressureDataset/20230611/sub_info.npy',allow_pickle=True).item()
    sub_weight = sub_info[sub_ids]['weight']
    pre_indice = syncIndice[0]
    for frame_ids in trange(len(syncIndice)):
        live_indice = syncIndice[frame_ids]
        if live_indice-pre_indice>=0:
            insole_frame = insole_data[live_indice+ insole_offset]
            imgColor = m_insole.show_insole(insole_frame)
            imgColor = cv2.resize(imgColor,(int(imgColor.shape[1]*ratio),int(imgColor.shape[0]*ratio)))

            insole = m_insole.sigmoidNorm(insole_frame,sub_weight)
            img_norm = m_insole.showNormalizedInsole(insole)
            img_norm = cv2.resize(img_norm,(int(img_norm.shape[1]*ratio),int(img_norm.shape[0]*ratio)))

            contact_map = m_insole.press2Cont(insole_frame,sub_weight,th=0.6)
            cont_img = m_insole.showContact(contact_map)
            cont_img = cv2.resize(cont_img,(int(cont_img.shape[1]*ratio),int(cont_img.shape[0]*ratio)))

            img_press = np.concatenate([imgColor,img_norm,cont_img],axis=1)

            img_path = basdir+'keypoints/%s/%s/%03d.png'%(sub_ids,seq_name,frame_ids)
            img_ori = cv2.imread(img_path)
            img_ori = cv2.resize(img_ori,(img_press.shape[1],int(img_ori.shape[0]/(img_ori.shape[1]/img_press.shape[1]))))
            # ic(img_ori.shape,img_press.shape)
            img = np.concatenate([img_ori,img_press],axis=0)
            cv2.imwrite(save_dir+'%03d.png'%frame_ids,img)
        pre_indice = live_indice

def savePressure(basdir,
                 sub_ids,
                 seq_name,
                 ratio=10,
                 insole_offset=0,
                 start_frame=0,
                 end_frame=0,
                 save_dir=None):
    src_dir = '/data/20230611_rtm/'
    basdir = src_dir
    
    insoleFile = basdir + "insole_pkl/%s/%s.pkl"%(sub_ids,seq_name)
    mocapFile = basdir + "keypoints/%s/%s"%(sub_ids,seq_name)
    offsetFile = basdir + "time_offset/%s/%s.txt"%(sub_ids,seq_name)

    # load insole
    insole_data = np.load(insoleFile,allow_pickle=True)
    # load time stamps
    with open(insoleFile.replace(".pkl", ".timestamps"), "rb") as f:
        insoleTimestamps_lines = pickle.load(f)
    insoleTimestamps = parseInsoleTimestamp(insoleTimestamps_lines)
    # frame number to frame time stamps
    mocapFrames = [x for x in os.listdir(mocapFile) if x.endswith('.png')]
    mocapNframe = len(mocapFrames)
    mocapTimestamps = np.arange(mocapNframe) * 0.03333

    offset = np.loadtxt(offsetFile)
    syncIndice = syncTwoSequences(mocapTimestamps + insoleTimestamps[0] + offset, insoleTimestamps)

    m_insole = InsoleModule('/data/PressureDataset')
    sub_info = np.load('/data/PressureDataset/20230611/sub_info.npy',allow_pickle=True).item()
    sub_weight = sub_info[sub_ids]['weight']

    # load csv
    key_frame_data, key_frame_list, csv_keys =\
        load_csv('/home/yuanhaolei/temp_linux/Sync/output/assigned_classes.csv')
    pre_indice = syncIndice[0]
    
    for frame_ids in trange(start_frame, end_frame+1):
        live_indice = syncIndice[frame_ids]

        if live_indice-pre_indice>=0:  
            insole_frame = insole_data[live_indice+ insole_offset]      
            key_name = f'{frame_ids:03d}.png'
            vis_item = True
            if key_name in key_frame_list:
                idx = key_frame_list.index(key_name)
                if key_frame_data[idx][csv_keys[1]] == '1': #left
                    insole_frame[0,:]=0
                if key_frame_data[idx][csv_keys[2]] == '1': #right
                    insole_frame[1,:]=0   
                if key_frame_data[idx][csv_keys[3]] == '1': #right
                    vis_item = False       
                          
            resultes = {
                    'time_stamp':live_indice,
                    'insole':insole_frame,
                    'visual':vis_item
                    }
            os.makedirs(osp.join(save_dir,'insole'), exist_ok=True)
            temp_output_dir = osp.join(save_dir,'insole', '%03d.npy'%frame_ids)
            np.save(temp_output_dir,resultes)   
        pre_indice = live_indice
         
    
    for frame_ids in trange(start_frame, end_frame+1):
        temp_output_dir = osp.join(save_dir,'insole', '%03d.npy'%frame_ids)
        insole_results = np.load(temp_output_dir, allow_pickle=True).item()
        insole_frame = insole_results['insole']

        imgColor = m_insole.show_insole(insole_frame)
        imgColor = cv2.resize(imgColor,(int(imgColor.shape[1]*ratio),int(imgColor.shape[0]*ratio)))

        insole = m_insole.sigmoidNorm(insole_frame,sub_weight)
        img_norm = m_insole.showNormalizedInsole(insole)
        img_norm = cv2.resize(img_norm,(int(img_norm.shape[1]*ratio),int(img_norm.shape[0]*ratio)))

        contact_map = m_insole.press2Cont(insole_frame,sub_weight,th=0.6)
        cont_img = m_insole.showContact(contact_map)
        cont_img = cv2.resize(cont_img,(int(cont_img.shape[1]*ratio),int(cont_img.shape[0]*ratio)))

        img_press = np.concatenate([imgColor,img_norm,cont_img],axis=1)

        img_path = basdir+'keypoints/%s/%s/%03d.png'%(sub_ids,seq_name,frame_ids)
        img_ori = cv2.imread(img_path)
        img_ori = cv2.resize(img_ori,(img_press.shape[1],int(img_ori.shape[0]/(img_ori.shape[1]/img_press.shape[1]))))
        # ic(img_ori.shape,img_press.shape)
        img = np.concatenate([img_ori,img_press],axis=0)
        os.makedirs(osp.join(save_dir,'img'), exist_ok=True)
        vis_item = insole_results['visual']
        cv2.imwrite(osp.join(save_dir,'img')+f'/{frame_ids:03d}_{vis_item}.png',img)

class KeypointModule():
    def __init__(self, 
                meta_path='/home/zhanghe/PretrainedModels/rtmpose/rtmpose_meta.json'):
        with open(meta_path) as file:
            self.rtm_meta = json.load(file)['meta_info']
        self.keypoint_colors = self.rtm_meta['keypoint_colors']
        self.skeleton_links = self.rtm_meta['skeleton_links']
        self.skeleton_link_colors = self.rtm_meta['skeleton_link_colors']

    def drawKeypoints(self, img, keypoints, bbox):
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
        for i in range(len(keypoints)):
            x = keypoints[i][0]
            y = keypoints[i][1]
            joint_color = self.keypoint_colors['__ndarray__'][i]
            img = cv2.circle(img, (int(x), int(y)), 3,
                            (joint_color[2], joint_color[1], joint_color[0]), -1)
        for j in range(len(self.skeleton_links)):
            ji0 = self.skeleton_links[j][0]
            ji1 = self.skeleton_links[j][1]
            bone_color = self.skeleton_link_colors['__ndarray__'][j]
            img = cv2.line(img, (int(keypoints[ji0][0]), int(keypoints[ji0][1])),
                        (int(keypoints[ji1][0]), int(keypoints[ji1][1])),
                        (bone_color[2], bone_color[1], bone_color[0]), 2)
        return img

# S4-A-PAOBU-1
# S4-A-PAOBU-2
# S4-A-TIAOYUAN-1
# S4-A-TIAOYUAN-3
# S4-B-PAOBU-2
# S4-B-PAOBU-3
# S4-B-SHIXINQIU-2
# S4-B-SHIXINQIU-3

def save_and_visual_kp(basdir,
            sub_ids,
            seq_name,
            ratio=10,
            insole_offset=0,
            select_frame=0,
            save_dir=None):
        src_dir = '/data/20230611_rtm/'
        basdir = src_dir
        mocapFile = basdir + "keypoints/%s/%s"%(sub_ids,seq_name)    

        keypoint_fn = osp.join(mocapFile, f'{select_frame:03d}.npy')
        keypoint_fn_refer = osp.join(mocapFile, f'{select_frame-1:03d}.npy')

        
        data = dict(np.load(keypoint_fn, allow_pickle=True).item())# only one people
        data_ref = dict(np.load(keypoint_fn_refer, allow_pickle=True).item())# only one people

        points = np.array(data['keypoints'])
        points_score = np.array(data['keypoint_scores'])
        # import pdb;pdb.set_trace()
        img_dir = '/data/PressureDataset/20230611'
        image = cv2.imread(os.path.join(img_dir, f'{sub_ids}', f'{seq_name}','color', f'{select_frame:03d}.png'))
        
        # operation
        temp_keypoints = data['keypoints']
        # temp_keypoints[23], temp_keypoints[20] = data['keypoints'][20], data['keypoints'][23]
        # temp_keypoints[21], temp_keypoints[22] = data['keypoints'][22], data['keypoints'][21]
        # temp_keypoints[25], temp_keypoints[24] = data['keypoints'][24], data['keypoints'][25]
        # temp_keypoints[20] = data_ref['keypoints'][20]
        # temp_keypoints[22] = data_ref['keypoints'][22]
        # temp_keypoints[24] = data_ref['keypoints'][24]
        # temp_keypoints[15] = data_ref['keypoints'][15]
        
        
        temp_keypoints[22] = data['keypoints'][20]
        # temp_keypoints[23] = data_ref['keypoints'][23]
        # temp_keypoints[25] = data_ref['keypoints'][25]
        # temp_keypoints[16] = data_ref['keypoints'][16]
                
        kp_module = KeypointModule()
        render_img = kp_module.drawKeypoints(img=image,
                                             keypoints=temp_keypoints,
                                             bbox=data['bbox'][0])
        
        cv2.imshow('img', render_img)
        cv2.waitKey(-1)

        data['keypoints'] = temp_keypoints
        os.makedirs(os.path.join(save_dir, 'keypoints'), exist_ok=True)
        np.save(os.path.join(save_dir, 'keypoints', f'{select_frame:03d}'), data)

#  sudo cp -r /home/yuanhaolei/temp_linux/S04/S4-A-TIAOYUAN-3/insole /data/PressureDataset/20230611/S04/S4-A-TIAOYUAN-3/
#  sudo cp -r /home/yuanhaolei/temp_linux/S04/S4-A-CEHUA-1/insole /data/PressureDataset/20230611/S04/S4-A-CEHUA-1/
#  sudo cp -r /home/yuanhaolei/temp_linux/S04/S4-A-PAOBU-3/insole /data/PressureDataset/20230611/S04/S4-A-PAOBU-3/
#  sudo cp -r /home/yuanhaolei/temp_linux/S04/S4-A-SHIXINQIU-3/insole /data/PressureDataset/20230611/S04/S4-A-SHIXINQIU-3/
#  sudo cp -r /home/yuanhaolei/temp_linux/S04/S4-A-TIAOSHENG-3/insole /data/PressureDataset/20230611/S04/S4-A-TIAOSHENG-3/


def cp_kp():
    sub_ids_list = ['S03', 'S04']
    # src_root = '/data/20230611_rtm/keypoints'
    src_root = '/home/yuanhaolei/temp_linux/'
    dist_root = '/data/PressureDataset/20230611'
    
    for sub_ids in sub_ids_list:
        seq_name_list = os.listdir(os.path.join(src_root, sub_ids))
        for seq_name in seq_name_list:
            seq_dir = os.path.join(src_root, sub_ids, seq_name, 'keypoints')
            dist_dir = os.path.join(dist_root, sub_ids, seq_name, 'keypoints')
            
            if os.path.exists(f'{seq_dir}'):
                for kp_npy in os.listdir(seq_dir):
                    if kp_npy.endswith('.npy'):
                        kp_src_dir = os.path.join(seq_dir, kp_npy)
                        kp_dist_dir = os.path.join(dist_dir, kp_npy)

                        command = ['sudo', 'cp', f'{kp_src_dir}', f'{kp_dist_dir}']
                        proc = subprocess.Popen(command)
                        proc.wait()
                    print(f'process {command}')        
                        
            # print(f'process {seq_dir}')
            # for kp_npy in os.listdir(seq_dir):
            #     if kp_npy.endswith('.npy'):
            #         kp_src_dir = os.path.join(seq_dir, kp_npy)
            #         kp_dist_dir = os.path.join(dist_dir, kp_npy)
                    
            #         command = ['sudo', 'cp', f'{kp_src_dir}', f'{kp_dist_dir}']
            #         proc = subprocess.Popen(command)
            #         proc.wait()

# S04 S4-A-PAOBU-3，左右反了
if __name__ == "__main__":
    # path = "/data/20230611_rtm/"
    path = '/home/yuanhaolei/temp_linux/'
    save_dir = '/home/yuanhaolei/temp_linux/Sync/'
    sub_ids = 'S04'
    seq_name = 'S4-B-TIAOYUAN-1'
    # insole_offset = 5

    
    # visual src pressure data
    # syncInsole(path,
    #            sub_ids,
    #            seq_name,
    #            save_dir=save_dir,
    #            insole_offset=insole_offset)
    
    # select date needed
    # insole_save_dir = f'/home/yuanhaolei/temp_linux/{sub_ids}/{seq_name}/'
    # savePressure(basdir=path,
    #              sub_ids=sub_ids,
    #              seq_name=seq_name,
    #              insole_offset=insole_offset,
    #              start_frame=122,
    #              end_frame=237,
    #              save_dir=insole_save_dir)


    # improve kp data
    # insole_save_dir = f'/home/yuanhaolei/temp_linux/{sub_ids}/{seq_name}/'
    # frame = 269
    # save_and_visual_kp(basdir=path,
    #                    sub_ids=sub_ids,
    #                    seq_name=seq_name,
    #                    select_frame=frame,
    #                    save_dir=insole_save_dir
    #                    )
    
    # save kp data
    cp_kp()