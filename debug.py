import copy

import numpy as np
import torch
import trimesh
import os
import os.path as osp
import math
import cv2,pickle
from tqdm import tqdm,trange
from icecream import ic

from lib.fitSMPL.Camera import RGBDCamera
from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.colorTerm import ColorTerm
from lib.fitSMPL.pressureTerm import PressureTerm
from lib.Utils.fileio import saveJointsAsOBJ,saveImgSeqAsvideo,saveNormalsAsOBJ


def show_insole():
    file = "E:/dataset/PressureDataset/S12/insole_pkl/S12-跳绳-1.pkl"
    # for file in os.listdir(path):
    #     if not file.endswith(".pkl"): continue
    #     file = path + file
    with open(file, "rb") as f:
        data = pickle.load(f)
    with open(file.replace(".pkl", ".timestamps"), "rb") as f:
        timestamps = pickle.load(f)
    ic(len(data), len(timestamps))
    ic(data[0].shape)
    exit()
    rows = 31
    cols = 11
    N_frame = len(data)
    img = np.ones((rows, cols * 2), dtype=np.uint8)
    for i in range(N_frame):
        imgL = np.uint8(data[i, 0] * 5)
        imgR = np.uint8(data[i, 1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        imgColor = cv2.applyColorMap(imgLarge, cv2.COLORMAP_HOT)
        cv2.imshow("img", imgColor)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    exit()

def minorReviseInsole2Smpl():
    insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()
    footLR_ids = np.loadtxt('essentials/footLR_ids.txt').astype(np.int32)
    ic(footLR_ids.shape)
    insole2smplR={}
    for i in range(footLR_ids.shape[0]):
        Li = footLR_ids[i,0]
        Ri = footLR_ids[i,1]
        _l = insole2smplL[str(Li)]
        _r = copy.deepcopy(_l)
        _r[1,:] = 10-_l[1,:]
        insole2smplR[str(Ri)] = _r
    np.save('debug/insole2smplR.npy', insole2smplR)
    exit()

if __name__ == '__main__':
    # minorReviseInsole2Smpl()
    footL_ids = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footR_ids = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    footL2R= np.loadtxt('essentials/footLR_ids.txt').astype(np.int32)
    insole2smplR = np.load('debug/insole2smplR.npy', allow_pickle=True).item()
    insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()
    ic(insole2smplR['6630'])
    ic(insole2smplL['3228'])