import torch
import torch.nn as nn
import cv2
import time
import trimesh
import numpy as np
import pickle
from matplotlib import pyplot as plt
from icecream import ic

from lib.Utils.UVGenerator import UVGenerator

def readMask(maskPath):
    maskFileHandler = open(maskPath, "r")
    mask = []
    while True:
        # Get next line from file
        line = maskFileHandler.readline()
        # If line is empty then end of file reached
        if not line:
            break
        # print(line.split())
        mask.append(line.split())

    return np.array(mask, dtype=np.int32)

class PressureTerm(nn.Module):
    def __init__(self,
                 dtype=np.float32,
                 device='cpu'):
        super(PressureTerm, self).__init__()
        self.dtype = dtype
        self.device = device
        self.masksmplL = np.loadtxt("essentials/pressure/masksmplL.txt").astype(np.int32)
        self.masksmplR = np.loadtxt("essentials/pressure/masksmplR.txt").astype(np.int32)
        self.insoleMaskL = np.loadtxt("essentials/pressure/insoleMaskL.txt").astype(np.int32)#maskL
        self.insoleMaskR = np.loadtxt("essentials/pressure/insoleMaskR.txt").astype(np.int32)#maskR

        self.foot_ids = np.zeros(6890,dtype=np.int32)
        for i in range(self.masksmplL.shape[0]):
            for j in range(self.masksmplL.shape[1]):
                if self.masksmplL[i, j] > 0:
                    self.foot_ids[self.masksmplL[i, j]] += 1
                if self.masksmplR[i, j] > 0:
                    self.foot_ids[self.masksmplR[i, j]] += 1
        self.m_uv = UVGenerator()


    def show_insole_to_smpl(self):
        pklPath = "E:/dataset/PressureDataset/S12/insole_pkl/S12-跳绳-1.pkl"
        pressureData = np.load(pklPath, allow_pickle=True)# (2338, 2, 31, 11)
        print(pressureData.shape)

        points = trimesh.load('debug/hello_smpl.obj').vertices
        footLeft = points[self.masksmplL]
        footRight = points[self.masksmplR]

        fig = plt.figure(dpi=200)
        pressureIndex = np.zeros((31, 11, 2))
        for i in range(pressureIndex.shape[0]):
            for k in range(pressureIndex.shape[1]):
                pressureIndex[i, k] = np.array([pressureIndex.shape[0] - i, pressureIndex.shape[1] - k])


        index = 0
        for frame in range(pressureData.shape[0], 3):
            start = time.time()
            plt.clf()
            print(frame)

            vertexesL = []
            vertexesR = []
            vertexesRDict = {}


            for i in range(31):
                for k in range(11):
                    isExist = False
                    if self.masksmplR[i][k] != 0:
                        if False:
                            for vertex in vertexesR:
                                if vertex.index == masksmplR[i][k]:
                                    vertex.addPressure(pressureData[frame, 1, i, k])
                                    isExist = True
                                    # print(vertex.pressure)
                                    break
                            if not isExist:
                                vertexesR.append(Vertex(masksmplR[i][k], pressureData[frame, 1, i, k]))

                        if True:
                            if self.masksmplR[i][k] in vertexesRDict.keys():
                                vertexesRDict[self.masksmplR[i][k]] = vertexesRDict[self.masksmplR[i][k]] + pressureData[
                                    frame, 1, i, k]
                            else:
                                vertexesRDict[self.masksmplR[i][k]] = pressureData[frame, 1, i, k]

            pressureMapL = np.zeros((31, 11))
            pressureMapR = np.zeros((31, 11))

            for i in range(31):
                for k in range(11):
                    # if masksmplL[i][k] != 0:
                    #     pressureMapL[i,k] = getPressure(masksmplL[i,k],vertexesL)
                    if self.masksmplR[i][k] != 0:
                        if False:
                            pressureMapR[i, k] = getPressure(self.masksmplR[i, k], vertexesR)
                        if True:
                            pressureMapR[i, k] = vertexesRDict[self.masksmplR[i, k]]

            plt.subplot(121)
            plt.xlim((-0.16, -0.06))
            plt.ylim((-0.09, 0.17))
            # plt.scatter(footLeft[:,:,0],footLeft[:,:,2], c=pressureMapL[:,:],cmap='Blues')  #绘制散点图
            plt.scatter(footRight[:, :, 0], footRight[:, :, 2], c=pressureMapR[:, :], cmap='Blues')  # 绘制散点图
            plt.colorbar()
            # plt.show()
            plt.subplot(122)
            # plt.scatter(pressureIndex[:,:,1],pressureIndex[:,:,0], c=pressureData[frame,0,:,:],cmap='Blues')  #绘制散点图
            plt.scatter(pressureIndex[:, :, 1], pressureIndex[:, :, 0], c=pressureData[frame, 1, :, :],
                        cmap='Blues')  # 绘制散点图

            plt.colorbar()
            # plt.savefig('figs/Right/%04d.png'%index)

            plt.pause(0.01)
            index += 1
            end = time.time()
            print("time:", end - start)


    def show_insole(self,data):
        press_dim,rows,cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0] * 5)
        imgR = np.uint8(data[1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
        color_map = cv2.applyColorMap(imgLarge, cv2.COLORMAP_HOT)
        # cv2.imshow("img", imgColor)
        # cv2.waitKey(1)
        return color_map

    def insole2smpl(self,
                    frame_ids=None,
                    insole_data=None,
                    th=3.0,is_visual=False):
        smpl_press = np.zeros(6890,dtype=np.float32)
        for i in range(self.masksmplL.shape[0]):
            for j in range(self.masksmplL.shape[1]):
                if self.insoleMaskL[i, j] > 0:
                    smpl_press[self.masksmplL[i, j]] += insole_data[0,i,j]
                if self.insoleMaskR[i, j] > 0:
                    smpl_press[self.masksmplR[i, j]] += insole_data[1,i,j]

        foot_mask = np.where(self.foot_ids>0)[0]
        smpl_press[foot_mask] = smpl_press[foot_mask]/self.foot_ids[foot_mask]

        if is_visual:
            smpl_press[smpl_press<th] = 0
            smpl_color = np.uint8(smpl_press * 5)
            imgColor = cv2.applyColorMap(smpl_color, cv2.COLORMAP_HOT).reshape([-1,3])
            im = self.m_uv.getContUVMap(imgColor, save_path=None)
            color_map = self.show_insole(insole_data)
            im[:color_map.shape[0],:color_map.shape[1],:] = color_map
            cv2.imwrite('debug/insole2smpl/%03d.png'%frame_ids,im)

        contact_label = (smpl_press>th).astype(np.int32)
        return contact_label