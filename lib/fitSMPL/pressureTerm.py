import torch
import torch.nn as nn
import cv2
import time
import trimesh
import numpy as np
import pickle
from matplotlib import pyplot as plt
from icecream import ic


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

        self.dtype=dtype
        self.device=device



    def test(self):
        masksmplL = "essentials/pressure/masksmplL.txt"
        masksmplR = "essentials/pressure/masksmplR.txt"
        maskL = "essentials/pressure/insoleMaskL.txt"
        maskR = "essentials/pressure/insoleMaskR.txt"
        pklPath = "E:/dataset/PressureDataset/S12/insole_pkl/S12-跳绳-1.pkl"

        masksmplL = readMask(masksmplL)
        masksmplR = readMask(masksmplR)
        pressureData = np.load(pklPath, allow_pickle=True)
        print(pressureData.shape)
        # (2338, 2, 31, 11)

        print("maskL------------------------------")
        print(masksmplL)
        print("maskR------------------------------")
        print(masksmplR)

        points = trimesh.load('debug/hello_smpl.obj').vertices
        print(points)
        footLeft = points[masksmplL]
        footRight = points[masksmplR]
        # print(footLeft)

        frame = 276
        # plt.ion()
        fig = plt.figure(dpi=200)
        pressureIndex = np.zeros((31, 11, 2))
        for i in range(pressureIndex.shape[0]):
            for k in range(pressureIndex.shape[1]):
                pressureIndex[i, k] = np.array([pressureIndex.shape[0] - i, pressureIndex.shape[1] - k])

        # print(pressureIndex)

        index = 0
        for frame in range(276, pressureData.shape[0], 3):
            start = time.time()
            plt.clf()
            print(frame)

            vertexesL = []
            vertexesR = []
            vertexesRDict = {}

            # for i in range(31):
            #     for k in range(11):
            #         isExist = False
            #         if masksmplL[i][k] != 0:
            #             for vertex in vertexesL:
            #                 if vertex.index == masksmplL[i][k]:
            #                     vertex.addPressure(pressureData[frame,0,i,k])
            #                     isExist = True
            #                     # print(vertex.pressure)
            #                     break
            #             if not isExist:
            #                 vertexesL.append(Vertex(masksmplL[i][k],pressureData[frame,0,i,k]))

            for i in range(31):
                for k in range(11):
                    isExist = False
                    if masksmplR[i][k] != 0:
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
                            if masksmplR[i][k] in vertexesRDict.keys():
                                vertexesRDict[masksmplR[i][k]] = vertexesRDict[masksmplR[i][k]] + pressureData[
                                    frame, 1, i, k]
                            else:
                                vertexesRDict[masksmplR[i][k]] = pressureData[frame, 1, i, k]

            pressureMapL = np.zeros((31, 11))
            pressureMapR = np.zeros((31, 11))

            for i in range(31):
                for k in range(11):
                    # if masksmplL[i][k] != 0:
                    #     pressureMapL[i,k] = getPressure(masksmplL[i,k],vertexesL)
                    if masksmplR[i][k] != 0:
                        if False:
                            pressureMapR[i, k] = getPressure(masksmplR[i, k], vertexesR)
                        if True:
                            pressureMapR[i, k] = vertexesRDict[masksmplR[i, k]]

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
            # print(len(vertexes))
            # for vertex in vertexes:
            #     print(vertex.index, vertex.pressure)

            end = time.time()
            print("time:", end - start)