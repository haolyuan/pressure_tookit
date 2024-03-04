import numpy as np
import os,sys
sys.path.append('/home/yuanhaolei/Document/code/pressure_toolkit')
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import griddata
from icecream import ic

from syncInsole import InsoleModule

def toMatlab(insole,insole_mask):
    ic(insole.shape)
    ls=[]
    for hi in range(insole.shape[0]):
        for wi in range(insole.shape[1]):
            if insole_mask[hi,wi]>0.5:
                ls.append(np.array([hi,wi,insole[hi,wi]]))
            else:
                ls.append(np.array([hi, wi, -1]))
    ls = np.stack(ls)
    ic(ls.shape)
    np.savetxt('debug/insole_data.txt',ls)

def drawInsole():
    frame_info = {'data_id':'20230713','sub_id':'S01','seq_name':'A2','frame_idx':127}#annotation_frame
    # frame_info = {'data_id':'20230422','sub_id':'S12','seq_name':'MoCap_20230422_145854','frame_idx':30,'gender':'male'}
    # frame_info = {'data_id':'20230422','sub_id':'S10','seq_name':'MoCap_20230422_172438','frame_idx':94,'gender':'female'}
    # frame_info = {'data_id': '20230422', 'sub_id': 'S03', 'seq_name': 'MoCap_20230422_110125', 'frame_idx': 23,'gender':'female'}
    # frame_info = {'data_id': '20230422', 'sub_id': 'S03', 'seq_name': 'MoCap_20230422_110125', 'frame_idx':30}# [16,22, 27,30,32, 34]
    # frame_info = {'data_id': '20230422', 'sub_id': 'S12', 'seq_name': 'MoCap_20230422_150025', 'frame_idx':53}# [0,60, 70,80,90]
    basdir = '/data/PressureDataset'
    data_id = frame_info['data_id']
    sub_id = frame_info['sub_id']
    seq_name = frame_info['seq_name']
    frame_idx = frame_info['frame_idx']

    # load insole data
    insole_fn = osp.join(basdir,data_id,sub_id,seq_name,'insole/%03d.npy'%frame_idx)
    insole_info = np.load(insole_fn,allow_pickle=True).item()
    insole_data = insole_info['insole']
    # insole_data = np.ones_like(insole_data)
    # insole_data[1, -15:, :] = 0
    # make insole module
    insole_module = InsoleModule('/data/PressureDataset')
    sub_info = {}
    sub_info['20230422'] = np.load(osp.join(basdir, '20230422/sub_info.npy'), allow_pickle=True).item()
    sub_info['20230611'] = np.load(osp.join(basdir, '20230611/sub_info.npy'), allow_pickle=True).item()
    sub_info['20230713'] = np.load(osp.join(basdir, '20230713/sub_info.npy'), allow_pickle=True).item()
    sub_weight = sub_info[data_id][sub_id]['weight']

    # insole_data = insole_module.sigmoidNorm(insole_data, sub_weight)
    insole = np.concatenate([insole_data[0], insole_data[1]], axis=1)
    insole = np.pad(insole, (2, 2), 'constant')
    insole = cv2.resize(insole,(insole.shape[1]*10,insole.shape[0]*10))
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)

    maskImg = np.concatenate([insole_module.maskL, insole_module.maskR], axis=1).astype(np.float32)
    maskImg = np.pad(maskImg, (2, 2), 'constant')
    maskImg = cv2.resize(maskImg, (maskImg.shape[1] * 10, maskImg.shape[0] * 10))

    # insole[maskImg<0.2] = np.nan
    insole = np.concatenate([insole[:,:int(insole.shape[1]/2)],np.zeros([insole.shape[0],10]),insole[:,int(insole.shape[1]/2):]],axis=1)
    insole[:,int(insole.shape[1]/2):(int(insole.shape[1]/2)+10)] = np.nan
    # insole = np.pad(insole, (2, 2), 'constant')
    insole[insole<1e-3]=np.nan
    # insole[:,135:]=np.nan

    # insole[insole<0.3]=np.nan
    # draw
    xx = np.arange(0, insole.shape[0], 1)
    yy = np.arange(0, insole.shape[1], 1)
    X, Y = np.meshgrid(xx, yy)
    Z = insole[X,Y]

    ax = plt.axes(projection='3d')
    ax.patch.set_alpha(0)
    # ax.contour(X, Y, Z, zdir='z', offset=-1, cmap="jet")
    ax.plot_surface(X, Y, Z,rstride =2, cstride =2,cmap='jet',alpha=1)#, linewidth=0
    # ax.plot_trisurf(X, Y, Z,rstride = 1, cstride = 1,cmap='jet',shade=True)

    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    plt.grid(False)

    ax.set_zlim(0, 20, 5)
    # ax.set_zlim(0, 3, 0.1)
    # ax.zaxis.set_major_locator(MultipleLocator(0.5))
    # 设置视角
    # ax.view_init(53, 50)# top-down, left-right
    ax.view_init(40, 140)# top-down, left-right
    # 设置边界
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()


if __name__ == '__main__':
    drawInsole()
    exit()
    basdir = 'Z:'
    data_id = '20230713'
    sub_id = 'S01'
    seq_name = 'A2'
    frame_idx = 135

    # load insole data
    insole_fn = osp.join(basdir,data_id,sub_id,seq_name,'insole/%03d.npy'%frame_idx)
    insole_info = np.load(insole_fn,allow_pickle=True).item()
    insole_data = insole_info['insole']
    insole_data = np.ones_like(insole_data)

    # make insole module
    insole_module = InsoleModule()
    sub_info = {}
    sub_info['20230422'] = np.load(osp.join(basdir, '20230422/sub_info.npy'), allow_pickle=True).item()
    sub_info['20230611'] = np.load(osp.join(basdir, '20230611/sub_info.npy'), allow_pickle=True).item()
    sub_info['20230713'] = np.load(osp.join(basdir, '20230713/sub_info.npy'), allow_pickle=True).item()
    sub_weight = sub_info[data_id][sub_id]['weight']

    # insole_data = insole_module.sigmoidNorm(insole_data, sub_weight)
    insole = np.concatenate([insole_data[0], insole_data[1]], axis=1)
    insole = np.pad(insole, (2, 2), 'constant')
    insole = cv2.resize(insole,(insole.shape[1]*10,insole.shape[0]*10))
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)

    maskImg = np.concatenate([insole_module.maskL, insole_module.maskR], axis=1).astype(np.float32)
    maskImg = np.pad(maskImg, (2, 2), 'constant')
    maskImg = cv2.resize(maskImg, (maskImg.shape[1] * 10, maskImg.shape[0] * 10))

    insole[maskImg<0.2] = np.nan
    insole = np.concatenate([insole[:,:int(insole.shape[1]/2)],np.zeros([insole.shape[0],10]),insole[:,int(insole.shape[1]/2):]],axis=1)
    # insole[:,110:120] = np.nan
    # insole = np.pad(insole, (2, 2), 'constant')
    insole[insole<0.1]=np.nan
    # draw
    xx = np.arange(0, insole.shape[0], 1)
    yy = np.arange(0, insole.shape[1], 1)
    X, Y = np.meshgrid(xx, yy)
    Z = insole[X,Y]
    # zi = griddata((X, Y), Z, (X, Y), method='cubic')

    ax = plt.axes(projection='3d')
    # ax.patch.set_alpha(0)
    # ax.contour(X, Y, Z, zdir='z', offset=-1, cmap="jet")
    ax.plot_surface(X, Y, Z,rstride =2, cstride =2,cmap='jet',shade=True)
    # ax.plot_trisurf(X, Y, Z,rstride = 1, cstride = 1,cmap='jet',shade=True)


    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    plt.grid(False)

    ax.set_zlim(0, 20, 5)
    # ax.set_zlim(0, 3, 0.1)
    # ax.zaxis.set_major_locator(MultipleLocator(0.5))
    # 设置视角
    ax.view_init(60, 130)
    # 设置边界
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
    exit()
    # Z:/20230713/S01/A2/color/135.png


    ax = plt.axes(projection='3d')
    ax.patch.set_alpha(0)
    # 定义坐标轴
    xx = np.arange(-5, 5, 0.1)
    yy = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
    # 作图
    ax.plot_surface(X, Y, Z, cmap='jet')  # cividis pink_r summer viridis winter Greens Pastel1 YlGn
    ax.contour(X, Y, Z, zdir='z', offset=-1, cmap="jet")
    # 设置坐标间隔
    ax.set_zlim(-1, 1, 0.5)
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    # 设置视角
    ax.view_init(30, 45)
    # 设置边界
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()
