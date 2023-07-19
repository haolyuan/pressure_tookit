import sys
import cv2
import numpy as np
import trimesh
import os.path as osp
from tqdm import tqdm,trange
from icecream import ic

sys.path.append('E:/projects/pressure_toolkit')
from lib.dataset.PressureDataset import PressureDataset
from color_utils import rgb_code
from lib.Utils.fileio import read_json,save_json,saveImgSeqAsvideo

def mapInsole2Smpl():
    footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    masksmplL = np.loadtxt('essentials/pressure/masksmplL.txt').astype(np.int32)
    masksmplR = np.loadtxt('essentials/pressure/masksmplR.txt').astype(np.int32)
    # a = np.arange(0,33*11,dtype=np.int32).reshape([33,11])

    lsR = {}
    for ids in footIdsR:
        tmp = np.stack(np.where(masksmplR == ids))
        lsR[str(ids)] = tmp
    np.save('essentials/pressure/insole2smplR.npy', lsR)

    lsL = {}
    for ids in footIdsL:
        tmp = np.stack(np.where(masksmplL == ids))
        lsL[str(ids)] = tmp
    np.save('essentials/pressure/insole2smplL.npy', lsL)


def drawSMPLFoot(v_foot,footIds,faces,img_H=3300,img_W=1100,vert_color=None,save_name=None):
    tex_color = rgb_code['Purple']
    line_color = rgb_code['Green']
    x_col = img_W-(v_foot[:,0]-np.min(v_foot[:,0]))/(np.max(v_foot[:,0])-np.min(v_foot[:,0]))*(img_W-1)-1
    x_row = img_H-(v_foot[:,2]-np.min(v_foot[:,2]))/(np.max(v_foot[:,2])-np.min(v_foot[:,2]))*(img_H-1)-1

    img = np.ones(((img_H+50),(img_W+100),3), dtype=np.uint8) * 255
    point = np.concatenate([x_row.reshape([-1,1]).astype(np.int32),x_col.reshape([-1,1])],axis=1)

    for j in range(faces.shape[0]):
        x,y,z = faces[j]
        if x in footIds and y in footIds:
            xi = np.where(footIds==x)[0]
            yi = np.where(footIds==y)[0]
            img = cv2.line(img, (int(point[xi, 1]), int(point[xi, 0])),
                           (int(point[yi, 1]), int(point[yi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)
        if z in footIds and y in footIds:
            zi = np.where(footIds == z)[0]
            yi = np.where(footIds == y)[0]
            img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                           (int(point[yi, 1]), int(point[yi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)
        if z in footIds and x in footIds:
            zi = np.where(footIds == z)[0]
            xi = np.where(footIds == x)[0]
            img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                           (int(point[xi, 1]), int(point[xi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)

    for i in range(point.shape[0]):
        x,y = point[i, 0],point[i, 1]
        if vert_color is None:
            v_color = rgb_code['Black']
        else:
            v_color = vert_color[i,::-1]
        img = cv2.circle(img, (int(y), int(x)), 40, (int(v_color[2]), int(v_color[1]), int(v_color[0])), -1)
        img = cv2.putText(img, f"{footIds[i]}", (int(y), int(x)+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (tex_color[2],tex_color[1],tex_color[0]))

    if save_name is not None:
        cv2.imwrite(osp.join(save_dir, save_name), img)
    else:
        return img

def getVertsColor(pressure_img,footIds,insole2smpl):
    verts_color = np.zeros([footIds.shape[0],3])
    for i in range(footIds.shape[0]):
        ids = footIds[i]
        if str(ids) in insole2smpl.keys():
            tmp = insole2smpl[str(ids)]
            _color = pressure_img[tmp[0],tmp[1],:]
            if _color.shape[0] != 0:
                verts_color[i] = np.sum(_color,axis=0)
    return verts_color.astype(np.int32)


if __name__ == '__main__':
    # mapInsole2Smpl()
    save_dir = 'debug/insole2smpl'
    insoleMaskR = np.loadtxt('E:/dataset/PressureDataset/insole_mask/insoleMaskR.txt').astype(np.int32)
    footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    model_temp = trimesh.load(osp.join(save_dir, 'v_template.obj'))
    v_template = np.array(model_temp.vertices)
    ic(footIdsR.shape, footIdsL.shape)
    v_footL = v_template[footIdsL, :]
    v_footR = v_template[footIdsR, :]
    # trimesh.Trimesh(vertices=v_template[footIdsR,:],process=False).export('debug/footR.obj')
    # trimesh.Trimesh(vertices=v_template[footIdsL,:],process=False).export('debug/footL.obj')
    foot_verts = np.concatenate([v_footL, v_footR], axis=0)
    verts_xz = foot_verts[:, [0, 2]]

    insole2smplR = np.load('essentials/pressure/insole2smplR.npy', allow_pickle=True).item()
    insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()

    m_data = PressureDataset(
        basdir='E:/dataset',
        dataset_name='PressureDataset',
        sub_ids='S12',
        seq_name='MoCap_20230422_145422',
    )
    for insole_ids in trange(1,145):
        # insole_ids = 24
        pressure_img = m_data.show_insole(insole_ids)
        pressure_img = cv2.applyColorMap(pressure_img, cv2.COLORMAP_HOT)
        # cv2.imwrite(osp.join(save_dir,'pressure%d.png'%insole_ids),pressure_img)

        vert_colorR = getVertsColor(pressure_img[:,], footIdsR, insole2smplR)
        imgR = drawSMPLFoot(v_footR,footIdsR,np.array(model_temp.faces),
                            vert_color=vert_colorR,save_name=None)

        vert_colorL = getVertsColor(pressure_img[:,], footIdsL, insole2smplL)
        imgL = drawSMPLFoot(v_footL,footIdsL,np.array(model_temp.faces),
                            vert_color=vert_colorL,save_name=None)
        img = np.concatenate([cv2.resize(pressure_img, (2200, 3350)),imgL,imgR],axis=1)
        cv2.imwrite('debug/insole2smpl/results/%03d.png'%insole_ids,img)

    saveImgSeqAsvideo('debug/insole2smpl/results', fps=10,ratio=10)




