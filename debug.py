import numpy as np
import torch
import trimesh
import os
import math
import cv2,pickle
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

if __name__ == '__main__':
    template_model = trimesh.load('essentials/smpl_uv/smpl_template.obj')
    verts = template_model.vertices
    normals = template_model.vertex_normals
    # saveNormalsAsOBJ('debug/smpl_normal.obj',verts,normals)
    valid_ids = np.where((verts[:,1]<-1.097)&(verts[:,0]>0))[0]
    verts_select = verts[valid_ids,:]
    ic(verts_select.shape)
    trimesh.Trimesh(vertices=verts_select,process=False).export('debug/footL.obj')
    foot_list=[]
    for i in valid_ids:
        cosine = np.dot(normals[i,:], np.array([0, -1, 0]))
        # ic(cosine)
        if cosine > math.cos(20):
            foot_list.append(i)
    ic(len(foot_list))
    trimesh.Trimesh(vertices=verts[foot_list,:], process=False).export('debug/footL_refine.obj')
    np.savetxt('debug/footL_ids.txt',foot_list,fmt='%d')
    exit()

    exit()
    m_pt = PressureTerm()
    m_pt.insole2smpl()
    exit()
    params = np.load('debug/init_param100.npy',allow_pickle=True).item()
    ic(params)

    m_smpl = SMPLModel(model_path='E:/bodyModels/smpl',
        num_betas=10, gender='male')

    params_dict = {
        'betas': torch.tensor(params['betas']),
        'global_orient':torch.tensor(params['global_orient']),
        'transl': torch.tensor(params['transl']),
        'body_pose': torch.tensor(params['body_pose']),
        'body_poseZ': torch.tensor(params['body_poseZ']),
    }
    m_smpl.setPose(**params_dict)
    m_smpl.updateShape()
    live_verts, J_transformed = m_smpl.updatePose()
    j_posed = m_smpl.vertices2joints(m_smpl.J_regressor, live_verts)
    saveJointsAsOBJ('debug/init_joints.obj', j_posed.detach().cpu().numpy()[0], m_smpl.parents)
    np.save('debug/init_joints.npy',j_posed.detach().cpu().numpy()[0])
    verts = live_verts.detach().cpu().numpy()[0]
    init_model = trimesh.Trimesh(vertices=verts,faces=m_smpl.faces,process=False)#.export('debug/init_model.obj')
