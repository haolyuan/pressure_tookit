import sys
import cv2
import numpy as np
import trimesh
from tqdm import tqdm,trange
from icecream import ic

sys.path.append('E:/projects/pressure_toolkit')
from lib.dataset.PressureDataset import PressureDataset
from color_utils import rgb_code


if __name__ == '__main__':
    # m_data = PressureDataset(
    #     basdir='E:/dataset',
    #     dataset_name='PressureDataset',
    #     sub_ids='S12',
    #     seq_name='MoCap_20230422_145422',
    # )
    # for ids in trange(24,145):
    #     m_data.show_insole(ids)
    #     # frame_data = m_data.getFrameData(ids=ids)
    #     # ic(frame_data.keys())
    #     # ic(frame_data['insole'].shape)
    # exit()

    insoleMaskR = np.loadtxt('E:/dataset/PressureDataset/insole_mask/insoleMaskR.txt').astype(np.int32)
    footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    model_temp = trimesh.load('debug/v_template.obj')
    v_template = model_temp.vertices
    ic(footIdsR.shape,footIdsL.shape)
    v_footL = v_template[footIdsL,:]
    v_footR = v_template[footIdsR,:]
    # trimesh.Trimesh(vertices=v_template[footIdsR,:],process=False).export('debug/footR.obj')
    # trimesh.Trimesh(vertices=v_template[footIdsL,:],process=False).export('debug/footL.obj')
    foot_verts = np.concatenate([v_footL,v_footR],axis=0)
    ic(foot_verts.shape)
    verts_xz = foot_verts[:,[0,2]]
    ic(verts_xz.shape)
