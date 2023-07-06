import numpy as np
import torch
import trimesh
from icecream import ic

from lib.fitSMPL.Camera import RGBDCamera
from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.colorTerm import ColorTerm
from lib.Utils.fileio import saveJointsAsOBJ

if __name__ == '__main__':
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
