import torch
import trimesh
import numpy as np
from icecream import ic

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.depthTerm import DepthTerm

class SMPLSolver():
    def __init__(self,
                 model_path: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 color_size=None,depth_size=None,
                 cIntr=None,dIntr=None,
                 device=None,
                 dtype=torch.float32):
        super(SMPLSolver, self).__init__()

        self.device = device
        self.dtype = dtype
        self.m_smpl = SMPLModel(
            model_path=model_path,
            num_betas=num_betas,
            gender=gender)
        self.m_smpl.to(device)

        self.depth_term = DepthTerm(dIntr,depth_size[0],depth_size[1])

    def initShape(self,depth2floor,depth_scan,keypoints):
        floor2depth = np.linalg.inv(depth2floor)

        #set smpl params
        betas = np.zeros([1,11])
        betas[:,0] = 0.78
        betas = torch.tensor(betas,dtype=self.dtype,device=self.device)
        transl = np.array([[0,0.97,0]])
        transl = torch.tensor(transl,dtype=self.dtype,device=self.device)
        body_pose = np.zeros([1,69])
        body_pose[:,47] = -0.6
        body_pose[:,50] = 0.6
        body_pose = torch.tensor(body_pose, dtype=self.dtype, device=self.device)
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose
        }
        self.m_smpl.setPose(**params_dict)
        self.m_smpl.updateShape()
        verts, J_transformed = self.m_smpl.updatePose()
        init_model = trimesh.Trimesh(vertices=verts.detach().cpu().numpy()[0],
                        faces=self.m_smpl.faces, process=False)

        self.depth_term.findLiveVisibileVerticesIndex(init_model,floor2depth)

        exit()



        vertices = self.m_smpl.v_template.detach().cpu().numpy()#[0]
        m_model = trimesh.Trimesh(vertices=vertices,faces=self.m_smpl.faces,process=False)
        m_model.apply_transform(floor2depth)

        self.depth_term.findLiveVisibileVerticesIndex(mesh=m_model)
        exit()
        ic(depth_scan.shape)
        depth_scan = torch.tensor(depth_scan)
        ic(self.m_smpl.betas ,self.m_smpl.body_pose ,self.m_smpl.transl ,self.m_smpl.global_orient)
