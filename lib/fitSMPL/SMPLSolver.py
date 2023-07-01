import torch
import trimesh
import numpy as np
from tqdm import trange,tqdm
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from icecream import ic

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.depthTerm import DepthTerm
from lib.Utils.fileio import saveJointsAsOBJ

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

        vp, ps = load_model(expr_dir='E:/bodyModels/V02_05', model_code=VPoser,
                            remove_words_in_model_weights='vp_model.',
                            disable_grad=True)
        self.vp = vp.to(self.device)


        self.m_smpl = SMPLModel(
            model_path=model_path,
            num_betas=num_betas,
            gender=gender)
        self.m_smpl.to(device)


        self.depth_term = DepthTerm(dIntr,depth_size[0],depth_size[1],
                                    dtype=self.dtype,device=self.device)

    def initShape(self,
                  depth_vmap=None,depth_nmap=None,
                  depth2floor=None,depth2color=None,
                  color_img=None,keypoints=None,
                  max_iter=2000):

        floor2depth = np.linalg.inv(depth2floor)

        #==========set smpl params====================
        betas = np.zeros([1,11])
        betas[:,0] = 0.78
        betas = torch.tensor(betas,dtype=self.dtype,device=self.device)
        transl = np.array([[-0.1,0.97,-0.6]])
        transl = torch.tensor(transl,dtype=self.dtype,device=self.device)
        body_pose = np.zeros([1,69])
        body_pose[:,47] = -0.6
        body_pose[:,50] = 0.6
        body_pose = torch.tensor(body_pose, dtype=self.dtype, device=self.device)

        amass_body_poZ = self.vp.encode(body_pose[:,:-6]).mean
        amass_body_pose_rec = self.vp.decode(amass_body_poZ)['pose_body'].contiguous().view(-1, 63)
        body_pose = torch.cat([amass_body_pose_rec,torch.zeros([1,6],device=self.device)],dim=1)
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose,
            'body_poseZ':amass_body_poZ,
        }
        self.m_smpl.setPose(**params_dict)
        # self.m_smpl.updateShape()
        # live_verts, J_transformed = self.m_smpl.updatePose()
        # self.m_smpl.updateShape()
        # live_verts, J_transformed = self.m_smpl.updatePose()
        #
        # trimesh.Trimesh(vertices=live_verts.detach().cpu().numpy()[0],
        #                 faces=self.m_smpl.faces, process=False).export('debug/live_mesh_vposer.obj')
        # exit()
        self.m_smpl.betas.requires_grad = True
        # ==========main loop====================
        rough_optimizer = torch.optim.Adam([self.m_smpl.global_orient, self.m_smpl.transl, self.m_smpl.betas,
                                            self.m_smpl.body_poseZ], lr=0.01)
        w_verts3d = 1
        w_betas = 0.01
        for iter in trange(max_iter):
            # if iter==600:
            #     for param_group in rough_optimizer.param_groups:
            #         param_group['lr'] = 0.001
            #     w_lms2d = 8e-2
            # elif iter==1500:
            #     for param_group in rough_optimizer.param_groups:
            #         param_group['lr'] = 0.0001
            #     w_lms2d =1e-1
            self.m_smpl.updateShape()
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, J_transformed = self.m_smpl.updatePose(body_pose=body_pose_rec)
            depth_loss = self.depth_term.calcDepthLoss(
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces,
                floor2depth=floor2depth)*w_verts3d
            betas_reg_loss = torch.square(self.m_smpl.betas).mean() * w_betas
            # lms2d_loss = self.calcLmks2dLoss_colormap(lmsDet2d, cIntr, Td2c, w_lms2d)
            loss_geo = depth_loss + betas_reg_loss
            rough_optimizer.zero_grad()
            loss_geo.backward()
            rough_optimizer.step()

            if iter % 100 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)


        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        annot['body_pose'] = self.m_smpl.body_pose.detach().cpu().numpy()
        np.save('debug/init_param.npy',annot)