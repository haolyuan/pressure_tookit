import os

import torch
import trimesh
import numpy as np
import copy
import os.path as osp
from tqdm import trange,tqdm
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from smplx.vertex_joint_selector import VertexJointSelector
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from icecream import ic

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.depthTerm import DepthTerm
from lib.fitSMPL.colorTerm import ColorTerm
from lib.Utils.fileio import saveJointsAsOBJ,saveProjectedJoints,saveCorrsAsOBJ

class SMPLSolver():
    def __init__(self,
                 model_path: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 color_size=None,depth_size=None,
                 cIntr=None,dIntr=None,
                 depth2floor=None,depth2color=None,
                 w_verts3d=10,
                 w_betas=0.1,
                 w_joint2d=0.01,
                 seq_name='debug',
                 device=None,
                 dtype=torch.float32):
        super(SMPLSolver, self).__init__()

        self.device = device
        self.dtype = dtype
        self.depth2floor = depth2floor
        self.floor2depth = np.linalg.inv(depth2floor)
        self.depth2color = depth2color

        vp, ps = load_model(expr_dir='E:/bodyModels/V02_05', model_code=VPoser,
                            remove_words_in_model_weights='vp_model.',
                            disable_grad=True)
        self.vp = vp.to(self.device)


        self.m_smpl = SMPLModel(
            model_path=model_path,
            num_betas=num_betas,
            gender=gender)
        self.m_smpl.to(device)

        self.w_verts3d = w_verts3d
        self.w_betas = w_betas
        self.w_joint2d = w_joint2d

        self.depth_term = DepthTerm(cam_intr=dIntr,
                                    img_W=depth_size[0],
                                    img_H=color_size[1],
                                    depth2color=self.depth2color,
                                    depth2floor=self.depth2floor,
                                    dtype=self.dtype,device=self.device)
        self.color_term = ColorTerm(cam_intr=cIntr,
                                    img_W=color_size[0],
                                    img_H=color_size[1],
                                    depth2color=self.depth2color,
                                    depth2floor=self.depth2floor,
                                    dtype=self.dtype,device=self.device)

        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids).to(device)
        self.init_lr = 0.1
        self.num_train_epochs = 150

        self.results_dir = osp.join('debug',seq_name)
        if not osp.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by x every y epochs
        x = 0.1, y = args.num_train_epochs = 100
        """
        lr = self.init_lr * (0.1 ** (epoch // self.num_train_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def initShape(self,
                  depth_vmap=None,depth_nmap=None,
                  color_img=None,keypoints=None,
                  max_iter=1000):
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
        # trimesh.Trimesh(vertices=live_verts.detach().cpu().numpy()[0],
        #                 faces=self.m_smpl.faces, process=False).export('debug/live_mesh_vposer.obj')
        # exit()
        self.m_smpl.betas.requires_grad = True
        # ==========main loop====================
        rough_optimizer = torch.optim.Adam([self.m_smpl.global_orient, self.m_smpl.transl, self.m_smpl.betas,
                                            self.m_smpl.body_poseZ], lr=0.01)

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
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            live_joints = self.m_smpl.vertices2joints(self.m_smpl.J_regressor,live_verts)
            joints = self.vertex_joint_selector(live_verts, live_joints)[0]
            joint_loss = self.color_term.calcColorLoss(keypoints=keypoints, points=joints,img=color_img)*self.w_joint2d

            betas_reg_loss = torch.square(self.m_smpl.betas).mean() * self.w_betas

            loss_geo = depth_loss + betas_reg_loss + joint_loss
            rough_optimizer.zero_grad()
            loss_geo.backward()
            rough_optimizer.step()

            if iter % 100 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)
                projected_joints = self.color_term.projectJoints(joints)
                projected_joints = projected_joints.detach().cpu().numpy()
                saveProjectedJoints(filename='debug/init_%04d.png'%iter,
                                    img=copy.deepcopy(color_img),
                                    joint_projected=projected_joints)

        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        annot['body_pose'] = body_pose_rec0.detach().cpu().numpy()
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        np.save('debug/init_param%d.npy'%iter,annot)



    def modelTracking(self,
                      init_params=None,frame_ids=0,
                      depth_vmap=None,depth_nmap=None,
                      color_img=None,keypoints=None,
                      max_iter=1000):
        #==========set smpl params====================
        betas = torch.tensor(init_params['betas'],dtype=self.dtype,device=self.device)
        transl = torch.tensor(init_params['transl'],dtype=self.dtype,device=self.device)
        body_pose = torch.tensor(init_params['body_pose'], dtype=self.dtype, device=self.device)
        body_poseZ = torch.tensor(init_params['body_poseZ'], dtype=self.dtype, device=self.device)
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose,
            'body_poseZ':body_poseZ,
        }
        self.m_smpl.setPose(**params_dict)
        self.m_smpl.updateShape()
        # ==========main loop====================
        optimizer = torch.optim.Adam([self.m_smpl.global_orient, self.m_smpl.transl,
                                            self.m_smpl.body_poseZ], lr=self.init_lr)
        pbar = trange(max_iter)
        trimesh.Trimesh(vertices=depth_vmap,process=False).export(osp.join(self.results_dir,'frame%d_depth.obj'%frame_ids))
        for iter in pbar:
            self.adjust_learning_rate(optimizer, iter)
            # elif iter==1500:
            #     for param_group in rough_optimizer.param_groups:
            #         param_group['lr'] = 0.0001
            #     w_lms2d =1e-1
            pbar.set_description("Frame[%03d]:" % frame_ids)
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, J_transformed = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(iter=iter,
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            live_joints = self.m_smpl.vertices2joints(self.m_smpl.J_regressor,live_verts)
            joints = self.vertex_joint_selector(live_verts, live_joints)[0]
            joint_loss = self.color_term.calcColorLoss(keypoints=keypoints, points=joints,img=color_img)*self.w_joint2d

            loss_geo = depth_loss + joint_loss
            # loss_geo = joint_loss
            optimizer.zero_grad()
            loss_geo.backward()
            optimizer.step()


            # if iter % 100 == 0 and iter>0:
        _verts = live_verts.detach().cpu().numpy()[0]
        output_mesh = trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False)
        output_mesh.export(osp.join(self.results_dir,'frame%04d_%04d.obj'%(frame_ids,iter)))
        color_render, depth = self.color_term.renderMesh(output_mesh,color_img)
        # projected_joints = self.color_term.projectJoints(joints)
        # projected_joints = projected_joints.detach().cpu().numpy()
        saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),
                            img=color_render,
                            joint_projected=keypoints[:,:2])


        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        annot['body_pose'] = body_pose_rec0.detach().cpu().numpy()
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        np.save(osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),annot)
