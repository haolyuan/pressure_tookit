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
from scipy.spatial.transform import Rotation as R

from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.depthTerm import DepthTerm
from lib.fitSMPL.colorTerm import ColorTerm
from lib.fitSMPL.pressureTerm import PressureTerm
from lib.fitSMPL.contactTerm import ContactTerm
from lib.Utils.fileio import saveJointsAsOBJ,saveProjectedJoints,saveCorrsAsOBJ

class SMPLSolver():
    def __init__(self,
                 model_path: str,
                 num_betas: int = 10,
                 gender: str = 'neutral',
                 color_size=None,depth_size=None,
                 cIntr=None,dIntr=None,
                 depth2floor=None,depth2color=None,
                 w_verts3d=10,w_joint2d=0.01,
                 w_betas=0.1,
                 w_penetrate=10,w_contact=0,
                 sub_ids=None,
                 seq_name=None,
                 device=None,
                 dtype=torch.float32):
        super(SMPLSolver, self).__init__()

        self.device = device
        self.dtype = dtype
        self.depth2floor = depth2floor
        self.floor2depth = np.linalg.inv(depth2floor)
        self.depth2color = depth2color

        vp, ps = load_model(expr_dir='D:/utils/Vposer_models/V02_05', model_code=VPoser,
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
        self.w_penetrate = w_penetrate
        self.w_contact = w_contact
        self.w_temp_pose = 100

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
        self.press_term = PressureTerm(device=self.device)
        self.contact_term = ContactTerm(device=self.device)

        footL_ids = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
        footR_ids = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
        self.foot_ids = torch.tensor(np.concatenate([footL_ids,footR_ids]),device=self.device).long()

        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids).to(device)
        self.init_lr = 0.1
        self.num_train_epochs = 50
        
        # set pre pose to improve local joints smooth
        # add neck, head
        self.pre_pose = None

        self.openposemap = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        self.halpemap = torch.tensor([0,18,6,8,10,5,7,9,19,12,14,16,11,13,15,2,1,4,3,20,22,24,21,23,25])
        
        self.results_dir = osp.join('debug', sub_ids, seq_name)
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
        transl = np.array([[-0.1, 0.97, -0.6]])
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
            self.adjust_learning_rate(rough_optimizer, iter)
            self.m_smpl.updateShape()
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, live_joints = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]
            
            joint_loss = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                        points=source_color_joints,
                                                        img=color_img)* self.w_joint2d
            betas_reg_loss = torch.square(self.m_smpl.betas).mean() * self.w_betas

            # if iter < max_iter//2:
            #     self.m_smpl.body_poseZ.requires_grad = True

            #     loss_geo = depth_loss + betas_reg_loss + joint_loss
            # else:
            #     penetrate_loss = torch.mean(torch.abs(live_verts[0,self.foot_ids,1]))*self.w_penetrate #penetration
            #     # self.m_smpl.body_poseZ.requires_grad = False
            #     loss_geo = depth_loss + betas_reg_loss + penetrate_loss
            penetrate_loss = torch.mean(torch.abs(live_verts[0,self.foot_ids,1]))*self.w_penetrate #penetration
            
            loss_geo = depth_loss + betas_reg_loss + joint_loss + penetrate_loss
            rough_optimizer.zero_grad()
            loss_geo.backward()
            rough_optimizer.step()

            if iter % 100 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)
                projected_joints = self.color_term.projectJoints(source_color_joints)
                projected_joints = projected_joints.detach().cpu().numpy()
                saveProjectedJoints(filename='debug/init_%04d.png'%iter,
                                    img=copy.deepcopy(color_img),
                                    joint_projected=projected_joints)
                saveProjectedJoints(filename='debug/target_%04d.png'%iter,
                                    img=copy.deepcopy(color_img),
                                    joint_projected=target_color_joints[:, :2])                

        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        annot['body_pose'] = body_pose_rec0.detach().cpu().numpy()
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        return annot

    def initPose(self,
                init_betas=None,
                depth_vmap=None,depth_nmap=None,
                color_img=None,keypoints=None,
                contact_data=None,
                max_iter=1000):
        #==========set smpl params====================
        betas = torch.tensor(init_betas,dtype=self.dtype,device=self.device)
        transl = np.array([[-0.1, 0.97, -0.6]])
        transl = torch.tensor(transl,dtype=self.dtype,device=self.device)
        body_pose = torch.zeros([1, 69], dtype=self.dtype, device=self.device)

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
        self.m_smpl.updateShape()
        # live_verts, J_transformed = self.m_smpl.updatePose()
        # trimesh.Trimesh(vertices=live_verts.detach().cpu().numpy()[0],
        #                 faces=self.m_smpl.faces, process=False).export('debug/live_mesh_vposer.obj')
        # exit()
        contact_ids, _, _ = self.contact_term.contact2smpl(np.array(contact_data))
        
        assert len(set(contact_ids)) == 98*2
        
        # ==========main loop====================
        rough_optimizer = torch.optim.Adam([self.m_smpl.global_orient,
                                            self.m_smpl.transl,
                                            self.m_smpl.body_poseZ], lr=0.01)

        for iter in trange(max_iter):
            self.adjust_learning_rate(rough_optimizer, iter)
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, live_joints = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]
            
            joint_loss = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                        points=source_color_joints,
                                                        img=color_img)* self.w_joint2d
            betas_reg_loss = torch.square(self.m_smpl.betas).mean() * self.w_betas

            # if iter < max_iter//2:
            #     self.m_smpl.body_poseZ.requires_grad = True

            #     loss_geo = depth_loss + betas_reg_loss + joint_loss
            # else:
            #     penetrate_loss = torch.mean(torch.abs(live_verts[0,self.foot_ids,1]))*self.w_penetrate #penetration
            #     # self.m_smpl.body_poseZ.requires_grad = False
            #     loss_geo = depth_loss + betas_reg_loss + penetrate_loss
            penetrate_loss = torch.mean(torch.abs(live_verts[0, contact_ids, 1]))*self.w_penetrate #penetration
            
            loss_geo = depth_loss + betas_reg_loss + joint_loss + penetrate_loss
            rough_optimizer.zero_grad()
            loss_geo.backward()
            rough_optimizer.step()

            if iter % 100 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)
                projected_joints = self.color_term.projectJoints(source_color_joints)
                projected_joints = projected_joints.detach().cpu().numpy()
                saveProjectedJoints(filename=f'{self.results_dir}/init_{iter:04d}.png',
                                    img=copy.deepcopy(color_img),
                                    joint_projected=projected_joints)
                saveProjectedJoints(filename=f'{self.results_dir}/target_{iter:04d}.png',
                                    img=copy.deepcopy(color_img),
                                    joint_projected=target_color_joints[:, :2])                

        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        # set wrist pose to zero. vposer cannot handle wrist rot
        amass_body_pose_rec0[:, (20-1)*3:(21-1)*3+3] = torch.zeros([1, 6], device=self.device)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        annot['body_pose'] = body_pose_rec0.detach().cpu().numpy()
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        return annot

    def setInitPose(self,init_params=None):
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
        amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        # live_verts, J_transformed = self.m_smpl.updatePose(body_pose=body_pose_rec)
        # self.press_term.setVertsPre(live_verts)
        self.pre_pose = body_pose_rec[:, 20*3: 21*3+3].detach()

    def modelTracking(self,
                      frame_ids=0,
                      depth_vmap=None,depth_nmap=None,
                      color_img=None,keypoints=None,
                      contact_data=None,
                      max_iter=1000):

        contact_ids, _, _ = self.contact_term.contact2smpl(np.array(contact_data))
        optimizer = torch.optim.Adam([self.m_smpl.global_orient, self.m_smpl.transl,
                                            self.m_smpl.body_poseZ], lr=self.init_lr)
        pbar = trange(max_iter)
        trimesh.Trimesh(vertices=depth_vmap,process=False).export(osp.join(self.results_dir,'frame%d_depth.obj'%frame_ids))
        loss_list = []
        for iter in pbar:
            self.adjust_learning_rate(optimizer, iter)
            pbar.set_description("Frame[%03d]:" % frame_ids)
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, live_joints = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(iter=iter,
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]

            joint_loss = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                       points=source_color_joints,
                                                       img=color_img)* self.w_joint2d

            
            temp_pose_loss = torch.mean(torch.abs(body_pose_rec[:, 20* 3: 21* 3+ 3]
                                               - self.pre_pose))* self.w_temp_pose
            
            loss_geo = depth_loss + joint_loss # + temp_pose_loss

            if contact_ids.shape[0]>0:
                penetrate_loss = torch.mean(torch.abs(live_verts[0, contact_ids, 1])) * self.w_penetrate  # penetration
                # cont_loss = self.press_term.calcContLoss(live_verts=live_verts,
                #                                          contact_ids=contact_ids) * self.w_contact
                loss_geo = loss_geo + penetrate_loss #+ cont_loss

                if iter == 0:
                    print(penetrate_loss,contact_ids.shape[0])
            elif iter==0:
                    print(depth_loss, joint_loss)

            optimizer.zero_grad()
            loss_geo.backward()
            optimizer.step()
            loss_list.append(loss_geo)

            self.w_temp_pose = body_pose_rec[:, 20* 3: 21* 3+ 3].detach()
        
        self.press_term.setVertsPre(live_verts)

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


        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        # set wrist pose to zero. vposer cannot handle wrist rot
        amass_body_pose_rec0[:, (20-1)*3:(21-1)*3+3] = torch.zeros([1, 6], device=self.device)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        body_pose = body_pose_rec0.detach().cpu()
        
        global_orient = self.m_smpl.global_orient.detach().cpu()
        
        pose = torch.concat([global_orient, body_pose], dim=1)
        
        output_pose = torch.zeros((pose.shape[0], 24, 3, 3), device=pose.device)
        for i in range(output_pose.shape[1]):
            output_pose[:, i] = torch.tensor(R.from_rotvec(pose.numpy()[:, i* 3: i* 3+ 3]).as_matrix(),
                                             device=pose.device)

        # annot = {}
        # annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        # annot['transl'] = self.m_smpl.transl.detach().cpu()
        # annot['betas'] = self.m_smpl.betas.detach().cpu()        
        # annot['pose'] = output_pose

        return self.m_smpl.transl.detach().cpu(), output_pose,\
            self.m_smpl.betas.detach().cpu() 
