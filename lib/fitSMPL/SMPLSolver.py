import os

import torch
import trimesh
import numpy as np
import copy
import os.path as osp
import pickle as pkl
from tqdm import trange,tqdm
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from smplx.vertex_joint_selector import VertexJointSelector
from smplx.vertex_ids import vertex_ids as VERTEX_IDS
from icecream import ic
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


from lib.fitSMPL.SMPLModel import SMPLModel
from lib.fitSMPL.depthTerm import DepthTerm
from lib.fitSMPL.colorTerm import ColorTerm
from lib.fitSMPL.pressureTerm import PressureTerm
from lib.fitSMPL.contactTerm import ContactTerm
from lib.fitSMPL.gmmTerm import MaxMixturePriorLoss
from lib.Utils.fileio import saveJointsAsOBJ,saveProjectedJoints,saveCorrsAsOBJ
from lib.Utils.measurements import MeasurementsLoss
from lib.Utils.refineSMPL_utils import compute_normal_batch

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


        gmm_term = MaxMixturePriorLoss(prior_folder='essentials/smplify_essential')
        self.gmm_term = gmm_term.to(self.device)
        
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
        self.w_temp_foot = 10
        self.norm_weight = 10

        self.depth_term = DepthTerm(cam_intr=dIntr,
                                    img_W=depth_size[0],
                                    img_H=depth_size[1],
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

        self.w_h_term = MeasurementsLoss(self.m_smpl.faces)
        
        footL_ids = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
        footR_ids = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
        self.foot_ids = torch.tensor(np.concatenate([footL_ids,footR_ids]),device=self.device).long()

        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids).to(device)
        self.init_lr = 0.1
        self.num_train_epochs = 30
        
        # set pre pose to improve local joints smooth
        # add neck, head
        self.pre_pose = None

        self.openposemap = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        self.halpemap = torch.tensor([0,18,6,8,10,5,7,9,19,12,14,16,11,13,15,2,1,4,3,20,22,24,21,23,25])
        
        self.sub_ids = sub_ids
        self.seq_name = seq_name
        self.results_dir = osp.join('debug', sub_ids, seq_name)
        self.results_pose_dir = osp.join('debug/frame_debug', sub_ids, seq_name)
        self.pose_debug_dir = osp.join(self.results_pose_dir, 'single_frame')
        if not osp.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not osp.exists(self.results_pose_dir):
            os.makedirs(self.results_pose_dir)
            os.makedirs(self.pose_debug_dir)
            

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
                  weight_data=None,
                  frame_idx=0,
                  max_iter=1000):
        # depth_color = self.color_term.projectJoints(torch.tensor(depth_vmap, device=self.device, dtype=self.dtype))
        # os.makedirs(f'{self.results_dir}/gt_visual', exist_ok=True)
        # saveProjectedJoints(filename=f'{self.results_dir}/gt_visual/{frame_idx}.png',
        #                     img=color_img.copy(),
        #                     joint_projected=depth_color[:,:2])
        
        # import pdb;pdb.set_trace()

        #==========set smpl params====================
        betas = np.zeros([1,11])
        betas[:, 0] = 1 #
        betas = torch.tensor(betas,dtype=self.dtype,device=self.device)
        init_betas = betas.detach().clone()
        
        # transl = np.array([[-0.1, 0.97, -0.6]])
        # calculate coarse position to init
        aver_x = np.average(np.array(depth_vmap)[:, 0])
        aver_z = np.average(np.array(depth_vmap)[:, 2] - 0.1)
        transl = np.array([[aver_x, 0.97, aver_z]])
        transl = torch.tensor(transl,dtype=self.dtype,device=self.device)
        body_pose = np.zeros([1,69])
        body_pose[:,47] = -1.2
        body_pose[:,50] = 1.2
        body_pose = torch.tensor(body_pose, dtype=self.dtype, device=self.device)

        amass_body_poZ = self.vp.encode(body_pose[:,:-6]).mean
        amass_body_pose_rec = self.vp.decode(amass_body_poZ)['pose_body'].contiguous().view(-1, 63)
        # vposer初始脚有一个x方向的旋转
        # body_pose_rec = torch.cat([amass_body_pose_rec,torch.zeros([1,6],device=self.device)],dim=1)
        
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose,
            'body_poseZ':amass_body_poZ,
        }
        self.m_smpl.setPose(**params_dict)
        self.m_smpl.updateShape()
        self.m_smpl.initPlane()     

        # `````````````````init vposer value``````````````````
        T_verts, _, _,  T_plane = self.m_smpl.updatePose(body_pose=body_pose)
        trimesh.Trimesh(vertices=T_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).export('debug/t_smpl_aa.obj')
        # trimesh.Trimesh(vertices=T_plane[0].detach().cpu().numpy()).export('debug/t_plane.obj')

        # import pdb;pdb.set_trace()
        target_vert_T = T_verts.clone().detach()
        # ``````````````` only init vposer value````````````````
        self.m_smpl.transl.requires_grad = False
        self.m_smpl.global_orient.requires_grad = False
        # first use vposer to get init correct vposer value
        rough_optimizer_vposer = torch.optim.Adam([self.m_smpl.body_poseZ], lr=0.01)
        for iter in trange(max_iter):
            self.adjust_learning_rate(rough_optimizer_vposer, iter)
            self.m_smpl.updateShape()
            self.m_smpl.initPlane()       
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, _, _, _ = self.m_smpl.updatePose(body_pose=body_pose_rec)      
            
            verts3d_loss = torch.mean(torch.norm(target_vert_T - live_verts))  
            rough_optimizer_vposer.zero_grad()
            verts3d_loss.backward()
            rough_optimizer_vposer.step()    
            
        amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        live_verts, live_joints, live_joints_smpl, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)        
        trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).export('debug/smpl_T_optim_vposer.obj')
        # import pdb;pdb.set_trace()
        
        # ```````````````` shape optimization``````````````````
        self.m_smpl.transl.requires_grad = True
        self.m_smpl.global_orient.requires_grad = True       
        self.m_smpl.betas.requires_grad = True
        self.m_smpl.model_scale_opt.requires_grad = True
        # self.m_smpl.body_poseZ.requires_grad = False
        
        # ==========main loop====================
        rough_optimizer_shape = torch.optim.Adam([self.m_smpl.global_orient, self.m_smpl.transl,
                                              self.m_smpl.body_poseZ,
                                              self.m_smpl.model_scale_opt,self.m_smpl.betas
                                              ], lr=0.01) #
        for iter in trange(max_iter):
            self.adjust_learning_rate(rough_optimizer_shape, iter)
            self.m_smpl.updateShape()
            self.m_smpl.initPlane()
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, live_joints, live_joints_smpl, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)
            # trimesh.Trimesh(vertices=live_joints_smpl[0][:45, :].detach().cpu().numpy()).export('debug/smpl.obj')
            # trimesh.Trimesh(vertices=live_joints_smpl[0][45:, :].detach().cpu().numpy()).export('debug/smpl_extra.obj')
            
            depth_loss = self.depth_term.calcDepthLoss(
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints_3d = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]
            
            joint_loss, source_color_joints = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                        points=source_color_joints_3d,
                                                        img=color_img)
            joint_loss *= self.w_joint2d
            betas_reg_loss = torch.abs(self.m_smpl.betas- init_betas).mean() * self.w_betas# 

            # contact_data = [[1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1]]
            # contact_ids, _, _ = self.contact_term.contact2smpl(contact_data=np.array(contact_data))
            # penetrate_loss = self.contact_term.calcPenetrateLoss(
            #         live_verts=live_verts, contact_ids=contact_ids) * self.w_penetrate
            
            # penetrate_loss_1 = (torch.mean(torch.abs(live_plane[:, :live_plane.shape[1]//2, 1])) +\
            #     torch.mean(torch.abs(live_plane[:, live_plane.shape[1]//2:, 1]))) * self.w_penetrate
            plane_faces_list = [self.m_smpl.back_L_faces, self.m_smpl.back_R_faces, self.m_smpl.front_L_faces, self.m_smpl.front_R_faces]
            target_n_list = np.array([[0,1.0,0], [0,1.0,0], [0,1.0,0], [0,1.0,0]])
            norm_loss = self.contact_term.calcNormLoss(source_verts=live_plane,
                                                       verts_faces_list=plane_faces_list,
                                                       target_n_list=torch.tensor(target_n_list)) * self.norm_weight
            
            penetrate_loss = torch.mean(torch.abs(live_plane[:, :30, 1]) +\
                torch.abs(live_plane[:, 30+42:30*2+42, 1] ))* self.w_penetrate  +\
                torch.mean(torch.abs(live_plane[:, 30:30+42, 1] ) +\
                torch.abs(live_plane[:, 30*2+42:, 1] ))* self.w_penetrate                              
            
            # fix lower body x rotation. Hip-x, knee-x, ankle-x, head-xyz, should-xy, root-x
            amass_body_pose_rec_temp = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            global_rot = self.m_smpl.global_orient
            body_loss =torch.abs(global_rot[:, 0]) +\
                torch.abs(amass_body_pose_rec_temp[:, 0*3]) + torch.abs(amass_body_pose_rec_temp[: ,1*3]) +\
                torch.abs(amass_body_pose_rec_temp[:, 3*3]) + torch.abs(amass_body_pose_rec_temp[: ,4*3]) +\
                torch.abs(amass_body_pose_rec_temp[:, 6*3]) + torch.abs(amass_body_pose_rec_temp[: , 7*3]) +\
                torch.sum(torch.abs(amass_body_pose_rec_temp[:, 14*3: 15*3])) #+\
                    
                # torch.sum(torch.abs(amass_body_pose_rec_temp[:, 15*3: 15*3+ 2])) +\
                # torch.sum(torch.abs(amass_body_pose_rec_temp[:, 16*3: 16*3+ 2])) +\
        
            # height loss
            # target_weight = torch.tensor(weight_data, device=self.device, dtype=self.dtype)/ 9.8# N to kg
            target_height = torch.max(torch.tensor(depth_vmap[:, 1]) + 0.03)
            # source_w_h = self.w_h_term.forward(v=live_verts)
            # source_weight = source_w_h['mass']  
            # source_height = source_w_h['height']
            # weight_loss = torch.abs(source_weight - target_weight)
            source_height = torch.max(live_verts[:, :, 1])
            height_loss = torch.abs(source_height - target_height)
            
            loss_geo =  depth_loss + joint_loss + penetrate_loss +\
                height_loss * 10 +\
                + body_loss * 5 #+\
                # norm_loss  # +\
                #  + betas_reg_loss # + weight_loss # 
                
            
            # loss_geo = joint_loss
                
            rough_optimizer_shape.zero_grad()
            loss_geo.backward()
            rough_optimizer_shape.step()

            if iter % 50 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)
                trimesh.Trimesh(vertices=live_joints.detach().cpu().numpy()[0],process=False)\
                    .export('debug/init_%04d_jts_shape.obj'%iter)

                projected_joints = self.color_term.projectJoints(source_color_joints_3d)
                projected_joints = projected_joints.detach().cpu().numpy()
                output_mesh = trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False)
                color_render, depth = self.color_term.renderMesh(output_mesh,color_img)
                
                saveProjectedJoints(filename='debug/init_%04d.png'%iter,
                                    img=copy.deepcopy(color_render),
                                    joint_projected=projected_joints)
                saveProjectedJoints(filename='debug/target_%04d.png'%iter,
                                    img=copy.deepcopy(color_img),
                                    joint_projected=target_color_joints[:, :2])     
                # trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).export('debug/live_plane.obj')   
 

        annot = {}
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu().numpy()
        annot['transl'] = self.m_smpl.transl.detach().cpu().numpy()
        annot['betas'] = self.m_smpl.betas.detach().cpu().numpy()
        amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        annot['body_pose'] = body_pose_rec0.detach().cpu().numpy()
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu().numpy()
        annot['model_scale_opt'] = self.m_smpl.model_scale_opt.detach().cpu().numpy()
        import pdb;pdb.set_trace()
        return annot

    def initPose(self,
                init_shape=None,
                depth_vmap=None,depth_nmap=None,
                color_img=None,keypoints=None,
                contact_data=None,cliff_pose=None,
                max_iter=1000):
        #==========set smpl params====================
        betas = torch.tensor(init_shape[0],dtype=self.dtype,device=self.device)
        model_scale_opt = torch.tensor(init_shape[1],dtype=self.dtype,device=self.device)
        # calculate coarse position to init
        aver_x = np.average(np.array(depth_vmap)[:, 0])
        aver_z = np.average(np.array(depth_vmap)[:, 2] - 0.1)
        transl = np.array([[aver_x, 0.97, aver_z]])
        # transl = np.array([[-0.8, 0.97, -2]])
        
        transl = torch.tensor(transl,dtype=self.dtype,device=self.device)
        # body_pose = torch.zeros([1, 69], dtype=self.dtype, device=self.device)
        body_pose_cliff = torch.tensor(cliff_pose[:, 3:], dtype=self.dtype, device=self.device)

        amass_body_poZ = self.vp.encode(body_pose_cliff[:,:-6]).mean
        amass_body_pose_rec = self.vp.decode(amass_body_poZ)['pose_body'].contiguous().view(-1, 63)
        body_pose = torch.cat([amass_body_pose_rec,torch.zeros([1,6],device=self.device)],dim=1)
        
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose,
            'body_poseZ':amass_body_poZ,
            'model_scale_opt':model_scale_opt,
        }
        self.m_smpl.setPose(**params_dict)
        self.m_smpl.updateShape()
        self.m_smpl.initPlane()
        
        T_verts, _, _,  T_plane = self.m_smpl.updatePose(body_pose=body_pose_cliff)
        trimesh.Trimesh(vertices=T_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).export('debug/t_smpl.obj')
        trimesh.Trimesh(vertices=T_plane[0].detach().cpu().numpy()).export('debug/t_plane.obj')
        target_vert_T = T_verts.clone().detach()
        
        # ========== before main loop, init smpl vposer====================
        rough_optimizer_vposer = torch.optim.Adam([self.m_smpl.body_poseZ], lr=0.01)
        for iter in trange(max_iter):
            self.adjust_learning_rate(rough_optimizer_vposer, iter)
            self.m_smpl.updateShape()
            self.m_smpl.initPlane()       
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, _, _, _ = self.m_smpl.updatePose(body_pose=body_pose_rec)      
            
            verts3d_loss = torch.mean(torch.norm(target_vert_T - live_verts))  
            rough_optimizer_vposer.zero_grad()
            verts3d_loss.backward()
            rough_optimizer_vposer.step()    
            
        amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        live_verts, live_joints, live_joints_smpl, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)        
        trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).export('debug/smpl_T_optim_vposer.obj')
        
        
        contact_ids, _, _ = self.contact_term.contact2smpl(np.array(contact_data))
        assert len(set(contact_ids)) == 96*2
        
        # ==========main loop====================
        rough_optimizer_pose = torch.optim.Adam([self.m_smpl.global_orient,
                                            self.m_smpl.transl,
                                            self.m_smpl.body_poseZ], lr=0.01)

        for iter in trange(max_iter):
            self.adjust_learning_rate(rough_optimizer_pose, iter)
            amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            live_verts, live_joints, live_joints_smpl, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints_3d = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]
            joint_loss, source_color_joints = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                        points=source_color_joints_3d,
                                                        img=color_img)
            joint_loss *= self.w_joint2d
            betas_reg_loss = torch.square(self.m_smpl.betas).mean() * self.w_betas

            # if iter < max_iter//2:
            #     self.m_smpl.body_poseZ.requires_grad = True

            #     loss_geo = depth_loss + betas_reg_loss + joint_loss
            # else:
            #     penetrate_loss = torch.mean(torch.abs(live_verts[0,self.foot_ids,1]))*self.w_penetrate #penetration
            #     # self.m_smpl.body_poseZ.requires_grad = False
            #     loss_geo = depth_loss + betas_reg_loss + penetrate_loss
            # penetrate_loss = torch.mean(torch.abs(live_verts[0, contact_ids, 1]))*self.w_penetrate #penetration
            
            penetrate_loss = (torch.mean(torch.abs(live_plane[:, :live_plane.shape[1]//2, 1])) +\
                torch.mean(torch.abs(live_plane[:, live_plane.shape[1]//2:, 1])))* self.w_penetrate
            
            # body norm not correct
            # calculate norm direction in xoz
            source_v4norm = live_joints_smpl[:, 1:4, :]
            source_face_norm = F.normalize(
                torch.cross(source_v4norm[:, 0, :] - source_v4norm[:, 2, :],
                            source_v4norm[:, 0, :] - source_v4norm[:, 1, :]),
                dim=-1
            )
            source_face_norm_x, source_face_norm_z =\
                source_face_norm[:, 0], source_face_norm[:, 2]
            
            source_face_norm_xoz = F.normalize(torch.concat([source_face_norm_x, source_face_norm_z]), dim=0)
            
            target_v4norm_idx = np.where(depth_vmap[:, 1]<0.001)
            target_v4norm = depth_vmap[target_v4norm_idx, :]
            
            target_face_norm_x, target_face_norm_z =\
                -1 * torch.tensor(np.average(target_v4norm[:, :, 0]), device=self.device, dtype=self.dtype).unsqueeze(0),\
                -1 * torch.tensor(np.average(target_v4norm[:, :, 2]), device=self.device, dtype=self.dtype).unsqueeze(0)
                
            target_face_norm_xoz = F.normalize(torch.concat([target_face_norm_x, target_face_norm_z]), dim=0)
            body_face_loss = 1 - torch.dot(source_face_norm_xoz, target_face_norm_xoz)
            
            loss_geo = depth_loss + betas_reg_loss + joint_loss + penetrate_loss #+ body_face_loss *10
            rough_optimizer_pose.zero_grad()
            loss_geo.backward()
            rough_optimizer_pose.step()

            if iter % 50 == 0:
                _verts = live_verts.detach().cpu().numpy()[0]
                trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False).export('debug/init_%04d.obj'%iter)
                trimesh.Trimesh(vertices=live_joints[0].detach().cpu().numpy(),process=False).export('debug/init_jts_%04d.obj'%iter)
                
                projected_joints = self.color_term.projectJoints(source_color_joints_3d)
                projected_joints = projected_joints.detach().cpu().numpy()
                saveProjectedJoints(filename=f'{self.results_dir}/init_{iter:04d}.png',
                                    img=copy.deepcopy(color_img),
                                    joint_projected=projected_joints)
                saveProjectedJoints(filename=f'{self.results_dir}/target_{iter:04d}.png',
                                    img=copy.deepcopy(color_img),
                                    joint_projected=target_color_joints[:, :2])      
        trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).export(f'{self.results_dir}/result_plane.obj')          

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
        annot['model_scale_opt'] = self.m_smpl.model_scale_opt.detach().cpu().numpy()
        
        return annot

    def setInitPose(self,init_params=None):
        betas = torch.tensor(init_params['betas'],dtype=self.dtype,device=self.device)
        transl = torch.tensor(init_params['transl'],dtype=self.dtype,device=self.device)
        body_pose = torch.tensor(init_params['body_pose'], dtype=self.dtype, device=self.device)
        body_poseZ = torch.tensor(init_params['body_poseZ'], dtype=self.dtype, device=self.device)
        global_rot = torch.tensor(init_params['global_orient'], dtype=self.dtype, device=self.device)
        model_scale_opt = torch.tensor(init_params['model_scale_opt'], dtype=self.dtype, device=self.device)
        params_dict = {
            'betas':betas,
            'transl':transl,
            'body_pose':body_pose,
            'body_poseZ':body_poseZ,
            'global_orient':global_rot,
            'model_scale_opt':model_scale_opt
        }
        self.m_smpl.setPose(**params_dict)
        self.m_smpl.updateShape()
        
        # update plane with init shape
        self.m_smpl.initPlane()
        
        # amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        # body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        # live_verts, J_transformed = self.m_smpl.updatePose(body_pose=body_pose_rec)
        # self.press_term.setVertsPre(live_verts)
        
        self.pre_pose = body_pose.detach()

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
            live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)

            depth_loss = self.depth_term.calcDepthLoss(iter=iter,
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d

            source_color_joints_3d = live_joints[:, self.openposemap, :].squeeze(0)
            target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]

            joint_loss, source_color_joints = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                       points=source_color_joints_3d,
                                                       img=color_img)
            joint_loss *= self.w_joint2d
            
            # plane loss to fix foot
            if iter == 0:
                foot_temp_loss = torch.zeros(1, device=self.device, dtype=self.dtype)*self.w_temp_foot
                
                # trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
                #     export(f'{self.results_dir}/iter_smpl_0.obj')
                # trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).\
                #     export(f'{self.results_dir}/iter_plane_0.obj')
                    
                self.contact_term.update_foot_plane(
                    foot_plane=live_plane,
                    contact_data=contact_data,
                    foot_plane_ids_smplL=[self.m_smpl.foot_ids_back_smplL, self.m_smpl.foot_ids_front_smplL],
                    foot_plane_ids_smplR=[self.m_smpl.foot_ids_back_smplR, self.m_smpl.foot_ids_front_smplR])
                
                # depth_color = self.color_term.projectJoints(torch.tensor(depth_vmap, device=self.device, dtype=self.dtype))
                # saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_depth_in_color.png'%(frame_ids)),
                #                     img=color_img.copy(),
                #                     joint_projected=depth_color[:,:2])                
            else:
                foot_temp_loss = self.contact_term.calcTempLoss(
                    live_plane=live_plane,
                    contact_data=contact_data,
                    foot_plane_ids_smplL=[self.m_smpl.foot_ids_back_smplL, self.m_smpl.foot_ids_front_smplL],
                    foot_plane_ids_smplR=[self.m_smpl.foot_ids_back_smplR, self.m_smpl.foot_ids_front_smplR])*self.w_temp_foot
            
            
            loss_geo = depth_loss  + foot_temp_loss# + temp_pose_loss+ joint_loss
            
            if contact_ids.shape[0]>0:
                # penetration
                penetrate_loss = self.contact_term.calcPenetrateLoss(
                    live_verts=live_verts, contact_ids=contact_ids) * self.w_penetrate
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

            
        # trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
        #     export(f'{self.results_dir}/iter_smpl_final.obj')
        # trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).\
        #     export(f'{self.results_dir}/iter_plane_final.obj')

        self.press_term.setVertsPre(live_verts)# useless
        


        amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)
        _verts = live_verts.detach().cpu().numpy()[0]
        output_mesh = trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False)
        output_mesh.export(osp.join(self.results_dir,'frame%04d_%04d.obj'%(frame_ids,iter)))
        color_render, depth = self.color_term.renderMesh(output_mesh,color_img)
        # projected_joints = self.color_term.projectJoints(joints)
        # projected_joints = projected_joints.detach().cpu().numpy()
        saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),
                            img=color_render,
                            joint_projected=keypoints[:,:2])
        # saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),
        #                     img=color_render,
        #                     joint_projected=keypoints[:,:2])

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

        annot = {}
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu()
        annot['transl'] = self.m_smpl.transl.detach().cpu()
        annot['betas'] = self.m_smpl.betas.detach().cpu()        
        annot['body_pose'] = body_pose
        annot['model_scale_opt'] = self.m_smpl.model_scale_opt.detach().cpu()  
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu()  
        torch.save(annot, f'{self.results_pose_dir}/{frame_ids}.pth')
        
        
        return self.m_smpl.transl.detach().cpu(), output_pose,\
            self.m_smpl.betas.detach().cpu() 

    def modelTracking_single_frame(self,
                      frame_ids=0,
                      depth_vmap=None,depth_nmap=None,
                      color_img=None,keypoints=None,
                      contact_data=None,
                      max_iter=1000):

        # contact gt
        contact_ids, _, _ = self.contact_term.contact2smpl(np.array(contact_data))
        # 2d gt
        target_color_joints = torch.tensor(keypoints, device=self.device)[self.halpemap, :]
        target_color_joints_conf = target_color_joints[:, 2]


        trimesh.Trimesh(vertices=depth_vmap,process=False).export(osp.join(self.results_dir,'frame%d_depth.obj'%frame_ids))
        
        # stage 0: optim vposer
        self.m_smpl.global_orient.requires_grad = False
        self.m_smpl.body_pose.requires_grad = True
        self.m_smpl.body_poseZ.requires_grad = False

        optimizer_0 = torch.optim.Adam([self.m_smpl.body_pose,
                                        self.m_smpl.transl], lr=self.init_lr)#,self.m_smpl.global_orient self.m_smpl.body_poseZ,
        iter_0 = max_iter
        pbar = trange(iter_0)       
        for iter in pbar:
            self.adjust_learning_rate(optimizer_0, iter)
            pbar.set_description("Frame[%03d]:" % frame_ids)
            # amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            # body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            # live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)
            
            live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=self.m_smpl.body_pose)

            depth_loss = self.depth_term.calcDepthLoss(iter=iter,
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d
            source_color_joints = live_joints[:, self.openposemap, :].squeeze(0)

            joint_loss, source_joints = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                       points=source_color_joints,
                                                       img=color_img)
            joint_loss *= self.w_joint2d

        
            # plane loss to fix foot
            if iter == 0:
                foot_temp_loss = torch.zeros(1, device=self.device, dtype=self.dtype)*self.w_temp_foot
                
                # trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
                #     export(f'{self.results_dir}/iter_smpl_0.obj')
                # trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).\
                #     export(f'{self.results_dir}/iter_plane_0.obj')
                    
                self.contact_term.update_foot_plane(
                    foot_plane=live_plane,
                    contact_data=contact_data,
                    foot_plane_ids_smplL=[self.m_smpl.foot_ids_back_smplL, self.m_smpl.foot_ids_front_smplL],
                    foot_plane_ids_smplR=[self.m_smpl.foot_ids_back_smplR, self.m_smpl.foot_ids_front_smplR])
                # import pdb;pdb.set_trace()
                # depth_color = self.color_term.projectJoints(torch.tensor(depth_vmap, device=self.device, dtype=self.dtype))
                # saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_depth_in_color.png'%(frame_ids)),
                #                     img=color_img.copy(),
                #                     joint_projected=depth_color[:,:2])
            # if iter <50:
            #     loss_geo_0 =  depth_loss
            # else:
            loss_geo_0 =  joint_loss + depth_loss
            
            # gmm_loss
            pose_prior_loss = self.gmm_term.forward(body_pose=self.m_smpl.body_pose)
            loss_geo_0 += pose_prior_loss * 0.01
            
            # temp loss for low conf 2d joints
            if target_color_joints_conf[3] < 0.65: # RElbow, need fix RShoulder
                loss_geo_0 += torch.norm(self.pre_pose[:, 16*3: 16*3+3] - self.m_smpl.body_pose[:, 16*3:16*3+3])
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 16*3: 16*3+3],self.m_smpl.body_pose[:, 16*3: 16*3+3])
            if target_color_joints_conf[4] < 0.65: # RWrist, need fix RElbow
                loss_geo_0 += torch.norm(self.pre_pose[:, 18*3: 18*3+3] - self.m_smpl.body_pose[:, 18*3:18*3+3])
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 18*3: 18*3+3], self.m_smpl.body_pose[:, 18*3: 18*3+3])           
            if target_color_joints_conf[6] < 0.65: # LElbow, need fix LShoulder
                loss_geo_0 += torch.norm(self.pre_pose[:, 15*3: 15*3+3] - self.m_smpl.body_pose[:, 15*3: 15*3+3])    
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 15*3: 15*3+3], self.m_smpl.body_pose[:, 15*3: 15*3+3])
            if target_color_joints_conf[7] < 0.65: # LWrist, need fix LElbow
                loss_geo_0 += torch.norm(self.pre_pose[:, 17*3: 17*3+3] - self.m_smpl.body_pose[:, 17*3: 17*3+3])  
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 17*3: 17*3+3], self.m_smpl.body_pose[:, 17*3: 17*3+3])
            
            # fix wrist and hand aa
            loss_geo_0 += torch.norm(self.m_smpl.body_pose[:, 19*3:19*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 20*3:20*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 21*3:21*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 22*3:22*3+3])
            
            # fix head aa and neck
            loss_geo_0 += torch.norm(self.m_smpl.body_pose[:, 14*3: 14*3+ 3]) +\
                torch.norm(self.m_smpl.body_pose[:, 11*3: 11*3+ 3])

            if iter%10 == 0:
                os.makedirs(f'{self.pose_debug_dir}/{frame_ids}', exist_ok=True)
                trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
                    export(f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_{iter}_stage_0.obj')
                saveProjectedJoints(filename=f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_{iter}_source_stage_0.png',
                            img=color_img.copy(),
                            joint_projected=source_joints)
                saveProjectedJoints(filename=f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_target_stage_0.png',
                            img=color_img.copy(),
                            joint_projected=target_color_joints)

            
            optimizer_0.zero_grad()
            loss_geo_0.backward()
            optimizer_0.step()
        
        # import pdb;pdb.set_trace()    
        
        # stage 1: optim global rot, transl, vposer
        self.m_smpl.global_orient.requires_grad = True
        optimizer_1 = torch.optim.Adam([self.m_smpl.global_orient,
                                        self.m_smpl.transl,
                                        self.m_smpl.body_pose], lr=self.init_lr)
                                        # self.m_smpl.body_pose], lr=self.init_lr)#self.m_smpl.body_poseZ
        iter_1 = max_iter//2
        pbar = trange(iter_1)       
        for iter in pbar:
            self.adjust_learning_rate(optimizer_1, iter)
            pbar.set_description("Frame[%03d]:" % frame_ids)
            # amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
            # body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
            # live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)
            
            live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=self.m_smpl.body_pose)

            depth_loss = self.depth_term.calcDepthLoss(iter=iter,
                depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                live_verts=live_verts, faces=self.m_smpl.faces)*self.w_verts3d
            source_color_joints = live_joints[:, self.openposemap, :].squeeze(0)

            joint_loss, source_joints = self.color_term.calcColorLoss(keypoints=target_color_joints,
                                                       points=source_color_joints,
                                                       img=color_img)
            joint_loss *= self.w_joint2d
            if iter%10 == 0:
                os.makedirs(f'{self.pose_debug_dir}/{frame_ids}', exist_ok=True)
                trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
                    export(f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_{iter}_stage_1.obj')
                saveProjectedJoints(filename=f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_{iter}_source_stage_1.png',
                            img=color_img.copy(),
                            joint_projected=source_joints)
                saveProjectedJoints(filename=f'{self.pose_debug_dir}/{frame_ids}/iter_smpl_target_stage_1.png',
                            img=color_img.copy(),
                            joint_projected=target_color_joints)
        
            # plane loss to fix foot
            foot_temp_loss = self.contact_term.calcTempLoss(
                live_plane=live_plane,
                contact_data=contact_data,
                foot_plane_ids_smplL=[self.m_smpl.foot_ids_back_smplL, self.m_smpl.foot_ids_front_smplL],
                foot_plane_ids_smplR=[self.m_smpl.foot_ids_back_smplR, self.m_smpl.foot_ids_front_smplR])* self.w_temp_foot
                
            loss_geo_1 = joint_loss + depth_loss  +  foot_temp_loss # + c 
            
            if contact_ids.shape[0]>0:
                # penetration
                penetrate_loss = self.contact_term.calcPenetrateLoss(
                    live_verts=live_verts, contact_ids=contact_ids) * self.w_penetrate
                # cont_loss = self.press_term.calcContLoss(live_verts=live_verts,
                #                                          contact_ids=contact_ids) * self.w_contact
                loss_geo_1 = loss_geo_1 + penetrate_loss #+ cont_loss

            # gmm_loss
            pose_prior_loss = self.gmm_term.forward(body_pose=self.m_smpl.body_pose)
            loss_geo_1 =  loss_geo_1 + pose_prior_loss * 0.01

            # temp loss for low conf 2d joints
            if target_color_joints_conf[3] < 0.65: # RElbow, need fix RShoulder
                loss_geo_0 += torch.norm(self.pre_pose[:, 16*3: 16*3+3] - self.m_smpl.body_pose[:, 16*3:16*3+3])
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 16*3: 16*3+3],self.m_smpl.body_pose[:, 16*3: 16*3+3])
            if target_color_joints_conf[4] < 0.65: # RWrist, need fix RElbow
                loss_geo_0 += torch.norm(self.pre_pose[:, 18*3: 18*3+3] - self.m_smpl.body_pose[:, 18*3:18*3+3])
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 18*3: 18*3+3], self.m_smpl.body_pose[:, 18*3: 18*3+3])           
            if target_color_joints_conf[6] < 0.65: # LElbow, need fix LShoulder
                loss_geo_0 += torch.norm(self.pre_pose[:, 15*3: 15*3+3] - self.m_smpl.body_pose[:, 15*3: 15*3+3])    
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 15*3: 15*3+3], self.m_smpl.body_pose[:, 15*3: 15*3+3])
            if target_color_joints_conf[7] < 0.65: # LWrist, need fix LElbow
                loss_geo_0 += torch.norm(self.pre_pose[:, 17*3: 17*3+3] - self.m_smpl.body_pose[:, 17*3: 17*3+3])  
                # if iter%10 == 0:
                #     print(self.pre_pose[:, 17*3: 17*3+3], self.m_smpl.body_pose[:, 17*3: 17*3+3])
            
            # fix wrist and hand aa
            loss_geo_0 += torch.norm(self.m_smpl.body_pose[:, 19*3:19*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 20*3:20*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 21*3:21*3+3]) +\
                torch.norm(self.m_smpl.body_pose[:, 22*3:22*3+3])
                                            
            # fix head aa and neck
            loss_geo_0 += torch.norm(self.m_smpl.body_pose[:, 14*3: 14*3+ 3]) +\
                torch.norm(self.m_smpl.body_pose[:, 11*3: 11*3+ 3])            


            optimizer_1.zero_grad()
            loss_geo_1.backward()
            optimizer_1.step()        
        
        import pdb;pdb.set_trace()    
        # trimesh.Trimesh(vertices=live_verts[0].detach().cpu().numpy(), faces=self.m_smpl.faces).\
        #     export(f'{self.results_dir}/iter_smpl_final.obj')
        # trimesh.Trimesh(vertices=live_plane[0].detach().cpu().numpy()).\
        #     export(f'{self.results_dir}/iter_plane_final.obj')

        self.press_term.setVertsPre(live_verts)# useless


        # amass_body_pose_rec = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        # body_pose_rec = torch.cat([amass_body_pose_rec, torch.zeros([1, 6], device=self.device)], dim=1)
        # live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=body_pose_rec)

        live_verts, live_joints, _, live_plane = self.m_smpl.updatePose(body_pose=self.m_smpl.body_pose)
        
        _verts = live_verts.detach().cpu().numpy()[0]
        output_mesh = trimesh.Trimesh(vertices=_verts,faces=self.m_smpl.faces,process=False)
        output_mesh.export(osp.join(self.results_dir,'frame%04d_%04d.obj'%(frame_ids,iter)))
        color_render, depth = self.color_term.renderMesh(output_mesh,color_img)
        # projected_joints = self.color_term.projectJoints(joints)
        # projected_joints = projected_joints.detach().cpu().numpy()
        saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),
                            img=color_render,
                            joint_projected=keypoints[:,:2])
        
        # import pdb;pdb.set_trace()
        
        # saveProjectedJoints(filename=osp.join(self.results_dir,'frame%04d_%04d.png'%(frame_ids,iter)),
        #                     img=color_render,
        #                     joint_projected=keypoints[:,:2])

        # amass_body_pose_rec0 = self.vp.decode(self.m_smpl.body_poseZ)['pose_body'].contiguous().view(-1, 63)
        # # set wrist pose to zero. vposer cannot handle wrist rot
        # amass_body_pose_rec0[:, (20-1)*3:(21-1)*3+3] = torch.zeros([1, 6], device=self.device)
        # body_pose_rec0 = torch.cat([amass_body_pose_rec0, torch.zeros([1, 6], device=self.device)], dim=1)
        # body_pose = body_pose_rec0.detach().cpu()
        
        body_pose = self.m_smpl.body_pose.detach().cpu()
        
        global_orient = self.m_smpl.global_orient.detach().cpu()
        
        pose = torch.concat([global_orient, body_pose], dim=1)
        
        output_pose = torch.zeros((pose.shape[0], 24, 3, 3), device=pose.device)
        for i in range(output_pose.shape[1]):
            output_pose[:, i] = torch.tensor(R.from_rotvec(pose.numpy()[:, i* 3: i* 3+ 3]).as_matrix(),
                                             device=pose.device)

        annot = {}
        annot['body_poseZ'] = self.m_smpl.body_poseZ.detach().cpu()
        annot['transl'] = self.m_smpl.transl.detach().cpu()
        annot['betas'] = self.m_smpl.betas.detach().cpu()        
        annot['body_pose'] = body_pose
        annot['model_scale_opt'] = self.m_smpl.model_scale_opt.detach().cpu()  
        annot['global_orient'] = self.m_smpl.global_orient.detach().cpu()  
        torch.save(annot, f'{self.results_pose_dir}/{frame_ids}.pth')
        
        
        return self.m_smpl.transl.detach().cpu(), output_pose,\
            self.m_smpl.betas.detach().cpu() 

    