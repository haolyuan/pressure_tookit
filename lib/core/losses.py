import torch.nn as nn
import torch
import numpy as np
import trimesh

from lib.fitSMPL.depthTerm import DepthTerm
from lib.fitSMPL.colorTerm import ColorTerm
from lib.fitSMPL.gmmTerm import MaxMixturePriorLoss
from lib.fitSMPL.contactTerm import ContactTerm


class SMPLifyMMVPLoss(nn.Module):
    def __init__(self,
                 essential_root=None,
                 model_faces=None,
                 # depth related
                 dIntr=None,depth_size=None,
                 # color related
                 cIntr=None, color_size=None,  
                 # temp foot related
                 temp_contact_label=None,  
                 pre_pose=None,           
                 #  loss weight
                 depth_weight=10,keypoint_weights=0.01,
                 shape_weights=0.1,
                 penetrate_weights=10,
                 limb_weights=1, gmm_weights=0.01,
                 tfoot_weights=0.0,
                 tpose_weights=0.0,
                 # others
                 stage='init_shape',
                 dtype=torch.float32):
        super(SMPLifyMMVPLoss, self).__init__()
        
        self.dtype = dtype
        # fitting stage
        assert stage in ['init_shape', 'init_pose', 'tracking']
        self.stage = stage
        
        # loss weights
        self.register_buffer('depth_weight', torch.tensor(depth_weight, dtype=dtype))       
        self.register_buffer('keypoint_weights', torch.tensor(keypoint_weights, dtype=dtype))       
        self.register_buffer('shape_weights', torch.tensor(shape_weights, dtype=dtype))       
        self.register_buffer('penetrate_weights', torch.tensor(penetrate_weights, dtype=dtype))           
        self.register_buffer('gmm_weights', torch.tensor(gmm_weights, dtype=dtype))       
        self.register_buffer('limb_weights', torch.tensor(limb_weights, dtype=dtype))       
        self.register_buffer('tfoot_weights', torch.tensor(tfoot_weights, dtype=dtype))       
        self.register_buffer('tpose_weights', torch.tensor(tpose_weights, dtype=dtype))

        self.depth_term = DepthTerm(essential_root=essential_root,
                                    cam_intr=dIntr,
                                    img_W=depth_size[0],
                                    img_H=depth_size[1],
                                    faces=model_faces,
                                    save_obj=True,
                                    dtype=dtype)
        self.color_term = ColorTerm(cam_intr=cIntr,
                                    img_W=color_size[0],
                                    img_H=color_size[1],
                                    dtype=dtype)        
        self.gmm_term = MaxMixturePriorLoss(prior_folder=f'{essential_root}/smplify_essential')
        self.contact_term = ContactTerm(essential_root=essential_root)
        
        # add temp loss      
        self.iter_idx_0 = True # whether iteration first
        self.pre_contact_data = temp_contact_label
        self.pre_pose = pre_pose

        self.model_faces = model_faces

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, joint_weights, joint_mapper,
                gt_joints, gt_depth_nmap, gt_depth_vmap, gt_contact):
        
        if gt_contact is not None:
            contact_ids, _, _ = self.contact_term.contact2smpl(np.array(gt_contact))
        elif self.stage != 'init_shape':
            print(f'stage {self.stage} has no contact data')
            raise ValueError    
  
        # smpl output
        live_vertices = body_model_output.vertices
        live_joints = body_model_output.joints
        live_plane = body_model_output.foot_plane
        live_pose = body_model_output.body_pose
        live_betas = body_model_output.betas
        # trimesh.Trimesh(vertices=live_plane.detach().cpu().numpy()[0]).export('debug/curr_footplane.obj')


        device = live_joints.device
        
        # ==== joint loss / 2d keypoint loss ====
        openposemap_color_joints = live_joints[:, joint_mapper[0], :].squeeze(0)
        halpemap_color_joints = gt_joints[joint_mapper[1], :].clone().detach().to(device)

        
        
        
        projected_joints = camera.projectJoints(openposemap_color_joints)
        joint_loss = self.keypoint_weights * self.color_term(keypoint_data=halpemap_color_joints,
                                             projected_joints=projected_joints,
                                             joint_weights=joint_weights)

        
        
        # ==== vertices loss / 3d depth loss ====
        # trimesh.Trimesh(vertices=live_vertices[0].detach().cpu().numpy(), faces=self.model_faces).\
        #     export('debug/new_framework/live_verts_floor.obj')
        live_vertices_depth = camera.tranform3d(points=live_vertices,
                                                type='f2d')# bs should be set as 1
        depth_loss = self.depth_weight * self.depth_term(depth_vmap=gt_depth_vmap,
                                                      depth_nmap=gt_depth_nmap,
                                                      live_verts=live_vertices_depth)
        
        total_loss = joint_loss + depth_loss
        
        if self.stage == 'init_shape':
            # ==== dense penetrate loss ====
            # foot should be stand close to ground when A-pose
            penetrate_loss = self.penetrate_weights * (torch.mean(torch.abs(live_plane[:, :30, 1]) +\
                torch.abs(live_plane[:, 30+42:30*2+42, 1] ))  +\
                torch.mean(torch.abs(live_plane[:, 30:30+42, 1] ) +\
                torch.abs(live_plane[:, 30*2+42:, 1] )))
            total_loss += penetrate_loss
        
            # ==== limbs fixing loss ====
            limbfixed_loss = torch.norm(live_pose[:, 19*3: 19*3+3]) +\
                torch.norm(live_pose[:, 20*3: 20*3+3]) +\
                torch.norm(live_pose[:, 21*3: 21*3+3]) +\
                torch.norm(live_pose[:, 22*3: 22*3+3]) +\
                torch.norm(live_pose[:, 6*3: 7*3+3]) +\
                torch.norm(live_pose[:, 9*3: 10*3+3])
            limbfixed_loss *= self.limb_weights
            
            # ==== gmm loss ====
            pose_prior_loss = self.gmm_term(body_pose=live_pose) * self.gmm_weights
            total_loss += pose_prior_loss
            

        
        elif self.stage == 'init_pose':
            # ==== gmm loss ====
            pose_prior_loss = self.gmm_term(body_pose=live_pose) * self.gmm_weights
            total_loss +=  pose_prior_loss
            
            # limbfixed_loss = torch.norm(live_pose[:, 19*3: 19*3+3]) +\
            #     torch.norm(live_pose[:, 20*3: 20*3+3]) +\
            #     torch.norm(live_pose[:, 21*3: 21*3+3]) +\
            #     torch.norm(live_pose[:, 22*3: 22*3+3])         
            # limbfixed_loss *= self.limb_weights

            # ==== temp pose loss ====
            # temp loss for low conf 2d joints
            gt_conf = gt_joints[:, 2]
            if gt_conf[3] < 0.65: # RElbow, need fix RShoulder
                total_loss += torch.norm(self.pre_pose[:, 16*3: 16*3+3] - live_pose[:, 16*3:16*3+3])
            if gt_conf[4] < 0.65: # RWrist, need fix RElbow
                total_loss += torch.norm(self.pre_pose[:, 18*3: 18*3+3] - live_pose[:, 18*3:18*3+3])
            if gt_conf[6] < 0.65: # LElbow, need fix LShoulder
                total_loss += torch.norm(self.pre_pose[:, 15*3: 15*3+3] - live_pose[:, 15*3: 15*3+3])    
            if gt_conf[7] < 0.65: # LWrist, need fix LElbow
                total_loss += torch.norm(self.pre_pose[:, 17*3: 17*3+3] - live_pose[:, 17*3: 17*3+3])  

            # fix joints rot
            total_loss += torch.norm(live_pose[:, 19*3:19*3+3]) +\
                torch.norm(live_pose[:, 20*3:20*3+3]) +\
                torch.norm(live_pose[:, 21*3:21*3+3]) +\
                torch.norm(live_pose[:, 22*3:22*3+3]) +\
                torch.norm(live_pose[:, 14*3: 14*3+ 3]) +\
                torch.norm(live_pose[:, 11*3: 11*3+ 3]) +\
                torch.norm(live_pose[:, 17*3+2: 17*3+3]) +\
                torch.norm(live_pose[:, 18*3+2: 18*3+3]) +\
                torch.relu( -1 * live_pose[:, 18*3+1].squeeze(0)) +\
                torch.relu( 1 * live_pose[:, 17*3+1].squeeze(0))   

            if contact_ids.shape[0] > 0:
                penetrate_loss = torch.mean(torch.abs(live_vertices[0, contact_ids, 1]))* self.penetrate_weights
                total_loss += penetrate_loss
        
        elif self.stage == 'tracking':
            # ==== gmm loss ====
            pose_prior_loss = self.gmm_term(body_pose=live_pose) * self.gmm_weights

            total_loss += pose_prior_loss

            # ==== temp pose loss ====
            # temp loss for low conf 2d joints
            gt_conf = gt_joints[:, 2]
            if gt_conf[3] < 0.65: # RElbow, need fix RShoulder
                total_loss += torch.norm(self.pre_pose[:, 16*3: 16*3+3] - live_pose[:, 16*3:16*3+3])
            if gt_conf[4] < 0.65: # RWrist, need fix RElbow
                total_loss += torch.norm(self.pre_pose[:, 18*3: 18*3+3] - live_pose[:, 18*3:18*3+3])
            if gt_conf[6] < 0.65: # LElbow, need fix LShoulder
                total_loss += torch.norm(self.pre_pose[:, 15*3: 15*3+3] - live_pose[:, 15*3: 15*3+3])    
            if gt_conf[7] < 0.65: # LWrist, need fix LElbow
                total_loss += torch.norm(self.pre_pose[:, 17*3: 17*3+3] - live_pose[:, 17*3: 17*3+3])     
   
            
            # fix joints rot
            total_loss += torch.norm(live_pose[:, 19*3:19*3+3]) +\
                torch.norm(live_pose[:, 20*3:20*3+3]) +\
                torch.norm(live_pose[:, 21*3:21*3+3]) +\
                torch.norm(live_pose[:, 22*3:22*3+3]) +\
                torch.norm(live_pose[:, 14*3: 14*3+ 3]) +\
                torch.norm(live_pose[:, 11*3: 11*3+ 3]) +\
                torch.norm(live_pose[:, 17*3+2: 17*3+3]) +\
                torch.norm(live_pose[:, 18*3+2: 18*3+3]) +\
                torch.relu( -1 * live_pose[:, 18*3+1].squeeze(0)) +\
                torch.relu( 1 * live_pose[:, 17*3+1].squeeze(0))   


            if contact_ids.shape[0] > 0:
                penetrate_loss = torch.mean(torch.abs(live_vertices[0, contact_ids, 1]))* self.penetrate_weights
                total_loss += penetrate_loss
            
            if self.iter_idx_0:
                self.contact_term.update_foot_plane(
                    foot_plane=live_plane,
                    contact_data=self.pre_contact_data,
                    foot_plane_ids_smplL=[body_model_output.foot_ids[0], body_model_output.foot_ids[1]],
                    foot_plane_ids_smplR=[body_model_output.foot_ids[2], body_model_output.foot_ids[3]])
                # foot_temp_loss = self.contact_term.calcTempLoss(
                #     live_plane=live_plane,
                #     contact_data=gt_contact,
                #     foot_plane_ids_smplL=[body_model_output.foot_ids[0], body_model_output.foot_ids[1]],
                #     foot_plane_ids_smplR=[body_model_output.foot_ids[2], body_model_output.foot_ids[3]])* self.tfoot_weights    
                # update iteration state
                self.iter_idx_0 = False
            else:
                foot_temp_loss = self.contact_term.calcTempLoss(
                    live_plane=live_plane,
                    contact_data=gt_contact,
                    foot_plane_ids_smplL=[body_model_output.foot_ids[0], body_model_output.foot_ids[1]],
                    foot_plane_ids_smplR=[body_model_output.foot_ids[2], body_model_output.foot_ids[3]])* self.tfoot_weights                    
                    
                total_loss += foot_temp_loss    
        # import pdb;pdb.set_trace()
        # print(joint_loss, depth_loss, pose_prior_loss, penetrate_loss) 
        
        # print(live_pose[:, 7*3:7*3+3], live_pose[:, 6*3:6*3+3])
               
        return total_loss