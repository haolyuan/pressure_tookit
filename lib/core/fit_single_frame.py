import torch
from tqdm import tqdm
import trimesh, copy
import numpy as np
import cv2

from lib.core.losses import SMPLifyMMVPLoss
from lib.Utils import loadweights
from lib.core import fitting
from lib.optimizers import optim_factory


def fit_single_frame(img,
                     depth_mask,
                     keypoints,
                     depth_map,
                     contact_label,
                     init_pose,
                     essential_root,
                     body_model,
                     camera,
                     depth_size, color_size,
                     joint_mapper,
                     joint_weights=None,
                     stage='init_shape'
                     ):

    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')
    dtype = torch.float32
    
    gt_depth_vmap, gt_depth_nmap, dv_floor, dn_floor = camera.preprocessDepth(depth_map, depth_mask)
    trimesh.Trimesh(vertices=dv_floor.detach().cpu().numpy()).export('debug/new_framework/dv_floor.obj')
    
    # TODO: it will be beneficial to fitting shape if init model scale for different ages if available
    # init transl
    init_transl = torch.zeros((1, 3), dtype=dtype, device=device)
    init_transl[:, 0] = torch.mean(dv_floor[:, 0])
    init_transl[:, 1] = 0.97
    init_transl[:, 2] = torch.mean(dv_floor[:, 2] - 0.1)
    # init body_pose
    init_pose = torch.zeros([1,body_model.NUM_JOINTS*3], dtype=dtype, device=device)
    # init_pose[:,47] = -0.6
    # init_pose[:,50] = 0.6

    params_dict = {
        'transl': init_transl,
        'body_pose': init_pose,
        
    }
    body_model.setPose(**params_dict)
    body_model.update_shape()
    body_model.init_plane()
    model_output = body_model.update_pose()

    loss = SMPLifyMMVPLoss(essential_root=essential_root,
                           model_faces=body_model.faces,
                           dIntr=camera.dIntr, depth_size=depth_size,
                           cIntr=camera.cIntr, color_size=color_size,
                           stage=stage,
                           dtype=dtype
                           )
    loss = loss.to(device)


    

    # import pdb;pdb.set_trace()



    # load weights
    opt_weights = loadweights.load_weights(stage)

    with fitting.FittingMonitor(stage=stage) as monitor:
    #######################################################################
    # start fitting
        if stage == 'init_shape':
            all_optiparamnames = {# 
                0:['betas', 'model_scale_opt', 'transl', 'global_orient', 'body_pose'],
                1:['body_pose', 'transl', 'global_orient']
            }
        else:
            all_optiparamnames = {
                0:['body_pose', 'transl', 'global_orient']
            }  
        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='optim part')):
            optiparamnames = all_optiparamnames[opt_idx]
            final_params = [x[1] for x in body_model.named_parameters() \
                               if x[0] in optiparamnames and x[1].requires_grad]

            loss.reset_loss_weights(curr_weights)
            
            body_optimizer, body_create_graph = optim_factory.create_optimizer(
                        final_params,
                        optim_type='adam'
                        ) 
            body_optimizer.zero_grad()
            

            # create closure for body fitting
            closure = monitor.create_fitting_closure(
                        body_optimizer,
                        body_model,
                        camera=camera,
                        joint_mapper=joint_mapper,
                        joint_weights=joint_weights,
                        gt_joints=keypoints,
                        gt_depth_nmap=gt_depth_nmap,
                        gt_depth_vmap=gt_depth_vmap,
                        gt_contact=contact_label,
                        loss=loss,
                        create_graph=body_create_graph)

            final_loss_val = monitor.run_fitting(
                                   body_optimizer,
                                   closure,
                                   final_params,
                                   body_model)
            
            if stage == 'init_shape':
                body_model.update_shape()
                body_model.init_plane()
                model_output = body_model.update_pose()
            else:
                model_output = body_model.update_pose()
                
            vertices = model_output.vertices.detach().cpu().numpy().squeeze(0)
            
            ####    save output results ####
            mesh = trimesh.Trimesh(vertices=vertices,
                                   faces=body_model.faces
                                    )
            mesh.export('debug/new_framework/live_verts_floor.obj')
            # render
            color_vertices = camera.tranform3d(model_output.vertices, type='f2c')
            mesh = trimesh.Trimesh(vertices=color_vertices.detach().cpu().numpy(),
                                   faces=body_model.faces
                                    )
            # mesh.export('debug/new_framework/mesh_.obj')
            loss.color_term.renderMesh(mesh=mesh, img=img)
            # save joints data
            live_joints = model_output.joints
            img_src, img_tar = copy.deepcopy(img), copy.deepcopy(img)
            halpemap_color_joints = keypoints[joint_mapper[1], :].clone().detach().cpu().numpy()
            openposemap_color_joints = live_joints[:, joint_mapper[0], :].squeeze(0)
            projected_joints = camera.projectJoints(openposemap_color_joints)
            for i in range(projected_joints.shape[0]):
                x = projected_joints[i, 0]
                y = projected_joints[i, 1]
                img_src = cv2.putText(img_src, f"{i}", (int(x), int(y)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                img_src = cv2.circle(img_src, (int(x), int(y)), 1, (0, 0, 255), 0)    

                x = halpemap_color_joints[i, 0]
                y = halpemap_color_joints[i, 1]
                img_tar = cv2.putText(img_tar, f"{i}", (int(x), int(y)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                img_tar = cv2.circle(img_tar, (int(x), int(y)), 1, (0, 0, 255), 0)   

            cv2.imwrite('debug/new_framework/kp_src.png', img_src)
            cv2.imwrite('debug/new_framework/kp_tar.png', img_tar)
            import pdb;pdb.set_trace()



    import pdb;pdb.set_trace()
