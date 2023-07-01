import copy
import torch
import scipy
import trimesh
import torch.nn as nn
import cv2
import trimesh
import math
import numpy as np
from icecream import ic

from lib.visualizer.renderer import modelRender
from lib.Utils.depth_utils import depth2PointCloud#(depth_map,fx,fy,cx,cy)

class DepthTerm(nn.Module):
    def __init__(self,cam_intr,img_W,img_H,dtype,device):
        super(DepthTerm, self).__init__()
        self.cam_intr = cam_intr #fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.dtype=dtype
        self.device=device
        hand_ids = np.loadtxt('essentials/hand_ids.txt').astype(np.int32)
        self.valid_verts = np.ones(6890)
        self.valid_verts[hand_ids] = 0

        self.renderer = modelRender(cam_intr,img_W,img_H)



    def findLiveVisibileVerticesIndex(self, mesh, floor2depth, near_size=4, th=0.005):
        _mesh = copy.deepcopy(mesh)
        _mesh.apply_transform(floor2depth)
        color_render, depth = self.renderer.render(_mesh)
        # cv2.imwrite('debug/test.png',color_render)
        # _mesh.export('debug/mesh_.obj')
        # mesh.export('debug/mesh.obj')
        point_cloud = depth2PointCloud(depth, self.cam_intr[0], self.cam_intr[1],
                                       self.cam_intr[2], self.cam_intr[3])
        kdtree = scipy.spatial.cKDTree(point_cloud.reshape([-1, 3]))
        dists, indices = kdtree.query(_mesh.vertices, k=near_size)
        min_dists = np.min(dists, axis=-1)
        flame_visible_idx = np.where((min_dists < th)&(self.valid_verts>0.5))[0]

        return flame_visible_idx

    def findDepthCorrs(self,depth_vmap=None,depth_nmap=None,
                       live_verts=None, faces=None,
                       floor2depth=None,
                       icp_near_size=32,icp_theta_thresh=np.pi / 6,
                       icp_dist_thresh=0.05):
        # live_verts_cpu = live_verts.detach().cpu().numpy()[0]
        live_mesh = trimesh.Trimesh(vertices=live_verts,
                        faces=faces, process=False)
        live_normals = live_mesh.vertex_normals
        smpl_visible_idx = self.findLiveVisibileVerticesIndex(live_mesh,floor2depth)
        verts_src = live_verts[smpl_visible_idx,:]
        normal_src = live_normals[smpl_visible_idx,:]

        kdtree = scipy.spatial.cKDTree(depth_vmap)
        dists, indices = kdtree.query(verts_src, k=icp_near_size)
        tar_normals = depth_nmap[indices.reshape(-1)].reshape(-1, icp_near_size, 3)

        cosine = np.einsum('ijk,ik->ij', tar_normals, normal_src)
        valid = (dists < icp_dist_thresh) & (cosine > math.cos(icp_theta_thresh))
        valid_indices = np.argmax(valid, axis=1)
        indices_corr = indices[np.arange(valid.shape[0]), valid_indices]

        # save
        # tar_verts = depth_vmap[indices_corr]
        # with open('debug/vposer/corrs%d.obj'%frame_idx, 'w') as fp:
        #     for vi in range(verts_src.shape[0]):
        #         fp.write('v %f %f %f\n' % (verts_src[vi, 0], verts_src[vi, 1], verts_src[vi, 2]))
        #         fp.write('v %f %f %f\n' % (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
        #     for li in range(verts_src.shape[0]):
        #         fp.write('l %d %d\n' % (2*li + 1, 2*li+2))
        # exit()
        return smpl_visible_idx,indices_corr

    def calcDepthLoss(self,depth_vmap=None, depth_nmap=None,
                      live_verts=None,faces=None,floor2depth=None):
        live_verts_cpu = live_verts.detach().cpu().numpy()[0]
        # live_mesh = trimesh.Trimesh(vertices=live_verts_cpu,
        #                 faces=faces, process=False)
        smpl_ids,depth_ids = self.findDepthCorrs(depth_vmap=depth_vmap, depth_nmap=depth_nmap,
                                                 live_verts=live_verts_cpu,faces=faces,
                                                 floor2depth=floor2depth)

        depth_vmap = torch.tensor(depth_vmap,dtype=self.dtype,device=self.device)
        # depth_nmap = torch.tensor(depth_nmap,dtype=self.dtype,device=self.device)
        smpl_ids = torch.tensor(smpl_ids,device=self.device).long()
        depth_ids = torch.tensor(depth_ids,device=self.device).long()
        delta = (live_verts[0,smpl_ids,:]-depth_vmap[depth_ids,:])
        dist = torch.norm(delta, dim=-1)
        depth_loss = torch.mean(dist,dim=0)
        return depth_loss
