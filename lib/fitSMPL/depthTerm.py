import copy
import math
import numpy as np
import scipy
import torch
import torch.nn as nn
import trimesh
from lib.visualizer.renderer import modelRender

from lib.utils.depth_utils import depth2PointCloud


class DepthTerm(nn.Module):

    def __init__(self,
                 essential_root=None,
                 cam_intr=None,
                 img_W=640,
                 img_H=576,
                 faces=None,
                 save_obj=False,
                 dtype=np.float32,
                 device='cpu'):
        super(DepthTerm, self).__init__()

        self.cam_intr = cam_intr.cpu().numpy()  # fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.dtype = dtype
        self.device = device
        hand_ids = np.loadtxt(f'{essential_root}/hand_ids.txt').astype(
            np.int32)
        self.valid_verts = np.ones(6890)
        self.valid_verts[hand_ids] = 0

        self.renderer = modelRender(cam_intr, img_W, img_H)

        self.foot_ids_surfaces = np.load(
            f'{essential_root}/foot_related/foot_ids_surfaces.npy').tolist()

        self.model_faces = faces

        self.save_obj = save_obj

    def findLiveVisibileVerticesIndex(self, mesh, near_size=4, th=0.005):
        _mesh = copy.deepcopy(mesh)
        color_render, depth = self.renderer.render(_mesh)

        # cv2.imwrite('debug/new_framework/test.png',color_render)
        # _mesh.export('debug/new_framework/mesh_.obj')
        # import pdb;pdb.set_trace()

        # mesh.export('debug/mesh.obj')
        point_cloud = depth2PointCloud(depth, self.cam_intr[0],
                                       self.cam_intr[1], self.cam_intr[2],
                                       self.cam_intr[3])
        kdtree = scipy.spatial.cKDTree(point_cloud.reshape([-1, 3]))
        dists, indices = kdtree.query(_mesh.vertices, k=near_size)
        min_dists = np.min(dists, axis=-1)
        flame_visible_idx = np.where((min_dists < th)
                                     & (self.valid_verts > 0.5))[0]

        return flame_visible_idx

    def findCorrs(self,
                  depth_vmap=None,
                  depth_nmap=None,
                  live_verts=None,
                  icp_near_size=32,
                  icp_theta_thresh=np.pi / 12,
                  icp_dist_thresh=0.05):
        # TODO: run ICP on cuda
        depth_vmap = depth_vmap.cpu().numpy()
        depth_nmap = depth_nmap.cpu().numpy()
        live_verts = live_verts.detach().cpu().numpy()

        live_mesh = trimesh.Trimesh(
            vertices=live_verts, faces=self.model_faces, process=False)

        # live_mesh.export('debug/new_framework/smpl_vertices.obj')
        # trimesh.Trimesh(vertices=depth_vmap).export('debug/new_framework/gt_depth_vmap.obj')
        # import pdb;pdb.set_trace()

        live_normals = live_mesh.vertex_normals
        smpl_visible_idx_w_foot = self.findLiveVisibileVerticesIndex(live_mesh)
        smpl_visible_idx = [
            idx for idx in smpl_visible_idx_w_foot
            if idx not in self.foot_ids_surfaces
        ]

        verts_src = live_verts[smpl_visible_idx, :]
        normal_src = live_normals[smpl_visible_idx, :]
        # trimesh.Trimesh(vertices=verts_src).export('debug/new_framework/smpl_visible.obj')

        # find smpl verts currs to depth, each smpl v corres to one depth v
        kdtree_depth = scipy.spatial.cKDTree(depth_vmap)
        dists_smpl2depth, indices_smpl2depth = kdtree_depth.query(
            verts_src, k=icp_near_size)

        tar_normals_smpl2depth = depth_nmap[indices_smpl2depth.reshape(
            -1)].reshape(-1, icp_near_size, 3)

        cosine_smpl2depth = np.einsum('ijk,ik->ij', tar_normals_smpl2depth,
                                      normal_src)
        valid_smpl2depth = (dists_smpl2depth < icp_dist_thresh) & (
            cosine_smpl2depth > math.cos(icp_theta_thresh))
        valid_indices_smpl2depth = np.argmax(valid_smpl2depth, axis=1)
        indices_corr_smpl2depth = indices_smpl2depth[
            np.arange(valid_smpl2depth.shape[0]), valid_indices_smpl2depth]

        # find depth verts currs to smpl, each depth v corres to one smpl v
        kdtree_depth2smpl = scipy.spatial.cKDTree(verts_src)
        dists_depth2smpl, indices_depth2smpl = kdtree_depth2smpl.query(
            depth_vmap, k=icp_near_size)

        tar_normals_depth2smpl = normal_src[indices_depth2smpl.reshape(
            -1)].reshape(-1, icp_near_size, 3)
        cosine_depth2smpl = np.einsum('ijk,ik->ij', tar_normals_depth2smpl,
                                      depth_nmap)
        valid_depth2smpl = (dists_depth2smpl < icp_dist_thresh) & (
            cosine_depth2smpl > math.cos(icp_theta_thresh))
        valid_indices_depth2smpl = np.argmax(valid_depth2smpl, axis=1)
        indices_corr_depth2smpl = indices_depth2smpl[
            np.arange(valid_depth2smpl.shape[0]), valid_indices_depth2smpl]

        # save
        if self.save_obj:
            tar_verts = depth_vmap[indices_corr_smpl2depth]
            with open('debug/corrs_smpl2depth%d.obj', 'w') as fp:
                for vi in range(verts_src.shape[0]):
                    fp.write(
                        'v %f %f %f\n' %
                        (verts_src[vi, 0], verts_src[vi, 1], verts_src[vi, 2]))
                    fp.write(
                        'v %f %f %f\n' %
                        (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
                for li in range(verts_src.shape[0]):
                    fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))

            tar_verts = verts_src[indices_corr_depth2smpl]
            with open('debug/corrs_depth2smpl%d.obj', 'w+') as fp:
                for vi in range(depth_vmap.shape[0]):
                    fp.write('v %f %f %f\n' %
                             (depth_vmap[vi, 0], depth_vmap[vi, 1],
                              depth_vmap[vi, 2]))
                    fp.write(
                        'v %f %f %f\n' %
                        (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
                for li in range(depth_vmap.shape[0]):
                    fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))
        return smpl_visible_idx, indices_corr_smpl2depth, \
            indices_corr_depth2smpl

    def forward(self, depth_vmap=None, depth_nmap=None, live_verts=None):
        """_summary_

        Args:
            depth_vmap (torch.tensor, optional): depth cloud. Defaults to None.
            depth_nmap (torch.tensor, optional): depth cloud norm.
                Defaults to None.
            live_verts (torch.Size([6890, 3]), optional):
                smpl surface vertices. Defaults to None.

        Returns:
            _type_: _description_
        """
        smpl_ids_vis, depth_ids, smpl_ids = self.findCorrs(
            depth_vmap=depth_vmap,
            depth_nmap=depth_nmap,
            live_verts=live_verts)

        # calculate smpl2depth corres loss
        smpl_ids_vis = torch.tensor(smpl_ids_vis, device=self.device).long()
        # depth_ids = torch.tensor(depth_ids, device= self.device).long()
        src_verts_smpl2depth = live_verts[smpl_ids_vis, :]
        tar_verts_smpl2depth = depth_vmap[depth_ids, :]
        # calculate depth2smpl corres loss
        src_verts_depth2smpl = depth_vmap
        tar_verts_depth2smpl = live_verts[smpl_ids_vis, :][smpl_ids, :]

        delta_smpl2depth = src_verts_smpl2depth - tar_verts_smpl2depth
        dist_smpl2depth = torch.norm(delta_smpl2depth, dim=-1)
        delta_depth2smpl = src_verts_depth2smpl - tar_verts_depth2smpl
        dist_depth2smpl = torch.norm(delta_depth2smpl, dim=-1)
        # import pdb;pdb.set_trace()

        depth_loss = torch.mean(
            dist_smpl2depth, dim=0) + torch.mean(
                dist_depth2smpl, dim=0)

        return depth_loss
