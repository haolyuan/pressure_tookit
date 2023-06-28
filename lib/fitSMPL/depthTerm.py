import torch
import scipy
import trimesh
import torch.nn as nn
import cv2
import numpy as np
from icecream import ic

from lib.visualizer.renderer import modelRender

class DepthTerm(nn.Module):
    def __init__(self,cam_intr,img_W,img_H):
        super(DepthTerm, self).__init__()
        self.renderer = modelRender(cam_intr,img_W,img_H)



    def findLiveVisibileVerticesIndex(self, mesh, near_size=4, th=0.005, upsampleFlag=False):
        _, depth = self.renderer.render(mesh)

    #     _, depth_map = self.render_depth(self.live_vertices_cpu, dIntr)
    #     point_cloud = depth2PointCloud(dIntr, depth_map)
    #     kdtree = scipy.spatial.cKDTree(point_cloud.reshape([-1, 3]))
    #     dists, indices = kdtree.query(self.live_vertices_cpu, k=near_size)
    #     min_dists = np.min(dists, axis=-1)
    #     flame_visible_idx = np.where(min_dists < th)[0]
    #     # visible_vertices =vertices[flame_visible_idx[0],:]
    #     # ic(visible_vertices.shape)
    #     # with open('./debug/flame_arkit/visible_vertices.obj', 'w') as fp:
    #     #     for v in visible_vertices:
    #     #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #     # exit()
    #     if upsampleFlag:
    #         visible_idx_list = flame_visible_idx.tolist()
    #         visible_face = []
    #         for i in range(self.face_upsample.shape[0]):
    #             faceOne = self.face_upsample[i]
    #             if (faceOne[0] in visible_idx_list) and (faceOne[1] in visible_idx_list) and (
    #                     faceOne[2] in visible_idx_list):
    #                 visible_face.append(faceOne)
    #         visible_upsample_trianlge = np.array(visible_face)
    #         return flame_visible_idx, visible_upsample_trianlge
    #     else:
    #         return flame_visible_idx
