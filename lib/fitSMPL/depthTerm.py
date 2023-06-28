import copy
import torch
import scipy
import trimesh
import torch.nn as nn
import cv2
import trimesh
import numpy as np
from icecream import ic

from lib.visualizer.renderer import modelRender
from lib.Utils.depth_utils import depth2PointCloud#(depth_map,fx,fy,cx,cy)

class DepthTerm(nn.Module):
    def __init__(self,cam_intr,img_W,img_H):
        super(DepthTerm, self).__init__()
        self.cam_intr = cam_intr #fx,fy,cx,cy
        self.img_W = img_W
        self.img_H = img_H
        self.renderer = modelRender(cam_intr,img_W,img_H)



    def findLiveVisibileVerticesIndex(self, mesh, floor2depth, near_size=4, th=0.005, upsampleFlag=False):
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
        flame_visible_idx = np.where(min_dists < th)[0]

        # vertices = mesh.vertices
        # visible_vertices = vertices[flame_visible_idx, :]
        # ic(visible_vertices.shape)
        # with open('./debug/visibility.obj', 'w') as fp:
        #     for v in visible_vertices:
        #         fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        # exit()
        return flame_visible_idx

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
