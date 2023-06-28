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
        hand_ids = np.loadtxt('essentials/hand_ids.txt').astype(np.int32)
        self.valid_verts = np.ones(6890)
        self.valid_verts[hand_ids] = 0

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
        flame_visible_idx = np.where((min_dists < th)&(self.valid_verts>0.5))[0]

        return flame_visible_idx
