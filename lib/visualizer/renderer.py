import numpy as np
import trimesh
import cv2
import math
import pyrender
from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation as R
from icecream import ic

class modelRender():
    def __init__(self,cam_intr,img_W,img_H,color=[1.0, 1.0, 0.9], wireframe=False):
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        R = np.eye(3)
        T = np.zeros([3, 1])
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R.T
        camera_pose[:3, 3:4] = np.dot(-R.T, T)
        camera_pose[:, 1:3] = -camera_pose[:, 1:3]

        # self.light_node = self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))
        # self.light_node._matrix = camera_pose
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)
        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

        camera = pyrender.IntrinsicsCamera(fx=cam_intr[0], fy=cam_intr[1], cx=cam_intr[2], cy=cam_intr[3])
        self.cam_node = self.scene.add(camera, name='cam', pose=camera_pose)
        # set the scene

        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_W, viewport_height=img_H,point_size=1.0)

        if wireframe:
            self.render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            self.render_flags = RenderFlags.RGBA

    def render(self,mesh,img=None):
        mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)
        mesh_node = self.scene.add(mesh, 'mesh')
        rgb, depth = self.renderer.render(self.scene, flags=self.render_flags)
        if img is not None:
            valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
            output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
            image = output_img.astype(np.uint8)
        else:
            image = rgb
        self.scene.remove_node(mesh_node)
        return image,depth