import numpy as np
import open3d as o3d
import os.path as osp
import torch
import os
import glob
import cv2
from icecream import ic


from tqdm import tqdm

from lib.Utils.fileio import VideoWriter
from lib.Utils.misc import progress_bar

def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

class Camera():
    def __init__(self, dtype=torch.float32, device=None) -> None:
        self.img_w = 1280
        self.img_h = 720
        # self.focal_length = 608
        self.focal_length = estimate_focal_length(self.img_h, self.img_w)

        self.camera_center = torch.tensor([self.img_w // 2, self.img_h // 2], dtype=dtype, device=device)
        self.rotation = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=dtype, device=device)
        self.translation = torch.tensor([0, 0, 0], dtype=dtype, device=device)

class Visualizer(Camera):
    def __init__(self, cfg, time_str, view='camera', fps=30, dtype=torch.float32, device=None):
        super().__init__(dtype, device)
        task_cfg = cfg['task']
        
        self.output_path = osp.join('debug/visual', task_cfg['motion_type'], time_str)
        self.motion_type = task_cfg['motion_type']
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        
        self.fps = fps
        self.view = view

    # visualize the 3d mesh
    def render_sequence_3d(self, verts, faces, need_bg=False, save_video=True,
                        visible=False, need_norm=False, verbose=True, colors=None, view='camera'):
        """
        Render mesh animation using open3d.

        Parameters
        ----------
        verts : np.ndarray, shape [n, v, 3]
        Mesh vertices for each frame.
        faces : np.ndarray, shape [f, 3]
        Mesh faces.
        width : int
        Width of video.
        height : int
        Height of video.
        video_path : str
        Path to save the rendered video.
        fps : int, optional
        Video framerate, by default 30
        visible : bool, optional
        Wether to display rendering window, by default False
        need_norm : bool, optional
        Normalizing the vertices and locate camera automatically or not, by default
        True
        """
        if need_norm:
            if type(verts) == list:
                verts = np.stack(verts, 0)

            mean = np.mean(verts, axis=(0, 1), keepdims=True)
            scale = np.max(np.std(verts, axis=(0, 1), keepdims=True)) * 6
            verts = (verts - mean) / scale

        video_path = osp.join(self.output_path, self.motion_type + '_' + view + '.mp4')
        if save_video == True:
            writer = VideoWriter(video_path, self.img_w, self.img_h, self.fps,
                                 codec='mp4v')

        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertices = o3d.utility.Vector3dVector(verts[0])

        # 绘制open3d坐标系
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            width=4.0, height=4.0, depth=0.001)
        ground_plane.rotate(
            o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0]))
        ground_plane.translate([-2, -2, 0])

        # translate y to move the plane belove the foot
        ground_plane.translate([0, 0, 3])

        pcd_all = o3d.geometry.PointCloud()

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.img_w, height=self.img_h, visible=visible)
        vis.add_geometry(mesh)
        if view != 'camera':
            vis.add_geometry(axis)
            vis.add_geometry(ground_plane)
        if need_bg is False and view == 'camera':
            vis.add_geometry(pcd_all)

        view_control = vis.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        if view == 'front':
            cam_offset = 2
            cam_params.extrinsic = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 1.5],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ])
            view_control.set_zoom(1)
        elif view == 'side':
            cam_offset = 3
            cam_params.extrinsic = np.array([
                [0, 0, 1, -3],
                [0, -1, 0, -1],
                [-1, 0, 0, cam_offset],
                [0, 0, 0, 1],
            ])
            view_control.set_zoom(0.4)
        elif view == 'camera':
            cam_params.extrinsic = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.img_w,height=self.img_h,fx=self.focal_length,fy=self.focal_length,cx=self.img_w/2-0.5,cy=self.img_h/2-0.5)
        
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        # view_control.set_zoom(0.4)

        iterations = range(len(verts))
        if verbose:
            iterations = progress_bar(iterations)
        for idx in iterations:
            mesh.vertices = o3d.utility.Vector3dVector(verts[idx])
            mesh.compute_vertex_normals()
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
            if need_bg == True and view == 'camera':
                depth = (np.asarray(vis.capture_depth_float_buffer()) * 255).astype(np.uint8)
                mask = depth > 0
                bg_img_rgb = self.img_all[idx][:,:,::-1].copy()
                bg_img_rgb[mask] = frame[mask]
                frame = bg_img_rgb
            if save_video:
                writer.write_frame(frame)
        if save_video:
            writer.close()

    def render_two_sequence_3d(self, verts, faces, need_bg=False, save_video=True,
                        visible=False, need_norm=False, verbose=True, colors=None, view='camera'):
        """
        Render mesh animation using open3d.

        Parameters
        ----------
        verts : list, shape [num_people, n, v, 3]
        Mesh vertices for each frame.
        faces : np.ndarray, shape [f, 3]
        Mesh faces.
        width : int
        Width of video.
        height : int
        Height of video.
        video_path : str
        Path to save the rendered video.
        fps : int, optional
        Video framerate, by default 30
        visible : bool, optional
        Wether to display rendering window, by default False
        need_norm : bool, optional
        Normalizing the vertices and locate camera automatically or not, by default
        True
        """
        # if need_norm:
        #     if type(verts) == list:
        #         verts = np.stack(verts, 0)

        #     mean = np.mean(verts, axis=(0, 1), keepdims=True)
        #     scale = np.max(np.std(verts, axis=(0, 1), keepdims=True)) * 6
        #     verts = (verts - mean) / scale
        
        verts[1] = verts[1].detach().cpu().numpy()
        verts[0] = verts[0].detach().cpu().numpy()

        video_path = osp.join(self.output_path, self.motion_type + '_' + view + '.mp4')
        if save_video == True:
            writer = VideoWriter(video_path, self.img_w, self.img_h, self.fps,
                                 codec='mp4v')

        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertices = o3d.utility.Vector3dVector(verts[0][0])

        mesh1 = o3d.geometry.TriangleMesh()
        mesh1.triangles = o3d.utility.Vector3iVector(faces)
        mesh1.vertices = o3d.utility.Vector3dVector(verts[1][0])
        mesh1.paint_uniform_color([1, 0.706, 0])

        # 绘制open3d坐标系
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        camera = o3d.io.read_triangle_mesh('models/camera/camera.obj', True)
        ground_plane = o3d.geometry.TriangleMesh.create_box(
            width=4.0, height=4.0, depth=0.001)
        ground_plane.rotate(
            o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi/2, 0, 0]))
        
        ground_plane.translate([-2, -2, 0])

        # translate y to move the plane belove the foot
        ground_plane.translate([0, 0, 3])
        # ground_plane.translate([0, -0.5, 3])

        # pcd = o3d.geometry.PointCloud()
        pcd_all = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud[120:130,120:130].reshape(-1,3))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.img_w, height=self.img_h, visible=visible)
        vis.add_geometry(mesh)
        vis.add_geometry(mesh1)
        if view != 'camera':
            vis.add_geometry(camera)
            vis.add_geometry(ground_plane)
        if need_bg is False and view == 'camera':
            vis.add_geometry(pcd_all)

        view_control = vis.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        if view == 'front':
            cam_offset = 3.5
            cam_params.extrinsic = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ])
            view_control.set_zoom(1)
        elif view == 'side':
            cam_offset = 3
            cam_params.extrinsic = np.array([
                [0, 0, 1, -3],
                [0, -1, 0, 0.5],
                [-1, 0, 0, cam_offset],
                [0, 0, 0, 1],
            ])
            view_control.set_zoom(0.4)
        elif view == 'camera':
            cam_params.extrinsic = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            # cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.img_w,height=self.img_h,fx=(self.img_w * self.img_w + self.img_h * self.img_h) ** 0.5,fy=(self.img_w * self.img_w + self.img_h * self.img_h) ** 0.5,cx=self.img_w/2-0.5,cy=self.img_h/2-0.5)
            cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.img_w,height=self.img_h,fx=self.focal_length,fy=self.focal_length,cx=self.img_w/2-0.5,cy=self.img_h/2-0.5)
        elif view == 'overlook':
            cam_offset = 7
            extrinsic = np.array([
                [ 0.       ,  0.       ,  1., -3],
                [0.5      , -0.8660254,  0., 0],
                [-0.8660254, -0.5      ,  0., cam_offset],
                [0, 0, 0, 1],
            ])
            # rot = R.from_rotvec([np.pi/6,0,0]).as_matrix()
            # mat = copy.deepcopy(extrinsic[:3,:3])
            # ic(rot, mat)
            # ic(rot @ mat)
            # ic(extrinsic[:3,:3])
            # extrinsic[:3,:3] = copy.deepcopy(rot @ mat)
            # ic(extrinsic[:3,:3])
            cam_params.extrinsic = extrinsic
            # view_control.set_zoom(0.1)
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        # view_control.set_zoom(0.4)

        iterations = range(len(verts[0]))
        if verbose:
            iterations = progress_bar(iterations)
        for idx in iterations:
            mesh.vertices = o3d.utility.Vector3dVector(verts[0][idx])
            mesh1.vertices = o3d.utility.Vector3dVector(verts[1][idx])
            mesh.compute_vertex_normals()
            mesh1.compute_vertex_normals()
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx])
                mesh1.vertex_colors = o3d.utility.Vector3dVector(colors[idx])
            vis.update_geometry(mesh)
            vis.update_geometry(mesh1)
            vis.poll_events()
            vis.update_renderer()
            frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
            if need_bg == True and view == 'camera':
                depth = (np.asarray(vis.capture_depth_float_buffer()) * 255).astype(np.uint8)
                mask = depth > 0
                bg_img_rgb = self.img_all[idx][:,:,::-1].copy()
                bg_img_rgb[mask] = frame[mask]
                frame = bg_img_rgb
            if save_video:
                writer.write_frame(frame)
        if save_video:
            writer.close()