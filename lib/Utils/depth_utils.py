import os.path as osp
import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh
from icecream import ic
import os

def calcSceneDepth(scene_name,start_frame,end_frame):
    scene_list = []
    for idx in range(start_frame,end_frame+1):
        scene_path = scene_name % idx
        depth_map = imageio.imread(scene_path).astype(np.float32)# / 1000.
        depth_map = cv2.blur(depth_map, (7, 7))
        scene_list.append(depth_map)
    scene_depth = np.stack(scene_list)
    scene_mean = np.mean(scene_depth,axis=0)
    return scene_mean

def depth2PointCloud(depth_map,fx,fy,cx,cy):
    z = depth_map
    r = np.arange(depth_map.shape[0])
    c = np.arange(depth_map.shape[1])
    x, y = np.meshgrid(c, r)
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z
    pointCloud = np.dstack([x, y, z])
    return pointCloud

#============================Floor============================
def normalizeFloor(floor_normal,floor_trans=([0,0,0]), target_normal=np.array([0, 1, 0])):
    axis = np.cross(floor_normal, target_normal)
    axis = (axis) / np.linalg.norm(axis)
    angle = np.arccos(np.dot(floor_normal, target_normal))  # *180/math.pi
    aa = angle * axis
    rot = Rot.from_rotvec(aa)
    r2 = rot.as_matrix()
    floor_mat = np.eye(4)
    floor_mat[:3, :3] = r2
    floor_mat[:3, 3] = floor_trans
    return floor_mat

def calculateFloorNormal(depth_path, xy,fx,fy,cx,cy, save_path=None):
    # depth_map = imageio.imread(depth_path).astype(np.float32) / 1000.  # (576,640)
    depth_map = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
    pointCloud = depth2PointCloud(depth_map,fx,fy,cx,cy)  # (576,640,3)
    # trimesh.Trimesh(vertices=pointCloud.reshape(-1, 3),process=False).export('debug/pointCloud.obj')
    depth_slices = (pointCloud[xy[0]:xy[1], xy[2]:xy[3], :]).reshape([-1, 3])
    # trimesh.Trimesh(vertices=depth_slices,process=False).export('debug/depth_slice.obj')
    A = depth_slices
    b = -1 * np.ones(depth_slices.shape[0])
    ATA, ATb = A.T @ A, A.T @ b
    ATA_inv = np.linalg.inv(ATA)
    x = np.dot(ATA_inv, ATb)
    floor_normal = x / np.linalg.norm(x)

    floor_rot = normalizeFloor(floor_normal)
    depth_slices_rot = (floor_rot[:3, :3] @ depth_slices.T + floor_rot[:3, 3].reshape([3, 1])).T
    floor_trans = np.array(
        [-np.median(depth_slices_rot[:, 0]), -np.median(depth_slices_rot[:, 1]), -np.median(depth_slices_rot[:, 2])])
    depth2floor = normalizeFloor(floor_normal=floor_normal, floor_trans=floor_trans)
    if save_path is not None:
        results = {
            'trans':floor_trans,
            'normal':floor_normal,
            'depth2floor':depth2floor
        }
        np.save(save_path, results)

    pointCloud = pointCloud.reshape([-1,3])
    depth_slices_RT = (floor_rot[:3, :3] @ pointCloud.T + floor_trans.reshape([3, 1])).T
    trimesh.Trimesh(vertices=depth_slices_RT, process=False).export('debug/depth_slice_rot.obj')

    return floor_normal, floor_trans, depth2floor


if __name__ == '__main__':
    
    sub_id = 'S01'
    seq_name = 'MoCap_20230422_092117'
    
    os.makedirs(f'/data/yuanhaolei/PressureDataset_label/20230422/{sub_id}',exist_ok=True)

    cali_data = dict(np.load(f'/data/PressureDataset/20230422/{sub_id}/{seq_name}/calibration.npy',
                        allow_pickle=True).item())
    
    # height range, weight range
    floor_normal, floor_trans, depth2floor = calculateFloorNormal(
        f'/data/PressureDataset/20230422/{sub_id}/{seq_name}/depth/003.png',
        [509, 566, 237, 468],
        fx=cali_data['depth_Intr']['fx'],
        fy=cali_data['depth_Intr']['fy'],
        cx=cali_data['depth_Intr']['cx'],
        cy=cali_data['depth_Intr']['cy'],
        save_path=f'/data/yuanhaolei/PressureDataset_label/20230422/{sub_id}/floor_{sub_id}.npy')