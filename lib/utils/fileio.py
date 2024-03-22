import cv2
import json
import numpy as np
import os
import os.path as osp
import pickle
from icecream import ic


def read_json(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)


def read_pkl(filename):
    return pickle.load(open(filename, 'rb'))


def save_pkl(filename, data):
    file = open(filename, 'wb')
    pickle.dump(data, file)


# ==============================OBJ Saver==============================
def saveOBJ(filename, model):
    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' %
                             (v[0], v[1], v[2], vc[0], vc[1], vc[2]))


def saveFloorAsOBJ(filename, floor_point, floor_normal):
    with open(filename, 'w') as fp:
        fp.write('v %f %f %f\n' %
                 (floor_point[0], floor_point[1], floor_point[2]))
        fp.write('v %f %f %f\n' %
                 (floor_point[0] + floor_normal[0], floor_point[1] +
                  floor_normal[1], floor_point[2] + floor_normal[2]))
        fp.write('l 1 2\n')


def saveJointsAsOBJ(filename, joints, parents):
    with open(filename, 'w') as fp:
        for joint in joints:
            fp.write('v %f %f %f\n' % (joint[0], joint[1], joint[2]))
        for pi in range(1, parents.shape[0]):
            fp.write('l %d %d\n' % (pi + 1, parents[pi] + 1))


def saveNormalsAsOBJ(filename, verts, normals, ratio=0.2):
    with open(filename, 'w') as fp:
        for vi in range(verts.shape[0]):
            fp.write('v %f %f %f\n' %
                     (verts[vi, 0], verts[vi, 1], verts[vi, 2]))
            fp.write('v %f %f %f\n' % (verts[vi, 0] + ratio * normals[vi, 0],
                                       verts[vi, 1] + ratio * normals[vi, 1],
                                       verts[vi, 2] + ratio * normals[vi, 2]))
        for li in range(verts.shape[0]):
            fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))


def saveCorrsAsOBJ(filename, verts_src, tar_verts):
    with open(filename, 'w') as fp:
        for vi in range(verts_src.shape[0]):
            fp.write('v %f %f %f\n' %
                     (verts_src[vi, 0], verts_src[vi, 1], verts_src[vi, 2]))
            fp.write('v %f %f %f\n' %
                     (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
        for li in range(verts_src.shape[0]):
            fp.write('l %d %d\n' % (2 * li + 1, 2 * li + 2))


# ==============================Image Saver==============================
def saveProjectedJoints(filename=None, img=None, joint_projected=None):
    for i in range(joint_projected.shape[0]):
        x = joint_projected[i, 0]
        y = joint_projected[i, 1]
        # img = cv2.putText(img, f"{i}", (int(x), int(y)),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        img = cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 0)
    cv2.imwrite(filename, img)


# ==============================Video Saver==============================
def saveImgSeqAsvideo(basdir, fps=30, ratio=1.0, color=[0, 0, 255]):
    img_ls = sorted([
        x for x in os.listdir(basdir)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    img_width, img_height, _ = cv2.imread(osp.join(basdir, img_ls[0])).shape
    size = (int(img_height / ratio), int(img_width / ratio))
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # videoWrite = cv2.VideoWriter(video_path, fourcc, fps, size)
    video_path = osp.join(basdir, 'video.mp4')
    videoWrite = cv2.VideoWriter(video_path,
                                 cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                 fps, size)

    for ii in range(len(img_ls)):
        img_fn = img_ls[ii]
        img = cv2.imread(osp.join(basdir, img_fn))
        img = cv2.resize(img,
                         (int(img_height / ratio), int(img_width / ratio)))
        img = cv2.putText(img, f'{ii}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                          (color[0], color[1], color[2]), 2)
        videoWrite.write(img)
    videoWrite.release()
    ic('Free view video frame done!!')


class VideoWriter:
    """Write frames to a video.

    Call `write_frame` to write a single frame. Call `close` to release
    resource.
    """

    def __init__(self, path, width, height, fps, codec='avc1'):
        """
        Parameters
        ----------
        path : str
          Path to the video.
        width : int
          Frame width.
        height : int
          Frame height.
        fps : int
          Video frame rate.
        codec : str, optional
          Video codec, by default H264.
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps,
                                     (width, height))
        self.frame_idx = 0

    def write_frame(self, frame):
        """Write one frame.

        Parameters
        ----------
        frame : np.ndarray
          Frame to write.
        """
        self.video.write(np.flip(frame, axis=-1).copy())
        self.frame_idx += 1

    def close(self):
        """Release resource."""
        self.video.release()
