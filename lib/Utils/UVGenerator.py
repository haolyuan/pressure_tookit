# Convert contact label or 3d position to UV map.
# ZhangHe 2023.2.2
# References: https://gitee.com/seuvcl/CVPR2020-OOH

import numpy as np
import pickle
import cv2
import smplx
import argparse
from icecream import ic


class UVGenerator():
    def __init__(self, UV_pickle='essentials/smpl_uv/param.pkl',
                 mask_path='essentials/smpl_uv/UVMASK.png'):
        mask_img = cv2.imread(mask_path)
        self.uv_mask = np.sum(mask_img, axis=-1) < 1e-6
        self.uv_template = cv2.imread("../../bodyModels/smpl/smpl_uv/smpl_uv_20200910.png")
        # self.uv_mask = mask_img<1e-6
        with open(UV_pickle, 'rb') as f:
            tmp = pickle.load(f)
        for k in tmp.keys():
            setattr(self, k, tmp[k])

    def UV_interp(self, rgbs):
        # face_num = self.vt_faces.shape[0]
        vt_num = self.texcoords.shape[0]
        assert (vt_num == rgbs.shape[0])
        # uvs = self.vts #self.texcoords * np.array([[self.h - 1, self.w - 1]])
        triangle_rgbs = rgbs[self.vt_faces][self.face_id]
        bw = self.bary_weights[:, :, np.newaxis, :]
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        return im

    def getPositionUVMap(self, verts):
        '''
        Convert contact 3d position to UV map.
        '''
        xmin = verts[:, 0].min()
        xmax = verts[:, 0].max()
        ymin = verts[:, 1].min()
        ymax = verts[:, 1].max()
        zmin = verts[:, 2].min()
        zmax = verts[:, 2].max()
        vmin = np.array([xmin, ymin, zmin])
        vmax = np.array([xmax, ymax, zmax])
        box = (vmax - vmin).max()  # 2019.11.9 vmax.max()
        verts = (verts - vmin) / box - 0.5
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = verts[vt_to_v_index]
        uv = self.UV_interp(rgbs)
        return uv, vmin, vmax, box

    def refineUV(self, im, save_path=None):
        H, W, C = im.shape
        im = cv2.resize(im, (self.uv_template.shape[0], self.uv_template.shape[1]))
        ids = np.where(np.sum(im, axis=-1) < 100)
        im[ids[0], ids[1], :] = self.uv_template[ids[0], ids[1], :]
        im = cv2.resize(im, (H, W))
        if save_path is not None:
            cv2.imwrite(save_path, im)
        return im

    def getContUVMap(self, cont_lable, mask_color=[0.5, 0.5, 0.5], img_size=512, save_path=None):
        '''
        Convert contact label to UV map:
            cont_lable(BGR):[...,self-contact,body-scene contact]
            self-contact: blue
            body-scene contact: red
            no-contact: white
            mask: gray
        '''
        vt_to_v_index = np.array([
            self.vt_to_v[i] for i in range(self.texcoords.shape[0])
        ])
        rgbs = cont_lable[vt_to_v_index]
        triangle_rgbs = rgbs[self.vt_faces][self.face_id]
        bw = self.bary_weights[:, :, np.newaxis, :]
        im = np.matmul(bw, triangle_rgbs).squeeze(axis=2)
        im[self.uv_mask, :] = mask_color  # np.array([0.5,0.5,0.5])
        im = (im * 255).astype(np.uint8)
        im = cv2.resize(im, (img_size, img_size))
        im = self.refineUV(im)
        if save_path is not None:
            cv2.imwrite(save_path, im)
        return im


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--seq_name', required=True, help='sequence name')
    # args = parser.parse_args()
    cont_fn = '/data/zhangh/rich_mnt/human_scene_contact/BBQ_001_guitar/00196/genCont_001.npy'
    conta_lable = np.load(cont_fn, allow_pickle=True)

    cont_rgb = np.zeros([6890, 3])
    cont_rgb[conta_lable < -0.5] = [0, 1, 0]
    cont_rgb[conta_lable > 0.5] = [1, 0, 0]

    m_gen = UVGenerator()
    m_gen.getContUVMap(cont_rgb, save_path='debug/uvtmp.png')

