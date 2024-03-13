import cv2
import trimesh
import numpy as np
from lib.utils.color_utils import rgb_code
from icecream import ic

class visFootImage():
    def __init__(self):
        self.footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
        self.footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
        model_temp = trimesh.load('essentials/smpl_uv/smpl_template.obj')
        v_template = np.array(model_temp.vertices)
        self.v_footL, self.v_footR = v_template[self.footIdsL, :], v_template[self.footIdsR, :]
        foot_verts = np.concatenate([self.v_footL, self.v_footR], axis=0)
        verts_xz = foot_verts[:, [0, 2]]
        self.faces = np.array(model_temp.faces)

        self.insole2smplR = np.load('essentials/pressure/insole2smplR.npy', allow_pickle=True).item()
        self.insole2smplL = np.load('essentials/pressure/insole2smplL.npy', allow_pickle=True).item()

    def drawSMPLFoot(self, v_foot, footIds, img_H=3300, img_W=1100,
                     contact_label=None, vert_color=None, point_size=40):
        tex_color = rgb_code['DarkCyan']
        line_color = rgb_code['Green']
        x_col = img_W - (v_foot[:, 0] - np.min(v_foot[:, 0])) / (np.max(v_foot[:, 0]) - np.min(v_foot[:, 0])) * (
                    img_W - 1) - 1
        x_row = img_H - (v_foot[:, 2] - np.min(v_foot[:, 2])) / (np.max(v_foot[:, 2]) - np.min(v_foot[:, 2])) * (
                    img_H - 1) - 1

        img = np.ones(((img_H + 50), (img_W + 100), 3), dtype=np.uint8) * 255
        point = np.concatenate([x_row.reshape([-1, 1]).astype(np.int32), x_col.reshape([-1, 1])], axis=1)

        for j in range(self.faces.shape[0]):
            x, y, z = self.faces[j]
            if x in footIds and y in footIds:
                xi = np.where(footIds == x)[0]
                yi = np.where(footIds == y)[0]
                img = cv2.line(img, (int(point[xi, 1]), int(point[xi, 0])),
                               (int(point[yi, 1]), int(point[yi, 0])),
                               (line_color[2], line_color[1], line_color[0]), 2)
            if z in footIds and y in footIds:
                zi = np.where(footIds == z)[0]
                yi = np.where(footIds == y)[0]
                img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                               (int(point[yi, 1]), int(point[yi, 0])),
                               (line_color[2], line_color[1], line_color[0]), 2)
            if z in footIds and x in footIds:
                zi = np.where(footIds == z)[0]
                xi = np.where(footIds == x)[0]
                img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                               (int(point[xi, 1]), int(point[xi, 0])),
                               (line_color[2], line_color[1], line_color[0]), 2)

        if contact_label is not None and vert_color is None:
            for i in range(point.shape[0]):
                x, y = point[i, 0], point[i, 1]
                _cont_label = contact_label[i]
                if _cont_label > 0.5:
                    v_color = rgb_code['Black']
                else:
                    v_color = rgb_code['White']
                img = cv2.circle(img, (int(y), int(x)), point_size, (int(v_color[2]), int(v_color[1]), int(v_color[0])),
                                 -1)
                img = cv2.putText(img, f"{footIds[i]}", (int(y), int(x) + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (tex_color[2], tex_color[1], tex_color[0]))
        elif contact_label is None and vert_color is not None:
            for i in range(point.shape[0]):
                x, y = point[i, 0], point[i, 1]
                v_color = vert_color[i, ::-1]
                img = cv2.circle(img, (int(y), int(x)), point_size, (int(v_color[2]), int(v_color[1]), int(v_color[0])),
                                 -1)
                img = cv2.putText(img, f"{footIds[i]}", (int(y), int(x) + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (tex_color[2], tex_color[1], tex_color[0]))

        return img

    def drawContactSMPL(self, contact_label):
        verts_num = int(contact_label.shape[0] / 2)
        imgL = self.drawSMPLFoot(self.v_footL, self.footIdsL, contact_label=contact_label[:verts_num])
        imgR = self.drawSMPLFoot(self.v_footR, self.footIdsR, contact_label=contact_label[verts_num:])
        img = np.concatenate([imgL, imgR], axis=1)
        # cv2.imwrite('debug/insole2smpl/results/%03d.png'%insole_ids,img)
        return img

    def getVertsColor(self, pressure_img, footIds, insole2smpl):
        verts_color = np.zeros([footIds.shape[0], 3])
        for i in range(footIds.shape[0]):
            ids = footIds[i]
            if str(ids) in insole2smpl.keys():
                tmp = insole2smpl[str(ids)]
                _color = pressure_img[tmp[0], tmp[1], :]
                if _color.shape[0] != 0:
                    verts_color[i] = np.sum(_color, axis=0)
        return verts_color.astype(np.int32)

    def drawPressureSMPL(self, pressure_img):
        H, W, _ = pressure_img.shape
        W_mid = int(W / 2)

        vert_colorL = self.getVertsColor(pressure_img[:, :W_mid, :], self.footIdsL, self.insole2smplL)
        imgL = self.drawSMPLFoot(self.v_footL, self.footIdsL,
                                 vert_color=vert_colorL)

        vert_colorR = self.getVertsColor(pressure_img[:, W_mid:, :], self.footIdsR, self.insole2smplR)
        imgR = self.drawSMPLFoot(self.v_footR, self.footIdsR,
                                 vert_color=vert_colorR)

        img = np.concatenate([imgL, imgR], axis=1)
        return img