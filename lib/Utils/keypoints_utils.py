import json,sys
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm,trange
from icecream import ic
from color_utils import rgb_code

parents={
    'body':[[1,2],[2,9],[9,8],[8,12],[12,5],[1,5],
            [2,3],[3,4],[5,6],[6,7],
            [9,10],[10,11],[11,24],[11,22],[11,23],
            [12,13],[13,14],[14,21],[14,19],[14,20],
            [0,15],[0,16],[15,17],[16,18]],
    'hand':[[0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],
            [0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20]],
}#openpose

def drawkeppoints(
        img_path=None,
        kp_path=None,
        bone_thick=2,
        joint_thick=1,
        draw_joints=False,
        joint_color=rgb_code['Red'],
        bone_color=rgb_code['Cyan']):

    img = cv2.imread(img_path)
    with open(kp_path) as file:
        data = json.load(file)
    keypoints = data['people'][0]
    pose_keypoints_2d = np.array(keypoints['pose_keypoints_2d']).reshape([-1, 3])
    face_keypoints_2d = np.array(keypoints['face_keypoints_2d']).reshape([-1, 3])
    hand_left_keypoints_2d = np.array(keypoints['hand_left_keypoints_2d']).reshape([-1, 3])
    hand_right_keypoints_2d = np.array(keypoints['hand_right_keypoints_2d']).reshape([-1, 3])

    for line in parents['body']:
        start_ids, end_ids = line
        start_point = pose_keypoints_2d[start_ids, :]
        end_point = pose_keypoints_2d[end_ids, :]
        if start_point[0] > 1e-6 and start_point[1] > 1e-6 \
                and end_point[0] > 1e-6 and end_point[1] > 1e-6:
            img = cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                           (bone_color[2], bone_color[1], bone_color[0]), bone_thick)

    for line in parents['hand']:
        start_ids, end_ids = line
        start_point = hand_left_keypoints_2d[start_ids, :]
        end_point = hand_left_keypoints_2d[end_ids, :]
        if start_point[0] > 1e-6 and start_point[1] > 1e-6 \
                and end_point[0] > 1e-6 and end_point[1] > 1e-6:
            img = cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                           (bone_color[2], bone_color[1], bone_color[0]), bone_thick)

    for line in parents['hand']:
        start_ids, end_ids = line
        start_point = hand_right_keypoints_2d[start_ids, :]
        end_point = hand_right_keypoints_2d[end_ids, :]
        if start_point[0] > 1e-6 and start_point[1] > 1e-6 \
                and end_point[0] > 1e-6 and end_point[1] > 1e-6:
            img = cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                           (bone_color[2], bone_color[1], bone_color[0]), bone_thick)

    if pose_keypoints_2d[4, 0] > 1e-6 and pose_keypoints_2d[4, 1] > 1e-6 \
            and hand_right_keypoints_2d[0, 0] > 1e-6 and hand_right_keypoints_2d[0, 1] > 1e-6:
        img = cv2.line(img, (int(pose_keypoints_2d[4, 0]), int(pose_keypoints_2d[4, 1])),
                       (int(hand_right_keypoints_2d[0, 0]), int(hand_right_keypoints_2d[0, 1])),
                       (bone_color[2], bone_color[1], bone_color[0]), bone_thick)
    if pose_keypoints_2d[7, 0] > 1e-6 and pose_keypoints_2d[7, 1] > 1e-6 \
            and hand_left_keypoints_2d[0, 0] > 1e-6 and hand_left_keypoints_2d[0, 1] > 1e-6:
        img = cv2.line(img, (int(pose_keypoints_2d[7, 0]), int(pose_keypoints_2d[7, 1])),
                       (int(hand_left_keypoints_2d[0, 0]), int(hand_left_keypoints_2d[0, 1])),
                       (bone_color[2], bone_color[1], bone_color[0]), bone_thick)

    for i in range(face_keypoints_2d.shape[0]):
        x = face_keypoints_2d[i, 0]
        y = face_keypoints_2d[i, 1]
        conf = face_keypoints_2d[i, 2]
        img = cv2.circle(img, (int(x), int(y)), 2, (bone_color[2], bone_color[1], bone_color[0]), -1)


    if draw_joints:
        for i in range(pose_keypoints_2d.shape[0]):
            x =pose_keypoints_2d[i,0]
            y =pose_keypoints_2d[i,1]
            conf =pose_keypoints_2d[i,2]
            img = cv2.putText(img, f"{i}", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (joint_color[2],joint_color[1],joint_color[0]))
            img = cv2.circle(img, (int(x), int(y)), joint_thick, (0,0,255),0)
        # for i in range(hand_left_keypoints_2d.shape[0]):
        #     x =hand_left_keypoints_2d[i,0]
        #     y =hand_left_keypoints_2d[i,1]
        #     conf =hand_left_keypoints_2d[i,2]
        #     img = cv2.putText(img, f"{i}", (int(x), int(y)),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (joint_color[2],joint_color[1],joint_color[0]))
        #     img = cv2.circle(img, (int(x), int(y)), joint_thick, (0,0,255),0)
        # for i in range(hand_right_keypoints_2d.shape[0]):
        #     x =hand_right_keypoints_2d[i,0]
        #     y =hand_right_keypoints_2d[i,1]
        #     conf =hand_right_keypoints_2d[i,2]
        #     img = cv2.putText(img, f"{i}", (int(x), int(y)),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (joint_color[2],joint_color[1],joint_color[0]))
        #     img = cv2.circle(img, (int(x), int(y)), joint_thick, (0,0,255),0)

    return img


if __name__ =='__main__':
    basdir = 'E:/dataset/PressureDataset/S12/RGBD/MoCap_20230422_145333'
    for i in trange(1,160):
        img_fn = 'frame_%d'%i
        img_path = osp.join(basdir,'color/%s.png'%img_fn)
        kp_path = osp.join(basdir,'openpose/frame_%d_keypoints.json'%i)
        img = drawkeppoints(img_path=img_path,kp_path=kp_path,
                            bone_thick=2,
                            draw_joints=True)

        cv2.imwrite('debug/kp/%s_kp.png' % img_fn, img)

