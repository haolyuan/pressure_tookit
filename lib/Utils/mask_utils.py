import os
import imageio,cv2
import os.path as osp
import numpy as np
from tqdm import tqdm

from lib.Utils.depth_utils import calcSceneDepth

def calcDepthMask(th=50.):
    scene_depth_path = "D:/code_C++/mkv_tracker/results/s12/MoCap_20230422_150025/depth"
    scene_path = osp.join(scene_depth_path, 'frame_%d.png')
    scene_depth = calcSceneDepth(scene_path,127,136)

    basedir = 'D:/code_C++/mkv_tracker/results/s12'
    rgbd_dir = osp.join(basedir,'MoCap_20230422_145333')
    depth_dir = osp.join(rgbd_dir,'depth')
    mask_dir = osp.join(rgbd_dir,'mask')
    if not osp.exists(mask_dir):

        os.makedirs(mask_dir)
    depth_fn_ls = sorted([x for x in os.listdir(depth_dir)
                         if x.endswith('.png') or x.endswith('.jpg')])
    h1,h2,w1,w2 = 155,487,210,515
    # h1,h2,w1,w2 = 155,460,238,450
    for depth_name in tqdm(depth_fn_ls):
        depth_path = osp.join(depth_dir,depth_name)
        depth_map = imageio.imread(depth_path).astype(np.float32)
        depth_map = cv2.blur(depth_map, (7, 7))
        a = abs(depth_map-scene_depth)>th
        b = depth_map > 100.
        c = depth_map < 2500.
        mask = np.logical_and(a, b,c)
        mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask[:h1,:]=0
        mask[h2:,:]=0
        mask[:,:w1]=0
        mask[:,w2:]=0
        cv2.imwrite(osp.join(mask_dir,depth_name),mask)





if __name__ == "__main__":
    calcDepthMask()