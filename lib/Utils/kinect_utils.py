import os
import os.path as osp
from tqdm import tqdm,trange
from icecream import ic

def renameRGBD(basdir,flag=[0,0,0,0]):
    '''
    :param basdir: rgbd dir
    :param flag: [color,depth,mask,keypoints]
    :return: None
    '''
    '''color'''
    if flag[0]:
        color_dir = osp.join(basdir,'color')
        file_fn_ls = sorted([x for x in os.listdir(color_dir)
                            if x.endswith('.png') or x.endswith('.jpg')])
        for file_fn in tqdm(file_fn_ls):
            ids = int(file_fn[6:-4])
            src = osp.join(color_dir,file_fn)
            dst = osp.join(color_dir,'%03d.png'%(ids-1))
            os.rename(src, dst)
    '''depth'''
    if flag[1]:
        depth_dir = osp.join(basdir, 'depth')
        file_fn_ls = sorted([x for x in os.listdir(depth_dir)
                            if x.endswith('.png') or x.endswith('.jpg')])
        for file_fn in tqdm(file_fn_ls):
            ids = int(file_fn[6:-4])
            src = osp.join(depth_dir, file_fn)
            dst = osp.join(depth_dir, '%03d.png' % (ids - 1))
            os.rename(src, dst)
    '''mask'''
    if flag[2]:
        mask_dir = osp.join(basdir, 'mask')
        file_fn_ls = sorted([x for x in os.listdir(mask_dir)
                            if x.endswith('.png') or x.endswith('.jpg')])
        for file_fn in tqdm(file_fn_ls):
            ids = int(file_fn[6:-4])
            src = osp.join(mask_dir, file_fn)
            dst = osp.join(mask_dir, '%03d.png' % (ids - 1))
            os.rename(src, dst)
    '''keypoints'''
    if flag[3]:
        mask_dir = osp.join(basdir, 'mask')
        file_fn_ls = sorted([x for x in os.listdir(mask_dir)
                            if x.endswith('.png') or x.endswith('.jpg')])
        for file_fn in tqdm(file_fn_ls):
            ids = int(file_fn[6:-4])
            src = osp.join(mask_dir, file_fn)
            dst = osp.join(mask_dir, '%03d.png' % (ids - 1))
            os.rename(src, dst)
    exit()
    # os.rename(images_fn, dst)

if __name__ == '__main__':
    basdir = 'E:/dataset/PressureDataset/20230422/RGBD/S12/MoCap_20230422_145422'
    renameRGBD(basdir,flag=[0,0,0,1])