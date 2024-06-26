import cv2
import glob
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset


def create_dataset(
        basdir,
        dataset_name=None,
        sub_ids=None,
        seq_name=None,
        start_img_idx=0,
        end_img_idx=-1,
        init_root=None,
        stage='init_shape',  # init_pose, tracking
):

    return Pressure_Dataset(
        basdir,
        dataset_name,
        sub_ids,
        seq_name,
        start_img_idx=start_img_idx,
        end_img_idx=end_img_idx,
        init_root=init_root,
        stage=stage)


def read_rtm_kpts(keypoint_fn):
    try:
        data = dict(np.load(keypoint_fn, allow_pickle=True).item())
    except:  # noqa: E722
        data = np.load(keypoint_fn, allow_pickle=True)[0]  # only one people

    points = np.array(data['keypoints'])
    points_score = np.array(data['keypoint_scores'])
    points_score = points_score[:, np.newaxis]
    target_keypoints = np.concatenate([points, points_score], axis=1)

    return target_keypoints


def load_contact(contact_fn):
    insole_data = dict(np.load(contact_fn, allow_pickle=True).item())
    region_l, region_r = insole_data['insole'][0], insole_data['insole'][1]
    contact_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    row_range = [range(0, 8), range(8, 15), range(15, 23), range(23, 31)]
    # left foot
    for row in range(region_l.shape[0]):
        for col in range(region_l.shape[1]):
            if region_l[row][col] != 0:
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 7):
                        contact_label[0][1] = 1
                    if col in range(7, 11):
                        contact_label[0][0] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[0][4] = 1
                    if col in range(4, 8):
                        contact_label[0][3] = 1
                    if col in range(8, 11):
                        contact_label[0][2] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 4):
                        contact_label[0][6] = 1
                    if col in range(4, 11):
                        contact_label[0][5] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 4):
                        contact_label[0][8] = 1
                    if col in range(4, 11):
                        contact_label[0][7] = 1
    # right foot
    for row in range(region_r.shape[0]):
        for col in range(region_r.shape[1]):
            if region_r[row][col] != 0:
                # 0, 1
                if row in row_range[0]:
                    if col in range(0, 4):
                        contact_label[1][0] = 1
                    if col in range(4, 11):
                        contact_label[1][1] = 1
                # 2, 3, 4
                if row in row_range[1]:
                    if col in range(0, 4):
                        contact_label[1][2] = 1
                    if col in range(4, 8):
                        contact_label[1][3] = 1
                    if col in range(8, 11):
                        contact_label[1][4] = 1
                # 5, 6
                if row in row_range[2]:
                    if col in range(0, 8):
                        contact_label[1][5] = 1
                    if col in range(8, 11):
                        contact_label[1][6] = 1
                # 7, 8
                if row in row_range[3]:
                    if col in range(0, 8):
                        contact_label[1][7] = 1
                    if col in range(8, 11):
                        contact_label[1][8] = 1
    return contact_label


def load_init_pose(data_fn, form='cliff'):
    """_summary_

    Args:
        data_fn (str): file path for loading init data
        form (str, optional): date format for pose data. Defaults to 'cliff'.

    Returns:
        numpy.ndarray: load previous pose data or \
            other network output pose data
    """
    # """_summary_

    # Args:
    #     data_fn (str): file path for loading init data

    # Returns:
    #     _type_: load previous pose data or other network output pose data
    # """

    init_global_rot = None
    init_transl = None

    # TODO: now only load cliff format
    if form == 'cliff':
        init_data = dict(np.load(data_fn).items())

        if init_data['pose'].shape[0] > 1:
            init_pose = np.expand_dims(init_data['pose'][0], 0)
        else:
            init_pose = init_data['pose']
        return init_pose[:, 3:], init_global_rot, init_transl
    elif form == 'tracking':
        init_data = dict(np.load(data_fn).items())
        init_pose = init_data['body_pose']
        init_global_rot = init_data['global_rot']
        init_transl = init_data['transl']
        return init_pose, init_global_rot, init_transl
    else:
        print('unsupported format when load init pose')
        raise TypeError


def load_init_shape(data_fn):
    shape_param = dict(np.load(data_fn).items())
    return shape_param['shape'], shape_param['model_scale_opt']


class Pressure_Dataset(Dataset):

    def __init__(
            self,
            basdir,
            dataset_name,
            sub_ids,
            seq_name,
            start_img_idx=0,
            end_img_idx=-1,
            init_root=None,
            dtype=torch.float32,
            stage='init_shape',  # init_pose, tracking
    ):
        super(Pressure_Dataset, self).__init__()

        self.dtype = dtype

        self.basdir = basdir
        self.dataset_name = dataset_name
        self.sub_ids = sub_ids
        self.seq_name = seq_name
        self.init_root = init_root

        self.start_idx = int(start_img_idx)
        self.end_idx = int(end_img_idx)

        self.stage = stage

        self.cnt = 0

        self.rgbd_path = osp.join(basdir, 'images', self.dataset_name,
                                  self.sub_ids, self.seq_name)
        # rgb
        self.img_paths = sorted(
            glob.glob(osp.join(self.rgbd_path, 'color', '**'), recursive=True))
        self.img_paths = [
            x for x in filter(lambda x: x.split('.')[-1] == 'png',
                              self.img_paths)
        ][self.start_idx:self.end_idx]
        # depth
        self.depth_paths = sorted(
            glob.glob(osp.join(self.rgbd_path, 'depth', '**'), recursive=True))
        self.depth_paths = [
            x for x in filter(lambda x: x.split('.')[-1] == 'png',
                              self.depth_paths)
        ][self.start_idx:self.end_idx]
        # depth_mask
        self.dmask_paths = sorted(
            glob.glob(
                osp.join(self.rgbd_path, 'depth_mask', '**'),
                recursive=True))  # mask
        self.dmask_paths = [
            x for x in filter(lambda x: x.split('.')[-1] == 'png',
                              self.dmask_paths)
        ][self.start_idx:self.end_idx]
        # keypoints
        # kp seq length corres to pressure, but not image
        kp_root = osp.join('input', f'{self.sub_ids}/{self.seq_name}')
        self.kp_paths = sorted(
            glob.glob(osp.join(kp_root, 'keypoints', '**'), recursive=True))
        self.kp_paths = [
            x
            for x in filter(lambda x: x.split('.')[-1] == 'npy', self.kp_paths)
        ][self.start_idx:self.end_idx]

        # pressure data, A-pose has no pressure data

        self.pressure_paths = sorted(
            glob.glob(
                osp.join(self.rgbd_path, 'insole', '**'), recursive=True))
        self.pressure_paths = [
            x for x in filter(lambda x: x.split('.')[-1] == 'npy',
                              self.pressure_paths)
        ]
        self.pressure_paths = [
            x for x in self.pressure_paths
            if int(x.rsplit('/', 1)[-1].split('.')[0]) >= self.start_idx
            and int(x.rsplit('/', 1)[-1].split('.')[0]) < self.end_idx
        ]

        # init shape data
        self.shape_path = None if stage == 'init_shape 'else \
            osp.join(self.basdir, 'annotations', self.dataset_name,
                     'smpl_pose', self.sub_ids,
                     f'init_shape_{self.sub_ids}.npz')

        # import pdb;pdb.set_trace()
        # joint mapper
        self.joint_mapper = self.init_joint_mapper()

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(25, dtype=np.float32)
        # These joints are ignored because SMPL has no neck.
        optim_weights[1] = 0
        # put higher weights on knee and elbow joints for mimic'ed poses
        optim_weights[[10, 13]] = 2
        optim_weights[[3, 6, 4, 7]] = 200

        optim_weights[[17, 18]] = 0

        return torch.tensor(optim_weights)

    def init_joint_mapper(self):
        openposemap = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24
        ])[np.newaxis, :]
        halpemap = np.array([
            0, 18, 6, 8, 10, 5, 7, 9, 19, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3,
            20, 22, 24, 21, 23, 25
        ])[np.newaxis, :]
        return np.concatenate([openposemap, halpemap], axis=0).tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= len(self.pressure_paths):
            raise StopIteration

        self.cnt += 1

        return self.read_item(self.cnt - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.read_item(idx)

    def read_item(self, idx):
        # load data

        # rgb
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path).astype(np.float32)  # [:, :, ::-1] / 255.0
        # depth
        depth_path = self.depth_paths[idx]
        depth_map = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_mask
        dmask_path = self.dmask_paths[idx]
        mask_ori = cv2.imread(dmask_path)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask_ori, kernel, 1)
        dmask = np.mean(mask, axis=-1)
        # keypoints
        kp_path = self.kp_paths[idx]
        keypoints = read_rtm_kpts(kp_path)
        frame_kp = torch.from_numpy(keypoints).float()
        # insole pressure
        if self.stage == 'tracking':
            pressure_path = self.pressure_paths[idx]
            contact_label = load_contact(pressure_path)
        elif self.stage == 'init_shape':
            # contact_label = None
            # we assume that people should keep A-pose when init shape
            contact_label = np.ones((2, 9)).tolist()
        else:  # init pose
            pressure_path = self.pressure_paths[idx]
            contact_label = load_contact(pressure_path)
            # optionally
            # contact_label = np.ones_like(np.array(contact_label)).tolist()

        # temp insole pressure
        if self.stage == 'tracking':
            pre_pressure_path = self.pressure_paths[idx - 1]
            pre_contact_label = load_contact(pre_pressure_path)
        else:
            pre_contact_label = None

        # load initial pose data
        # init pose data
        # TODO: combine different format data
        if self.stage == 'init_shape':
            init_pose, init_betas, init_scale, init_global_rot, init_transl = \
                None, None, None, None, None
        if self.stage == 'init_pose':
            init_path = osp.join(self.init_root, self.dataset_name,
                                 self.sub_ids, self.seq_name,
                                 f'{self.seq_name}_cliff_hr48.npz')
            init_pose, init_global_rot, init_transl = load_init_pose(
                init_path, form='cliff')
            init_betas, init_scale = load_init_shape(self.shape_path)
        if self.stage == 'tracking':
            curr_idx = self.start_idx + idx - 1
            init_path = osp.join(
                self.init_root.rsplit('/', 1)[0], 'results', self.dataset_name,
                self.sub_ids, self.seq_name, f'smpl_{curr_idx:03d}.npz')
            init_pose, init_global_rot, init_transl = load_init_pose(
                init_path, form='tracking')
            init_betas, init_scale = load_init_shape(self.shape_path)

        output_dict = {
            'root_path': self.rgbd_path,
            'depth_map': depth_map,
            'img': img,
            'depth_mask': dmask,
            'kp': frame_kp,
            'contact_label': contact_label,
            'pre_contact_label': pre_contact_label,
            'init_pose': init_pose,
            'init_betas': init_betas,
            'init_scale': init_scale,
            'init_global_rot': init_global_rot,
            'init_transl': init_transl
        }

        return output_dict
