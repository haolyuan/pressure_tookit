# MMVP
<img src="docs/teaser.png" width="100%">

This repo is used for optimzation and visualization for MMVP dataset:

MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors  
[He Zhang](https://github.com/zhanghebuaa), [Shenghao Ren](https://www.wjrzm.com/), [Haolei Yuan](https://github.com/haolyuan),
Jianhui Zhao, Fan Li, Shuangpeng Sun, Zhenghao Liang, [Tao Yu](https://ytrock.com/), Qiu Shen, Xun Cao

# Overview
<img src="docs/Mocap_20230422_172438.gif" width="50%"><img src="docs/Mocap_20230422_132043.gif" width="50%"><img src="docs/Mocap_20230422_151220.gif" width="50%"><img src="docs/Mocap_20230422_182056.gif" width="50%">

# News
- **[25/03/24]**: Code for optimzation are released!

# Demo
[Demo](https://github.com/Metaverse-AI-Lab-THU/MMVP-Dataset/tree/visualizing) for visulization released.

Instruction for optimization:
1. get essential files for optimizing [here](https://drive.google.com/file/d/15D2I9W4oYXZ2rbN94aFFevacua2JjOj9/view?usp=drive_link), then put it in code root.
2. Fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSf1uLLqXfnVXR2MoyRv5bF9D5gEMc8m8YASa38sTohzVcaVUg/viewform?usp=sf_link) to request authorization to use MMVP for non-commercial purposes. It may take one week to get reply. Or Contact [Tao Yu](https://ytrock.com/) at (ytrock@126.com). More details about MMVP Dataset [here](https://github.com/Metaverse-AI-Lab-THU/MMVP-Dataset/tree/visualizing).
3. We use [RTM-pose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) for 2d keypoints detection and [Cliff](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) for pose initialization.
4. for one sequence, the results of [RTM-pose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) should be saved under `input/sub_ids/seq_name/`.
5. Our optimization includes 3 stages: `init_shape`, `init_pose`, `tracking`. `init_shape` stage is applied for shape parameter initialization. `init_pose` stage is applied for pose estimation for the first frame in one sequence. `tracking` stage is applied for pose estimation for other frames in one sequence.
Users should run optimization follow the order: `init_shape`, `init_pose` by changing the parameter `fitting stage` in `configs/fit_smpl_rgbd.yaml`. Stage `tracking` will run automatically if you choose `init_pose` in the yaml file.
6. For `init_pose`, we use [Cliff](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) to get pose initial value for depth alignment. The pose initial value will be saved under `init_data_dir/`.
7. Users could make optimization for MMVP Dataset with the command below:
```
sh run.sh
```
You could change frame info in `configs/visualizing.yaml`.  
`basdir` is the root where you put MMVP Dataset.  
`dataset` is fixed as `20230422`.  
`sub_ids` could be `S01...S12`  
`seq_name` represents the seq under the `sub_ids` you select.  
`essential_root` represents the path you put the essential files for optimizing.  
`init_data_dir` represents the path you .  

Noticing that `frame_idx` may not cover all frames and could be selected in `annotations/smpl_pose`.

# Citation


# License
