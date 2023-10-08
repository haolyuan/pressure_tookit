# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import configargparse

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser

    description = 'PyTorch implementation of SMPLifyXMC'

    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='Mocap')#SMPLifyX
    # config file
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')

    # input and output files
    parser.add_argument('--dataset', default='PressureDataset', type=str,
                        help='The name of the dataset that will be used')
    parser.add_argument('--basdir', default='E:/dataset', type=str,
                        help='Base dir')
    parser.add_argument('--sub_ids', default='S12', type=str,
                        help='Subject ids')
    parser.add_argument('--seq_name', default='MoCap_20230422_145333', type=str,
                        help='Sequence name')
    parser.add_argument('--output_dir',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--color_size',
                        default=[1280,720], type=int,
                        nargs='*',
                        help='Color size')
    parser.add_argument('--depth_size',
                        default=[640,576], type=int,
                        nargs='*',
                        help='Color size')
    parser.add_argument('--frame_range',
                        default=[0,10], type=int,
                        nargs='*',
                        help='frame range')
    # body model model
    parser.add_argument('--model_folder',
                        default='../../bodyModels/smpl',
                        type=str,
                        help='The directory where the models are stored.')
    parser.add_argument('--num_shape_comps', default=10, type=int,
                        help='The number of betas.')
    parser.add_argument('--model_gender', default='neutral', type=str,
                        help='The number of betas.')

    # fitting / losses   init_model
    parser.add_argument('--init_model',
                        default=False,
                        action='store_true',
                        help='init or tracking')
    parser.add_argument('--shape_weights', default=0.01,
                        type=float, help='The weights of the Shape regularizer')
    parser.add_argument('--depth_weights', default=1.0,
                        type=float, help='The weights of the Depth')
    parser.add_argument('--keypoint_weights', default=0.01,
                        type=float, help='The weights of the Keypoints')
    parser.add_argument('--penetrate_weights', default=10.,
                        type=float, help='The weights of the Keypoints')
    parser.add_argument('--contact_weights', default=1.0,
                        type=float, help='The weights of the Keypoints')
    parser.add_argument('--maxiters', type=int, default=601,
                        help='The maximum iterations for the optimization')

    # # segments
    # parser.add_argument('--part_segm_fn', default='', type=str,
    #                     help='The file with the part segmentation for the' +
    #                     ' faces of the model')
    #
    # # camera
    # parser.add_argument('--use_calibration', default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Use camera intrinsics if available.')
    # parser.add_argument('--camera_type', type=str, default='persp',
    #                     choices=['persp'],
    #                     help='The type of camera used')
    # parser.add_argument('--focal_length',
    #                     default=3577,
    #                     type=float,
    #                     help='Value of focal length.')
    #
    # # initialize fitting
    # parser.add_argument('--kpts_sel_strategy',
    #                     default='confidence',
    #                     choices=['confidence', 'fixed'],
    #                     help='The keypoint selection strategy used to find the \
    #                     best initial global orientation. Confidence selects based on the \
    #                     OpenPose confidence score and fixed uses a fixed set of major body keypoints.')
    # parser.add_argument('--go_sel_strategy',
    #                     default='all',
    #                     choices=['all', 'frontal'],
    #                     help='From which global orientations the best initial global orientation \
    #                         is selected. Frontal only searches in frontal presented global orientations, \
    #                         e.g. used in MTP data. All seaches in 360 y-axis rotation, e.g. for multi-view.')
    #
    # # fitting / losses
    # parser.add_argument('--joints_to_ign', default=-1, type=int,
    #                     nargs='*',
    #                     help='Indices of joints to be ignored')
    # parser.add_argument('--use_joints_conf', default=True,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Use the confidence scores for the optimization')
    # parser.add_argument('--num_gaussians',
    #                     default=8,
    #                     type=int,
    #                     help='The number of gaussian for the Pose Mixture' +
    #                     ' Prior.')
    # parser.add_argument('--left_hand_prior_type', default='mog', type=str,
    #                     choices=['mog', 'l2', 'None'],
    #                     help='The type of prior that will be used to' +
    #                     ' regularize the optimization of the pose of the' +
    #                     ' left hand. Can be a Mixture of' +
    #                     ' Gaussians (mog)')
    # parser.add_argument('--right_hand_prior_type', default='mog', type=str,
    #                     choices=['mog', 'l2', 'None'],
    #                     help='The type of prior that will be used to' +
    #                     ' regularize the optimization of the pose of the' +
    #                     ' right hand. Can be a Mixture of' +
    #                     ' Gaussians (mog)')
    # parser.add_argument('--jaw_prior_type', default='l2', type=str,
    #                     choices=['l2', 'None'],
    #                     help='The type of prior that will be used to' +
    #                     ' regularize the optimization of the pose of the' +
    #                     ' jaw.')
    # parser.add_argument('--rho',
    #                     default=100,
    #                     type=float,
    #                     help='Value of constant of robust loss')
    # parser.add_argument('--body_pose_prior_weights',
    #                     default=[4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
    #                     nargs='*',
    #                     type=float,
    #                     help='The weights of the body pose regularizer')
    # parser.add_argument('--shape_weights',
    #                     default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
    #                     type=float, nargs='*',
    #                     help='The weights of the Shape regularizer')
    # parser.add_argument('--expr_weights',
    #                     default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
    #                     type=float, nargs='*',
    #                     help='The weights of the Expressions regularizer')
    # parser.add_argument('--face_joints_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weights for the facial keypoints' +
    #                     ' for each stage of the optimization')
    # parser.add_argument('--hand_joints_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0],
    #                     type=float, nargs='*',
    #                     help='The weights for the 2D joint error of the hands')
    # parser.add_argument('--jaw_pose_prior_weights',
    #                     nargs='*',
    #                     help='The weights of the pose regularizer of the' +
    #                     ' hands')
    # parser.add_argument('--hand_pose_prior_weights',
    #                     default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
    #                     type=float, nargs='*',
    #                     help='The weights of the pose regularizer of the' +
    #                     ' hands')
    # parser.add_argument('--scopti_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the collision term')
    # parser.add_argument('--data_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the collision term')
    # parser.add_argument('--camera_rot_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the collision term')
    # parser.add_argument('--body_global_orient_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the collision term')
    # parser.add_argument('--focal_length_weights',
    #                     default=[0.0, 0.0, 0.0, 2.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the collision term')
    # parser.add_argument('--mimicked_contact_weights',
    #                     default=[0.0, 0.0, 0.0, 0.0, 0.0], type=float,
    #                     nargs='*',
    #                     help='The weight for the sliding contact term')
    # parser.add_argument('--ign_part_pairs', default=None,
    #                     nargs='*', type=str,
    #                     help='Pairs of parts whose collisions will be ignored')
    # parser.add_argument('--use_hands', default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Use the hand keypoints in the SMPL' +
    #                     'optimization process')
    # parser.add_argument('--use_face', default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Use the facial keypoints in the optimization' +
    #                     ' process')
    # parser.add_argument('--use_face_contour', default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Use the dynamic contours of the face')
    # parser.add_argument('--use_contact',
    #                     default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Whether to use the sliding contact term')
    # parser.add_argument('--geo_thres', default = 0.3, type=float,
    #                     help='geodesic threshold for self-contact')
    # parser.add_argument('--eucl_thres', default = 0.02, type=float,
    #                     help='euclidean threshold for self-contact')
    # parser.add_argument('--gen_sc_inside_weight', default=0.5, type=float,
    #                     help='Pushing term weight')
    # parser.add_argument('--gen_sc_outside_weight',default=0.0, type=float,
    #                     help='Outside term weight')
    # parser.add_argument('--gen_sc_contact_weight', default=0.5, type=float,
    #                     help='Pulling term weight')
    # parser.add_argument('--use_presented_hand_pose',
    #                     default=False,
    #                     type=lambda x: x.lower() in ['true', '1'],
    #                     help='Whether to use the left and right hand poses from the presetend pose')
    #
    #
    # # optimizer flags
    # parser.add_argument('--optim_type', type=str, default='adam',
    #                     help='The optimizer used')
    # parser.add_argument('--lr', type=float, default=1e-6,
    #                     help='The learning rate for the algorithm')
    # parser.add_argument('--gtol', type=float, default=1e-8,
    #                     help='The tolerance threshold for the gradient')
    # parser.add_argument('--ftol', type=float, default=2e-9,
    #                     help='The tolerance threshold for the function')

    #
    # # cluster flags
    # parser.add_argument('--ds_start_idx', type=int, default=0,
    #                     help='set index at which to start processing dataset')
    # parser.add_argument('--cluster_bs', type=int, default=None,
    #                     help='number of dataset objects to process')
    args = parser.parse_args()


    args_dict = vars(args)

    return args_dict
