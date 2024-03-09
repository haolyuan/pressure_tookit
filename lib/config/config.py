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

    description = 'PyTorch implementation of MMVP GT generation'

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
    parser.add_argument('--label_output_dir', default='', type=str,
                        help='pose/mesh/shape data dir')
    parser.add_argument('--sub_ids', default='S01', type=str,
                        help='Subject ids')
    parser.add_argument('--seq_name', default='MoCap_20230422_145333', type=str,
                        help='Sequence name')
    parser.add_argument('--output_dir',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--init_data_dir',
                        default=None,
                        type=str,
                        help='init pose for fitting, used for stage init_pose/tracking')
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
    parser.add_argument('--essential_root',
                        type=str,
                        default='/data/nas_data/yuanhaolei/essential_files/essentials')
    # body model model
    # parser.add_argument('--model_folder',
    #                     default='essentials/bodyModels/smpl',
    #                     type=str,
    #                     help='The directory where the models are stored.')
    parser.add_argument('--num_shape_comps', default=10, type=int,
                        help='The number of betas.')
    parser.add_argument('--model_gender', default='neutral', type=str,
                        help='The gender of item.')

    # fitting / losses
    parser.add_argument('--start_idx',
                        default=0,
                        help='which frame data will be used in init stage')
    parser.add_argument('--end_idx',
                        default=-1,
                        help='end frame used in tracking stage')    
    parser.add_argument('--shape_weights', default=0.01,
                        type=float, help='The weights of the Shape regularizer')
    parser.add_argument('--depth_weights', default=1.0,
                        type=float, help='The weights of the Depth')
    parser.add_argument('--keypoint_weights', default=0.01,
                        type=float, help='The weights of the Keypoints')
    parser.add_argument('--penetrate_weights', default=10.,
                        type=float, help='The weights of the penetrate')
    parser.add_argument('--contact_weights', default=1.0,
                        type=float, help='The weights of contact')
    parser.add_argument('--limb_weights', default=1.0,
                        type=float, help='The weights of contact')
    parser.add_argument('--gmm_weights', default=1.0,
                        type=float, help='The weights of contact')
    parser.add_argument('--tpose_weights', default=1.0,
                        type=float, help='The weights of contact')
    parser.add_argument('--tfoot_weights', default=1.0,
                        type=float, help='The weights of contact')    
    parser.add_argument('--maxiters', type=int, default=100,
                        help='The maximum iterations for the optimization')

    parser.add_argument('--fitting_stage', type=str, default='init_shape',
                        help='current stage running')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict
