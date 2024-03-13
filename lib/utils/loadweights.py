# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


""" use smplify-xmc framework"""

defaultweights = {
        'depth_weights':[60.0],
        'shape_weights':[0.01],
        'keypoint_weights':[0.01],
        'penetrate_weights':[15.0],
        'limb_weights':[0.0],
        'gmm_weights':[0.01],
        'tfoot_weights':[10.0],
        'tpose_weights':[1.0]
}

def load_weights(stage):

    weights = defaultweights

    # TODO: change correct weights according to different stage
    depth_weight = weights['depth_weights']
    shape_weights = weights['shape_weights']
    keypoint_weights = weights['keypoint_weights']
    penetrate_weights = weights['penetrate_weights']
    limb_weights = weights['limb_weights']
    gmm_weights = weights['gmm_weights']
    tfoot_weights = weights['tfoot_weights']
    tpose_weights = weights['tpose_weights']

    return create_weights_dict(
        depth_weight, shape_weights, keypoint_weights,penetrate_weights,
        limb_weights, gmm_weights, tfoot_weights, tpose_weights,
        stage
    )

def create_weights_dict(
    depth_weight, shape_weights, keypoint_weights, penetrate_weights,
    limb_weights, gmm_weights, tfoot_weights, tpose_weights,
    stage
):
    # Weights used for common fitting
    opt_weights_dict = {'depth_weight': depth_weight,
                        'keypoint_weights': keypoint_weights,
                        'penetrate_weights':penetrate_weights                 
                        }    
    if stage == 'init_shape':
        opt_weights_dict['shape_weights'] = shape_weights
        opt_weights_dict['limb_weights'] = limb_weights
    elif stage == 'init_pose':
        opt_weights_dict['limb_weights'] = limb_weights
        opt_weights_dict['gmm_weights'] = gmm_weights
        opt_weights_dict['tpose_weights'] = tpose_weights
        
    elif stage == 'tracking':
        opt_weights_dict['gmm_weights'] = gmm_weights
        opt_weights_dict['limb_weights'] = limb_weights
        opt_weights_dict['tfoot_weights'] = tfoot_weights
        opt_weights_dict['tpose_weights'] = tpose_weights

    keys = opt_weights_dict.keys()

    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    return opt_weights
            