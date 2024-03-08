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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

class FittingMonitor(object):
    def __init__(self,
                 maxiters=100, 
                 ftol=2e-09, 
                 gtol=1e-05,
                 stage='init_shape'):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        
        self.stage = stage
        
    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass        

    def run_fitting(self, optimizer, closure, params, body_model):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
            Returns
            -------
                loss: float
                The final loss value
        '''
        prev_loss = None
        for n in range(self.maxiters):
            old_params = [x for x in body_model.named_parameters()]
            
            # curr_time = time.time()
            loss = optimizer.step(closure)
            # print(time.time() - curr_time)
            # import pdb;pdb.set_trace()

            if torch.isnan(loss).sum() > 0 or torch.isinf(loss).sum() > 0 or loss is None:
                print('Inf or NaN loss value, rolling back to old params!')
                old_params = dict([(x[0], x[1].data) for x in old_params])
                body_model.reset_params(**old_params)
                break

            # if n > 0 and prev_loss is not None and self.ftol > 0:
            #     loss_rel_change = rel_change(prev_loss, loss.item())

            #     if loss_rel_change <= self.ftol:
            #         print('break caused by loss rising')
            #         break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                
                print('break caused by others')
                        
                break

            prev_loss = loss.item()
            print(prev_loss, n)

        return prev_loss


    def create_fitting_closure(self,
                               optimizer, body_model,
                               camera, joint_mapper, joint_weights,
                               gt_joints, gt_depth_nmap, gt_depth_vmap, gt_contact,
                               loss=None,
                               create_graph=False):
        
        def fitting_func(backward=True):
            stop = False
            # we might encounter nan values in the optimization. In this case,
            # stop iterations here. This necessart, as lbfgs for example
            # perfroms multiple optimization steps in a single optimizer step.        
            for param in body_model.parameters():
                if np.any(np.isnan(param.data.cpu().numpy())) or \
                   np.any(np.isinf(param.data.cpu().numpy())):
                    print('nan in model')
                    backward = False
                    total_loss = torch.tensor(float('inf'))        
            if not stop:
                if backward:
                    optimizer.zero_grad()
                
                
                # update model
                if self.stage == 'init_shape':
                    body_model.update_shape()
                    body_model.init_plane()
                    body_model_output = body_model.update_pose()
                else:
                    # TODO:where to init model in other fitting stage?
                    body_model_output = body_model.update_pose()
                    
                
                total_loss = loss(
                                 body_model_output=body_model_output,
                                 camera=camera,
                                 joint_weights=joint_weights,
                                 joint_mapper=joint_mapper,
                                 gt_joints=gt_joints,
                                 gt_depth_nmap=gt_depth_nmap,
                                 gt_depth_vmap=gt_depth_vmap,
                                 gt_contact=gt_contact)

                if torch.isnan(total_loss).sum() > 0 or torch.isinf(total_loss).sum() > 0:
                    print('lbfgs - Inf or NaN loss value, skip backward pass!')
                    # skip backward step in this case
                else:
                    if body_model_output.vertices is not None:
                        loss.previousverts = body_model_output.vertices.detach().clone()
                    total_loss.backward(create_graph=create_graph)

                    self.steps += 1
            # print('fitting closure loss ', total_loss, self.steps)
            return total_loss
                                            
        return fitting_func
