# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *



def goal_pred_loss(grasp_pred, goal_batch ):
    """
    PM loss for grasp pose detection
    """
    grasp_pcs = transform_control_points( grasp_pred, grasp_pred.shape[0], device="cuda", rotz=True )
    grasp_pcs_gt = transform_control_points( goal_batch, goal_batch.shape[0], device="cuda", rotz=True )
    return torch.mean(torch.abs(grasp_pcs - grasp_pcs_gt).sum(-1))

def traj_action_loss(agent, pi, traj_action_batch, cont_expert_mask ):
    """
    PM loss for traj action
    """
    gt_act_pt = control_points_from_rot_and_trans(traj_action_batch[:,3:], traj_action_batch[:,:3], device='cuda')
    pi_act_pt = control_points_from_rot_and_trans(pi[:,3:], pi[:,:3], device='cuda')
    return torch.mean(torch.abs(gt_act_pt[cont_expert_mask] - pi_act_pt[cont_expert_mask]).sum(-1))

def traj_latent_loss(agent, pred_latent, target_latent):
    """
    L2 latent reconstruction loss
    """
    return F.mse_loss(pred_latent, target_latent)

def kl_loss(extra_pred, kl_scale):
    """
    KL with unit gaussian
    """
    mu, log_sigma = extra_pred
    return  kl_scale * torch.mean( -.5 * torch.sum(1. + log_sigma - mu**2 - torch.exp(log_sigma), dim=-1))
