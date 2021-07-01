# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *
from core.agent import Agent
from core.traj import TrajAgent
from core.loss import *

class BC(TrajAgent):
    def __init__(self, num_inputs, action_space, args):
        super(BC, self).__init__(num_inputs, action_space, args, name='BC')

    def load_weight(self, weights):
        self.policy.load_state_dict(weights[0])
        self.traj_feature_extractor.load_state_dict(weights[1])
        self.traj_feature_sampler.load_state_dict(weights[2])
        self.state_feature_extractor.load_state_dict(weights[3])

    def get_weight(self):
        return  [ self.policy.state_dict(),
                  self.traj_feature_extractor.state_dict(),
                  self.traj_feature_sampler.state_dict(),
                  self.state_feature_extractor.state_dict() ]

    def extract_feature(self, image_batch, point_batch,
                            action_batch=None, goal_batch=None, time_batch=None,  vis=False,
                            value=False, repeat=False, separate=False, traj_poses=None,
                            curr_joint=None, goal_point_state=None, traj_point_state=None,
                            traj_integer_time=None, joint_traj_state=None, train=False, no_action=False ):
        """
        extract policy state feature
        """
        use_traj = self.full_traj_embedding and traj_integer_time is not None and joint_traj_state is not None
        time_input  = traj_integer_time if use_traj else time_batch
        joint_input = joint_traj_state  if use_traj else curr_joint
        point_input = traj_point_state  if use_traj else point_batch
        point_batch = preprocess_points(self, point_input, joint_input, time_input)

        feature = self.unpack_batch(image_batch,  point_batch,  repeat=repeat )
        if self.use_time: feature[0] = torch.cat((feature[0], time_input[:,None]), dim=1)
        self.state_policy_feat_th = feature[0].clone().detach()
        self.state_policy_feat = self.state_policy_feat_th.cpu().numpy()

        if  self.traj_goal_mutual_conditioned:
            feature[0] = torch.cat((feature[0], self.traj_feat), dim=1)
        return feature


    def update_parameters(self, batch_data, updates, k, test=False):
        """ train agent """
        self.set_mode(False)
        self.prepare_data(batch_data)
        self.train_traj_net()

        state_batch, extra_feat_pred =  self.extract_feature(
                                        self.image_state_batch,
                                        self.point_state_batch,
                                        goal_batch=self.goal_batch,
                                        traj_point_state=self.traj_point_state_batch,
                                        time_batch=self.time_batch,
                                        curr_joint=self.curr_joint[:, :7],
                                        joint_traj_state=self.traj_joint_batch[:, :7],
                                        traj_integer_time=self.traj_integer_time_batch,
                                        train=True)
        self.pi, self.log_pi, _, self.aux_pred = self.policy.sample(state_batch)
        loss = self.compute_loss()

        self.optimize(loss, self.update_step)
        if self.re_sampler_step: # separately update sampler
            self.sampler_update_parameters()
        self.update_step += 1
        self.log_stat()
        return {k: float(getattr(self, k)) for k in self.loss_info }

    def sampler_update_parameters(self):
        """
        Second pass for sampler joint training
        """
        state_feat = torch.cat((self.state_policy_feat_th, self.sampler_traj_feat), dim=1)
        self.pi, self.log_pi, _, self.aux_pred = self.policy.sample(state_feat)
        self.sampler_gaussian  = (self.sampler_latent_mean, self.sampler_latent_logstd)

        # recons
        self.traj_latent_loss = traj_latent_loss(self,  self.sampler_traj_feat[self.target_expert_mask],
                                                        self.traj_feat_target[self.target_expert_mask].detach())

        # kl
        self.kl_loss = kl_loss(self.sampler_gaussian, self.kl_scale)

        # grasp and bc
        self.sampler_grasp_aux_loss = goal_pred_loss(self.aux_pred[self.target_goal_reward_mask, :7], self.target_grasp_batch[self.target_goal_reward_mask, :7] )
        self.sampler_bc_loss = traj_action_loss(self, self.pi, self.traj_expert_action_batch, self.target_expert_mask)

        self.traj_feature_sampler_opt.zero_grad() # only update sampler
        self.state_feat_encoder_optim.zero_grad()
        self.policy_optim.zero_grad()
        total_loss = self.kl_loss + self.traj_latent_loss + self.sampler_bc_loss + self.sampler_grasp_aux_loss
        total_loss.backward()
        self.traj_feature_sampler_opt.step()