# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *

from core import networks
from core.traj import TrajAgent
from core.loss import *

class DDPG(TrajAgent):
    def __init__(self, num_inputs, action_space, args):
        super(DDPG, self).__init__(num_inputs, action_space, args, name='DDPG')
        self.action_dim = self.action_space.shape[0]
        self.critic_value_dim = 0 if self.sa_channel_concat else self.action_dim
        self.value_loss_func = huber_loss
        self.critic_num_input = self.num_inputs
        self.critic, self.critic_optim, self.critic_scheduler, self.critic_target = get_critic(self)

    def load_weight(self, weights):
        self.policy.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        self.traj_feature_extractor.load_state_dict(weights[2])
        self.state_feature_extractor.load_state_dict(weights[3])

    def get_weight(self):
        return [self.policy.state_dict(),
                self.critic.state_dict(),
                self.traj_feature_extractor.state_dict(),
                self.state_feature_extractor.state_dict() ]

    def get_mix_ratio(self, update_step):
        """
        Get a mixed schedule for supervised learning and RL
        """
        idx = int((self.update_step > np.array(self.mix_milestones)).sum())
        mix_policy_ratio = get_valid_index(self.mix_policy_ratio_list, idx)
        mix_policy_ratio = min(mix_policy_ratio, self.ddpg_coefficients[4])
        mix_value_ratio  = get_valid_index(self.mix_value_ratio_list, idx)
        mix_value_ratio  = min(mix_value_ratio, self.ddpg_coefficients[3])
        return mix_value_ratio, mix_policy_ratio

    def extract_feature(self, image_batch, point_batch,
                            action_batch=None,  goal_batch=None, time_batch=None,
                            vis=False, value=False, repeat=False,
                            train=True, separate=False, curr_joint=None, traj_latent=None,
                            traj_point_state=None, traj_integer_time=None,  traj_action=None,
                            joint_traj_state=None, next_state=False, traj_time_batch=None):
        """
        extract features for policy learning
        """
        if curr_joint is not None: curr_joint = curr_joint[:, :7]
        feature_2  = value and not self.shared_feature
        use_traj   = False
        time_input   = traj_integer_time  if use_traj else  time_batch
        joint_input  = joint_traj_state   if use_traj else  curr_joint
        point_input  = traj_point_state   if use_traj else  point_batch[:,:,:1030]
        action_input = traj_action        if use_traj else  action_batch

        # clip time
        time_input = torch.clamp(time_input, 0., 25.)

        if self.sa_channel_concat and value:
            point_input = concat_state_action_channelwise(point_input, action_input)

        feature = self.unpack_batch(image_batch, point_input, gt_goal=goal_batch, val=feature_2, repeat=repeat)
        if self.use_time:
            feature[0] = torch.cat((feature[0], time_input[:,None]), dim=1)

        if not self.sa_channel_concat and value:
            feature[0] = torch.cat((feature[0], action_input), dim=1)
        self.state_policy_feat = feature[0].detach().cpu().numpy()
        return feature

    def target_value(self):
        """
        compute target value
        """
        next_time_batch = self.time_batch - 1
        next_traj_time_batch = self.traj_integer_time_batch - 1
        reward_batch = self.traj_reward_batch if self.full_traj_embedding else self.reward_batch
        mask_batch = self.traj_mask_batch if self.full_traj_embedding else self.mask_batch

        with torch.no_grad():
            next_state_batch, _  = self.extract_feature(self.next_image_state_batch,
                                                        self.next_point_state_batch,
                                                        action_batch=self.next_action_batch,
                                                        goal_batch=self.next_goal_batch,
                                                        traj_point_state=self.next_traj_point_state_batch,
                                                        time_batch=next_time_batch,
                                                        curr_joint=self.next_joint_batch,
                                                        joint_traj_state=self.traj_joint_batch,
                                                        traj_integer_time=next_traj_time_batch,
                                                        traj_action=self.next_traj_action_batch,
                                                        separate=True,
                                                        vis=False,
                                                        next_state=True,
                                                        value=False)

            next_action_mean, _, _, _ = self.policy_target.sample(next_state_batch)
            idx = int((self.update_step > np.array(self.mix_milestones)).sum())
            noise_scale = self.action_noise * get_valid_index(self.noise_ratio_list, idx)
            noise_delta = get_noise_delta(next_action_mean, noise_scale, self.noise_type)
            noise_delta[:, :3] = torch.clamp(noise_delta[:, :3], -0.01, 0.01)
            next_action_mean = next_action_mean + noise_delta
            if self.sa_channel_concat or not self.shared_feature:
                next_target_state_batch, _ = self.extract_feature( self.next_image_state_batch, self.next_point_state_batch, next_action_mean,
                                                                   self.next_goal_batch, next_time_batch, value=True, curr_joint=self.next_joint_batch)
            else:
                next_target_state_batch = torch.cat((next_state_batch, next_action_mean), dim=-1)

            min_qf_next_target = self.state_action_value(next_target_state_batch, next_action_mean, target=True)
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target
            return next_q_value

    def state_action_value(self, state_batch, action_batch, target=False, return_qf=False):
        """ compute state action value """
        critic = self.critic_target if target else self.critic
        qf1, qf2, critic_aux = critic(state_batch, None)
        min_qf = torch.min(qf1, qf2)
        min_qf = min_qf.squeeze(-1)

        if return_qf:
            return qf1.squeeze(-1), qf2.squeeze(-1), critic_aux
        return min_qf

    def critic_optimize(self):
        """
        optimize critic and feature gradient
        """
        self.critic_optim.zero_grad()
        self.state_feat_val_encoder_optim.zero_grad()
        critic_loss = sum([getattr(self, name) for name in get_loss_info_dict().keys() \
                            if name.endswith('loss') and name.startswith('critic')])
        if self.sa_channel_concat or not self.shared_feature:
            critic_loss.backward()
        else:
            critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        if self.train_value_feature: self.state_feat_val_encoder_optim.step()
        self.critic_optim.step()

    def compute_critic_loss(self, value_feat):
        """
        compute one step bellman error
        """
        self.next_q_value = self.target_value()
        self.qf1, self.qf2, self.critic_grasp_aux = self.state_action_value(value_feat, self.action_batch, return_qf=True)
        self.critic_loss = F.mse_loss(self.qf1.view(-1), self.next_q_value) + \
                           F.mse_loss(self.qf2.view(-1), self.next_q_value)

        if self.critic_aux:
            self.critic_grasp_aux_loss += goal_pred_loss(self.critic_grasp_aux[self.goal_reward_mask, :7], self.goal_batch[self.goal_reward_mask])

    def update_parameters(self, batch_data, updates, k, test=False):
        """ update agent parameters """
        self.mix_value_ratio, self.mix_policy_ratio = self.get_mix_ratio(self.update_step)
        self.set_mode(test)
        self.prepare_data(batch_data)

        if self.train_traj_feature :
            self.train_traj_net()

        # value
        if self.policy_update_gap > 0:
            value_feat, _ = self.extract_feature(self.image_state_batch,
                                                self.point_state_batch,
                                                action_batch=self.action_batch,
                                                goal_batch=self.goal_batch,
                                                traj_point_state=self.traj_point_state_batch,
                                                time_batch=self.time_batch,
                                                curr_joint=self.curr_joint,
                                                joint_traj_state=self.traj_joint_batch,
                                                traj_integer_time=self.traj_integer_time_batch,
                                                traj_time_batch=self.traj_time_batch,
                                                traj_action=self.traj_action_batch,
                                                value=True)
            self.compute_critic_loss(value_feat)
            self.critic_optimize()

        # policy
        if self.sa_channel_concat or not self.shared_feature:
            policy_feat, extra_feat_pred = self.extract_feature(self.image_state_batch,
                                                            self.point_state_batch,
                                                            goal_batch=self.goal_batch,
                                                            traj_point_state=self.traj_point_state_batch,
                                                            time_batch=self.time_batch,
                                                            curr_joint=self.curr_joint,
                                                            separate=True,
                                                            joint_traj_state=self.traj_joint_batch,
                                                            traj_integer_time=self.traj_integer_time_batch,
                                                            traj_time_batch=self.traj_time_batch)
        else:
            policy_feat = value_feat[..., :-self.action_dim]

        self.pi, _, _, self.aux_pred = self.policy.sample(policy_feat)

        # actor critic
        if self.has_critic and (self.update_step % self.policy_update_gap == 0) and self.policy_update_gap > 0:
            if self.sa_channel_concat or not self.shared_feature:
                value_pi_feat, _  = self.extract_feature(self.image_state_batch, self.point_state_batch,
                                                         self.pi, self.goal_batch, self.time_batch,
                                                         value=True, curr_joint=self.curr_joint)
            else:
                value_pi_feat = torch.cat((policy_feat, self.pi), dim=-1)
            self.qf1_pi, self.qf2_pi, critic_aux = self.state_action_value(value_pi_feat, self.pi, return_qf=True)
            self.qf1_pi = self.qf1_pi[~self.expert_reward_mask]
            self.qf2_pi = self.qf2_pi[~self.expert_reward_mask]
            self.actor_critic_loss = -self.mix_policy_ratio * self.qf1_pi.mean()

        loss = self.compute_loss()
        self.optimize(loss, self.update_step)
        self.update_step += 1

        # log
        self.log_stat()
        return {k: float(getattr(self, k)) for k in get_loss_info_dict().keys()}

