# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *
from torch.optim import Adam


from core import networks
from core.traj import TrajAgent
from core.loss import *

class DQN_HRL(TrajAgent):
    def __init__(self, num_inputs, action_space, args):
        super(DQN_HRL, self).__init__(num_inputs, action_space, args, name='DQN_HRL')
        self.action_dim = self.action_space.shape[0]
        self.critic_num_input = self.feature_input_dim + 1
        self.critic_value_dim = self.policy_traj_latent_size

        self.value_loss_func = F.smooth_l1_loss
        self.critic, self.critic_optim, self.critic_scheduler, self.critic_target = get_critic(self)

    def load_weight(self, weights):
        self.policy.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        self.traj_feature_sampler.load_state_dict(weights[2])
        self.traj_feature_extractor.load_state_dict(weights[3])
        self.state_feature_extractor.load_state_dict(weights[4])

    def get_weight(self):
        return [self.policy.state_dict(),
                self.critic.state_dict(),
                self.traj_feature_sampler.state_dict(),
                self.traj_feature_extractor.state_dict(),
                self.state_feature_extractor.state_dict()]

    def greedy_action(self):
        """ determine if use max score traj or random sample """
        return  np.random.uniform() < self.epsilon_greedy_list[0]

    @torch.no_grad()
    def compute_critic_value(self, img_state, point_state, timestep, curr_joint_th, goal_state):
        """
        compute score on state and traj feature and return argmax
        """
        feature = self.extract_feature(
                                    img_state,
                                    point_state,
                                    goal_batch=goal_state,
                                    traj_goal_batch=goal_state,
                                    time_batch=timestep,
                                    value=True,
                                    train=False,
                                    curr_joint=curr_joint_th)[0]

        qf1_pi, qf2_pi, critic_aux = self.state_action_value(feature, None, return_qf=True)
        self.q_min =  torch.min(qf1_pi, qf2_pi)
        self.q_stat = [self.q_min.min(), self.q_min.mean(), self.q_min.max()]
        opt_idx = torch.argmax(self.q_min)
        self.traj_feat = self.traj_feat[[opt_idx]].view(1, -1)
        self.sampler_traj_feat = self.sampler_traj_feat[[opt_idx]].view(1, -1)
        self.gaddpg_pred = torch.sigmoid(critic_aux[opt_idx, -1])

        return opt_idx

    @torch.no_grad()
    def critic_select_action(
        self, state, remain_timestep, curr_joint, vis=False):
        """
        Sample traj and select based on critic
        """
        self.set_mode(True)
        sample_num = self.test_traj_num if self.greedy_action() or self.test_mode else 1
        curr_joint_th = torch.cuda.FloatTensor([curr_joint.flatten()])[:, :7].repeat((sample_num, 1))
        img_state = torch.cuda.FloatTensor(state[0][1])[None]
        point_state = torch.cuda.FloatTensor(state[0][0])[None].repeat((sample_num, 1, 1))
        self.timestep = torch.Tensor([remain_timestep]).float().cuda()
        traj = self.select_traj(img_state,
                                point_state,
                                vis=vis,
                                remain_timestep=remain_timestep,
                                curr_joint=curr_joint_th)
        timestep = torch.Tensor([remain_timestep]).float().cuda().repeat(sample_num)

        self.timestep = timestep
        policy_feature = self.extract_feature(  img_state,
                                                point_state,
                                                time_batch=timestep,
                                                value=False,
                                                train=False,
                                                repeat=True,
                                                curr_joint=curr_joint_th)[0]
        actions = self.policy.sample(policy_feature)

        # select traj from MPC
        opt_idx = self.compute_critic_value( img_state, point_state, timestep, curr_joint_th, None )
        if traj is not None: #  visualizing the selected trajectory
            traj[0], traj[opt_idx] = traj[opt_idx], traj[0]

        action = actions[0].detach().cpu().numpy()[opt_idx]
        extra_pred = actions[1].detach().cpu().numpy()[opt_idx]
        action_sample = actions[2].detach().cpu().numpy()[opt_idx]
        extra_pred = self.gaddpg_pred.detach().cpu().numpy()
        aux_pred = actions[3].detach().cpu().numpy()[opt_idx]
        return action, traj, extra_pred, aux_pred

    def extract_feature(self, image_batch, point_batch,
                              action_batch=None,  goal_batch=None, time_batch=None,
                              vis=False, value=False, repeat=False, traj_goal_batch=None,
                              train=True, separate=False, curr_joint=None, traj_latent=None,
                              traj_point_state=None, traj_integer_time=None,  traj_action=None,
                              joint_traj_state=None, next_state=False, traj_time_batch=None,
                              use_offline_latent=True, no_action=False ):
        """
        extract features for policy learning
        """
        curr_joint = curr_joint[:, :7]
        use_traj   = self.full_traj_embedding and joint_traj_state is not None
        action_input = traj_latent if traj_latent is not None else self.traj_feat
        point_batch = preprocess_points(self, point_batch, curr_joint, time_batch)

        if not value:
            feature = self.unpack_batch(image_batch, point_batch,  val=False )
            feature[0] = torch.cat((feature[0], time_batch[:,None]), dim=1)
            self.state_policy_feat = feature[0].detach().cpu().numpy()
            if self.traj_feat is not None:
                feature[0] = torch.cat((feature[0], self.traj_feat), dim=1)
            return feature

        feature = self.unpack_batch(image_batch, point_batch, val=True, repeat=repeat)
        feature[0] = torch.cat((feature[0], time_batch[:,None], action_input), dim=1)
        return feature

    def target_value(self):
        """
        compute target value
        """
        self.dataset_traj_latent_head = None
        sampler_state_input_batch  = preprocess_points(self, self.next_point_state_batch, self.next_joint_batch, self.next_time_batch, append_pc_time=True)
        self.train_traj_idx_batch  = torch.arange(len(sampler_state_input_batch)).cuda().long().repeat(self.dqn_sample_num)
        self.traj_feature_sampler.eval()
        self.traj_feature_extractor.eval()
        with torch.no_grad():
            self.autoencode(sampler_state_input_batch, False )
            traj_latent =  self.sampler_traj_feat
            state_input_batch  = preprocess_points(self, self.next_point_state_batch, self.next_joint_batch, self.next_time_batch )
            state_input_batch  = state_input_batch.repeat(self.dqn_sample_num, 1, 1)
            next_time_batch    = self.next_time_batch.repeat(self.dqn_sample_num )
            next_joint_batch   = self.next_joint_batch.repeat(self.dqn_sample_num,  1)
            next_state_batch, _  = self.extract_feature(self.next_image_state_batch,
                                                        state_input_batch,
                                                        time_batch=next_time_batch,
                                                        curr_joint=next_joint_batch,
                                                        traj_latent=traj_latent,
                                                        next_state=True,
                                                        value=True)

            # use the target network to pick the max action
            next_target_state_batch = next_state_batch
            min_qf_next_target = self.state_action_value(next_target_state_batch, None, target=True)
            min_qf_next_target = min_qf_next_target.view(self.dqn_sample_num, -1).max(0)[0]
            next_q_value = self.target_reward_batch + (1 - self.target_mask_batch) * self.gamma * min_qf_next_target
            next_q_value = self.mix_value_ratio * next_q_value + (1 - self.mix_value_ratio) * self.target_return
            return next_q_value

    def state_action_value(self, state_batch, action_batch, target=False, return_qf=False):
        """ compute state action value """
        critic = self.critic_target if target else self.critic
        qf1, qf2, critic_aux = critic(state_batch, None)
        scale_func = torch.tanh
        if self.critic_tanh:
            qf1 = scale_func(qf1)
            qf2 = scale_func(qf2)

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

        critic_loss = sum([getattr(self, name) for name in get_loss_info_dict().keys() if name.endswith('loss') and name.startswith('critic')])
        critic_loss.backward()
        self.state_feat_val_encoder_optim.step()
        self.critic_optim.step()

        if hasattr(self, "critic_target"):
            half_soft_update(self.critic_target, self.critic, self.tau)
            if self.update_step % self.target_update_interval == 0:
                half_hard_update(self.critic_target, self.critic, self.tau)


    def compute_critic_loss(self, value_feat):
        """
        compute one step bellman error
        """
        self.next_q_value = self.target_value()
        self.qf1, self.qf2, self.critic_aux_pred = self.state_action_value(value_feat, self.action_batch, return_qf=True)
        next_q_value = self.next_q_value
        qf1 = torch.min(self.qf1, self.qf2)
        self.critic_loss = self.value_loss_func(self.qf1.view(-1), next_q_value) + self.value_loss_func(self.qf2.view(-1), next_q_value)

        # critic gaddpg auxiliary
        gaddpg_batch = self.gaddpg_batch.bool()
        gaddpg_pred = self.critic_aux_pred[gaddpg_batch, -1]
        target_gaddpg_batch = self.target_return[gaddpg_batch]

        target_gaddpg_batch = (target_gaddpg_batch > 0).float()
        self.critic_gaddpg_loss = self.gaddpg_scale * F.binary_cross_entropy_with_logits(gaddpg_pred, target_gaddpg_batch)

        self.gaddpg_pred_mean = gaddpg_pred.mean().item()
        self.gaddpg_mask_num = len(target_gaddpg_batch)
        self.gaddpg_mask_ratio = target_gaddpg_batch.sum() / len(target_gaddpg_batch)
        self.valid_latent_num = len(qf1)


    def update_parameters(self, batch_data, updates, k, test=False):
        """ update agent parameters """
        self.mix_value_ratio, self.mix_policy_ratio = self.get_mix_ratio(self.update_step - self.init_step)
        self.set_mode(test)
        self.prepare_data(batch_data)
        traj_latent = self.traj_dataset_traj_latent if self.full_traj_embedding else self.dataset_traj_latent

        # value
        value_feat, _ = self.extract_feature(self.image_state_batch,
                                             self.point_state_batch,
                                             action_batch=self.action_batch,
                                             goal_batch=self.goal_batch,
                                             traj_goal_batch=self.traj_goal_batch,
                                             traj_point_state=self.traj_point_state_batch,
                                             time_batch=self.time_batch,
                                             curr_joint=self.curr_joint,
                                             joint_traj_state=self.traj_joint_batch,
                                             traj_latent=traj_latent, #
                                             traj_integer_time=self.traj_integer_time_batch,
                                             traj_time_batch=self.traj_time_batch,
                                             traj_action=self.traj_action_batch,
                                             value=True)
        self.compute_critic_loss(value_feat)

        self.critic_optimize()

        self.update_step += 1

        # log
        self.log_stat()
        return {k: float(getattr(self, k)) for k in get_loss_info_dict().keys()}

