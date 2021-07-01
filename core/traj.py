# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from core.utils import *
from core.agent import Agent
from core.loss import *


class TrajAgent(Agent):
    """
    Extends Agent to have traj functionality
    """

    def autoencode(self, state_input_batch, train, sampler_extra_time=None):
        """
        vae for embedding latent
        """
        self.sampler_traj_feat, _, _, self.sampler_extra_feat_pred = self.traj_feature_sampler( state_input_batch,
                                                                                                train=train,
                                                                                                extra_time=sampler_extra_time)
        self.sampler_traj_feat, self.sampler_latent_mean, self.sampler_latent_logstd, self.latent_sample =  self.traj_feature_sampler.module.conditional_sampler_vae_head(
                                                                                        self.sampler_traj_feat,
                                                                                        self.train_traj_idx_batch,
                                                                                        self.dataset_traj_latent_head )

    def extract_traj_embedding_feature(self, image_batch, state_input_batch, action_batch=None, goal_batch=None,
                              time_batch=None, vis=False,  goal_point_state=None,
                              repeat=False, train=True, curr_joint=None, traj_integer_time=None,
                              traj_state=None, traj_inbatch_index=None,
                              joint_traj_state=None, traj_goal=None,  traj_time_batch=None, sample_num=None,
                              traj_point_state=None, val=False):
        """
        extract trajectory embedding latent
        """
        train = not self.test_mode
        traj_point_state = preprocess_points(self, traj_point_state, joint_traj_state[:, :7], traj_integer_time, traj=True)
        self.traj_feat, self.traj_extra_feat_pred = self.traj_feature_extractor(
                                                    traj_point_state=traj_point_state,
                                                    traj_state=traj_state,
                                                    train=train,
                                                    traj_inbatch_index=traj_inbatch_index,
                                                    traj_time=traj_time_batch )

        # including expert and learner
        self.traj_feat, self.internal_traj_recons = self.traj_feature_extractor.module.head(self.traj_feat, traj_inbatch_index, val=val)
        self.traj_head_feat_mean = self.traj_feat.mean()
        self.traj_feat_copy = self.traj_feat.clone()

        if  self.train_traj_sampler:
            self.dataset_traj_latent = self.traj_feat
            self.traj_feat = self.traj_feat[self.train_traj_idx_batch]
            self.traj_feat_target = self.traj_feat.clone().detach()

        else:
            if not hasattr(self, 'cont_traj_inbatch_index'):
                self.cont_traj_inbatch_index = self.train_traj_idx_batch
            self.traj_feat = self.traj_feat[self.cont_traj_inbatch_index]
            self.traj_feat_target = self.traj_feat.clone().detach()

    def extract_traj_sampling_feature(self, image_batch, state_input_batch, action_batch=None, goal_batch=None,
                              time_batch=None, vis=False, value=False, repeat=False,
                              train=True, separate=True, curr_joint=None, traj_integer_time=None,
                              traj_state=None, traj_inbatch_index=None, index_mask=None,
                              joint_traj_state=None, traj_time_batch=None, sample_num=None,
                              traj_pose=None, traj_point_state=None):
        """
        extract trajectory sampler latent
        """
        train = not self.test_mode
        state_input_batch  = preprocess_points(self, state_input_batch, curr_joint, time_batch, append_pc_time=True)
        self.dataset_traj_latent_head = None if not train else self.dataset_traj_latent[self.train_traj_idx_batch]
        self.autoencode(state_input_batch, train)

        if  self.use_sampler_latent or not train:
            self.traj_feat = self.sampler_traj_feat

    def train_traj_net(self ):
        """
        training for embedding and sampler
        """
        if  not self.train_traj_sampler or not self.use_offline_latent:
            self.extract_traj_embedding_feature(  self.image_state_batch,
                                    self.point_state_batch,
                                    traj_point_state=self.sparsify_sim_traj_point_state_batch,
                                    goal_batch=self.goal_batch,
                                    time_batch=self.time_batch,
                                    curr_joint=self.curr_joint,
                                    joint_traj_state=self.sparsify_sim_traj_joint_batch,
                                    traj_integer_time=self.sparsify_sim_traj_integer_time_batch,
                                    traj_inbatch_index=self.sparsify_sim_cont_traj_inbatch_index,
                                    traj_time_batch=self.sparsify_sim_traj_time_batch )

        if  self.train_traj_sampler and not self.ignore_traj_sampler:
            self.extract_traj_sampling_feature(self.image_state_batch,
                                    self.point_state_batch,
                                    traj_point_state=self.sparsify_sim_traj_point_state_batch,
                                    goal_batch=self.goal_batch,
                                    time_batch=self.time_batch,
                                    curr_joint=self.curr_joint,
                                    joint_traj_state=self.sparsify_sim_traj_joint_batch,
                                    traj_integer_time=self.sparsify_sim_traj_integer_time_batch,
                                    traj_inbatch_index=self.sparsify_sim_cont_traj_inbatch_index,
                                    traj_time_batch=self.sparsify_sim_traj_time_batch)



    @torch.no_grad()
    def select_traj(
        self,
        img_state,
        point_state,
        goal_state=None,
        vis=False,
        remain_timestep=0,
        repeat=False,
        curr_joint=None,
        gt_traj=None,
        sample_num=None,
    ):
        """
        process trajectory in test time
        """
        self.set_mode(True)
        query_traj_time = torch.linspace(0, 1, int(remain_timestep) + 1).cuda()[1:].float()
        self.train_traj_idx_batch = torch.Tensor([0]).long().cuda()
        if len(point_state) > 1:
            self.timestep = torch.Tensor([remain_timestep] * len(point_state)).float().cuda() # remain_timestep
            remain_timestep = self.timestep
            self.train_traj_idx_batch = torch.Tensor([0]).long().cuda().repeat(len(point_state))

        if gt_traj is not None:
            if not hasattr(self, 'traj_feature_extractor'): return
            traj_point_state, joint_traj_state, traj_action, traj_poses, traj_goals, traj_integer_time, query_traj_time = gt_traj
            self.extract_traj_embedding_feature(  img_state,
                                    point_state,
                                    time_batch=remain_timestep,
                                    goal_batch=goal_state,
                                    curr_joint=curr_joint,
                                    repeat=repeat,
                                    train=False,
                                    traj_time_batch=query_traj_time[:, 1][:,None],
                                    traj_point_state=traj_point_state,
                                    joint_traj_state=joint_traj_state,
                                    traj_integer_time=traj_integer_time,
                                    traj_inbatch_index=torch.zeros_like(traj_integer_time).long() )

            traj = None

        else:
            if not hasattr(self, 'traj_feature_sampler'): return
            self.extract_traj_sampling_feature(img_state,
                                        point_state,
                                        time_batch=remain_timestep,
                                        goal_batch=goal_state,
                                        curr_joint=curr_joint,
                                        repeat=repeat,
                                        train=False,
                                        traj_time_batch=query_traj_time[:,None],
                                        traj_inbatch_index=torch.zeros(len(point_state),
                                        device=torch.device('cuda')).long())
            print('sample traj latent')
            traj = None

        self.traj_feat_target_test = self.traj_feat.clone()
        if  has_check(self, 'multi_traj_sample') and gt_traj is None:
            traj = self.multi_select_traj(img_state, point_state, remain_timestep, curr_joint, vis)
        return traj

    @torch.no_grad()
    def multi_select_traj(
        self,
        img_state,
        point_state,
        remain_timestep=0,
        curr_joint=None,
        vis=False,
    ):
        """
        multi-sample trajectory selection in test time
        """
        vis_traj = None
        if type(curr_joint) is not np.ndarray:
            curr_joint = curr_joint.detach().cpu().numpy()
            finger_joints = np.ones((len(curr_joint), 2)) * 0.04
            curr_joint = np.concatenate((curr_joint, finger_joints), axis=1)

        if type(point_state) is not np.ndarray:
            point_state = point_state.detach().cpu().numpy()

        if vis:
            vis_traj, traj_time_batch,  joint_traj_state, traj_point_state, traj_integer_time = generate_simulated_learner_trajectory(
                                        point_state, curr_joint, self, remain_timestep, self.test_traj_num)
        return vis_traj
