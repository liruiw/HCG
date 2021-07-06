# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F

import numpy as np
from core import networks
from core.utils import *
from core.loss import *
import IPython
import time


class Agent(object):
    """
    A general agent class
    """

    def __init__(self, num_inputs, action_space, args, name):
        for key, val in args.items():
            setattr(self, key, val)

        self.name = name
        self.device = "cuda"
        self.update_step = 1
        self.init_step = 1
        self.action_dim = action_space.shape[0]
        self.has_critic = self.name != "BC"
        self.action_space = action_space
        self.num_inputs = num_inputs + self.num_input_extra
        self.traj_feat = None
        self.latent_sample = None
        self.test_mode = False
        self.use_debug_latent = False
        self.gaddpg_pred = 0.
        if has_check(self, 'traj_goal_mutual_conditioned') :
            self.num_inputs += self.policy_traj_latent_size
        self.policy, self.policy_optim, self.policy_scheduler, self.policy_target = get_policy_class('GaussianPolicy', self)


    def unpack_batch(
        self,
        state,
        point_state=None,
        vis=False,
        gt_goal=None,
        val=False,
        grasp_set=None,
        vis_image=False,
        repeat=False,
        traj_latent=None,
        separate=True
    ):
        """
        Extract features from point cloud input
        """
        if type(point_state) is list or type(point_state) is np.ndarray:
            point_state = torch.cuda.FloatTensor(point_state )
        if type(state) is list or type(state) is np.ndarray:
            state = torch.cuda.FloatTensor(state)

        state_feature, network_input = self.state_feature_extractor(
                                        point_state,
                                        feature_2=val,
                                        traj_latent=traj_latent,
                                        train=not self.test_mode)
        if  len(state_feature) != 2 or type(state_feature) is torch.Tensor: state_feature = [state_feature, None]
        return state_feature


    def gaddpg_step(self, state,  remain_timestep, curr_joint ):
        """ use GADDPG to forward pass """
        state = select_target_point(state)
        gaddpg_remain_step = max(min(remain_timestep + 1, 25), 1)
        return self.gaddpg.select_action(state, remain_timestep=gaddpg_remain_step, curr_joint=curr_joint)


    @torch.no_grad()
    def batch_select_action(
        self,
        state,
        actions=None,
        goal_state=None,
        vis=False,
        remain_timestep=0,
        repeat=False,
        curr_joint=None,
        gt_traj=None,
        sample_num=None
    ):
        """
        run policy forward pass in batch simulation
        """
        self.set_mode(True)
        traj = None
        curr_joint_th = torch.cuda.FloatTensor(curr_joint)[:, :7]
        img_state = torch.cuda.FloatTensor(state[0][1])
        point_state = torch.cuda.FloatTensor(state[0][0])
        timestep = remain_timestep
        self.timestep = timestep
        agent = self

        feature, extra = agent.extract_feature( img_state,
                                                point_state,
                                                time_batch=timestep,
                                                goal_batch=goal_state,
                                                vis=vis,
                                                value=False,
                                                train=False,
                                                repeat=repeat,
                                                curr_joint=curr_joint_th )

        actions = agent.policy.sample(feature)
        action = actions[0].detach().cpu().numpy()
        extra_pred = actions[1].detach().cpu().numpy()
        action_sample = actions[2].detach().cpu().numpy()
        aux_pred = actions[3].detach().cpu().numpy()
        return action, traj, extra_pred, aux_pred


    @torch.no_grad()
    def select_action(
        self,
        state,
        actions=None,
        goal_state=None,
        vis=False,
        remain_timestep=0,
        repeat=False,
        curr_joint=None,
        gt_traj=None,
        sample_num=None
    ):
        """
        policy output in test time
        """
        self.set_mode(True)
        multi_sample = has_check(self, 'multi_traj_sample') and gt_traj is None

        if multi_sample and hasattr(self, 'critic') and self.train_traj_sampler and self.critic_mpc:
            return self.critic_select_action(state, remain_timestep, curr_joint, vis=vis)

        if self.name == 'DQN_HRL' and gt_traj is None and vis:
            return self.critic_select_action(state, remain_timestep, curr_joint, vis=vis)

        curr_joint_th = torch.Tensor([curr_joint.flatten()]).float().cuda()[:, :7]
        img_state = torch.cuda.FloatTensor(state[0][1])[None]
        point_state = torch.cuda.FloatTensor(state[0][0])[None]
        timestep = torch.cuda.FloatTensor([remain_timestep])
        self.timestep = timestep

        if  has_check(self, 'train_traj_sampler') and gt_traj is None and has_check(self, 'train_traj_feature'):
            if  multi_sample: # multiple traj samples
                traj = self.select_traj(img_state,
                                    point_state.repeat((self.test_traj_num, 1, 1)),
                                    goal_state,
                                    vis=vis,
                                    remain_timestep=remain_timestep,
                                    curr_joint=curr_joint_th.repeat((self.test_traj_num, 1)))
                timestep = torch.Tensor([remain_timestep]).float().cuda()
                opt_idx = 0
                self.traj_feat = self.traj_feat[[opt_idx]]
            else:
                traj = self.select_traj(img_state, point_state, goal_state,
                                        vis=vis, remain_timestep=remain_timestep,
                                        curr_joint=curr_joint_th )

        else:
            traj = None

        # policy
        feature, extra = self.extract_feature(  img_state,
                                                point_state,
                                                time_batch=timestep,
                                                goal_batch=goal_state,
                                                value=False,
                                                train=False,
                                                repeat=repeat,
                                                curr_joint=curr_joint_th[:,:7] )


        if self.name == 'DQN_HRL' and vis and hasattr(self, 'sampler_traj_feat'):
            self.compute_critic_value( img_state, point_state, timestep, curr_joint_th, goal_state)

        actions = self.policy.sample(feature)
        action = actions[0].detach().cpu().numpy()[0]
        extra_pred = actions[1].detach().cpu().numpy()[0]
        action_sample = actions[2].detach().cpu().numpy()[0]
        aux_pred = actions[3].detach().cpu().numpy()[0]
        return action, traj, extra_pred, aux_pred


    def update_parameters(self, batch_data, updates, k):
        """
        To be inherited
        """
        return {}


    def compute_loss(self):
        """
        compute loss for policy and trajectory embedding
        """
        self.policy_grasp_aux_loss =  goal_pred_loss(self.aux_pred[self.target_goal_reward_mask, :7], self.target_grasp_batch[self.target_goal_reward_mask, :7] )
        self.bc_loss =  traj_action_loss(self, self.pi, self.traj_expert_action_batch, self.target_expert_mask)
        return sum([getattr(self, name) for name in self.loss_info if name.endswith('loss') and not name.startswith('critic')])

    def prepare_data(self, batch_data):
        """
        load batch data dictionary and compute extra data
        """
        update_step = self.update_step - self.init_step
        self.loss_info  = list(get_loss_info_dict().keys())

        for name in self.loss_info:
            setattr(self, name, torch.zeros(1, device=torch.device('cuda')))

        for k, v in batch_data.items():
            setattr(self, k, torch.cuda.FloatTensor(v))

        self.traj_time_batch = self.traj_idx_batch[:, 1, None]
        self.cont_traj_inbatch_index = self.traj_idx_batch[:, 0].cuda().long()

        self.traj_feat = None
        self.reward_mask = (self.return_batch > 0).view(-1)
        self.expert_mask = (self.expert_flag_batch >= 1).view(-1)

        self.expert_reward_mask = self.reward_mask * (self.expert_flag_batch >= 1).squeeze()
        self.perturb_flag_batch = self.perturb_flag_batch.bool()

        self.traj_expert_reward_mask = self.expert_reward_mask[self.cont_traj_inbatch_index]
        self.train_traj_idx_batch = self.cont_traj_inbatch_index
        self.sparsify_sim_traj_time_batch = self.sparsify_sim_traj_idx_batch[:, 1, None]
        self.sparsify_sim_cont_traj_inbatch_index = self.sparsify_sim_traj_idx_batch[:, 0].cuda().long()
        self.sparsify_sim_traj_expert_reward_mask = self.expert_reward_mask[self.sparsify_sim_cont_traj_inbatch_index]

        self.goal_reward_mask = torch.ones_like(self.time_batch).bool()
        self.traj_goal_reward_mask = torch.ones_like(self.traj_integer_time_batch).bool()
        self.target_grasp_batch = self.traj_goal_batch[:, :7] if self.full_traj_embedding  else self.goal_batch[:, :7]
        self.target_goal_reward_mask = self.goal_reward_mask[self.cont_traj_inbatch_index] if self.full_traj_embedding   else self.goal_reward_mask
        self.target_reward_mask = self.reward_mask[self.cont_traj_inbatch_index] if self.full_traj_embedding else self.reward_mask
        self.target_return = self.return_batch[self.cont_traj_inbatch_index] if self.full_traj_embedding else self.return_batch
        self.target_expert_mask = self.expert_mask[self.cont_traj_inbatch_index] if self.full_traj_embedding else self.expert_mask
        self.target_gaddpg_batch = (self.gaddpg_batch * self.reward_mask)
        self.target_expert_reward_mask = self.traj_expert_reward_mask if self.full_traj_embedding else self.expert_reward_mask
        self.next_time_batch = self.time_batch - 1
        self.next_traj_time_batch = self.traj_integer_time_batch - 1
        self.target_reward_batch = self.traj_reward_batch if self.full_traj_embedding else self.reward_batch
        self.target_mask_batch = self.traj_mask_batch if self.full_traj_embedding else self.mask_batch

    def log_stat(self):
        """
        log grad and param statistics for tensorboard
        """
        self.policy_grad = module_max_gradient(self.policy)
        self.feat_grad = module_max_gradient(self.state_feature_extractor.module.encoder)
        self.feat_param = module_max_param(self.state_feature_extractor.module.encoder)
        self.val_feat_grad = module_max_gradient(self.state_feature_extractor.module.value_encoder)
        self.val_feat_param = module_max_param(self.state_feature_extractor.module.value_encoder)
        self.policy_param = module_max_param(self.policy)
        self.reward_mask_num = self.reward_mask.float().sum()
        self.max_traj_sample_len = torch.unique(self.cont_traj_inbatch_index, return_counts=True)[1].max()
        self.traj_num = len(self.reward_mask)
        self.train_batch_size = len(self.target_expert_reward_mask)

        if hasattr(self, 'traj_feature_extractor'):
            self.traj_grad = module_max_gradient(self.traj_feature_extractor)
            self.traj_param = module_max_param(self.traj_feature_extractor)

        if hasattr(self, 'sampler_gaussian'):
            self.sampler_mean = self.sampler_gaussian[0].mean().item()
            self.sampler_logsigma = self.sampler_gaussian[1].mean().item()

        if  self.train_traj_sampler and hasattr(self, 'sampler_traj_feat'):
            self.traj_sampler_grad = module_max_gradient(self.traj_feature_sampler)
            self.traj_sampler_param = module_max_param(self.traj_feature_sampler)

        if self.has_critic:
            self.value_mean, self.value_mean_2  = self.qf1.mean(), self.qf2.mean()
            self.target_mean = self.next_q_value.mean()
            self.return_mean = self.traj_return_batch.mean()
            self.value_min, self.value_max  = self.qf1.min(), self.qf1.max()

            self.expert_reward_mask_num = self.expert_reward_mask.sum()
            self.goal_reward_mask_num = self.goal_reward_mask.sum()
            self.reward_mask_num = self.reward_mask.sum()
            self.return_min, self.return_max = self.return_batch.min(), self.return_batch.max()

            self.critic_grad = module_max_gradient(self.critic)
            self.critic_param = module_max_param(self.critic)

    def set_mode(self, test):
        """
        set training or test mode for network
        """
        self.test_mode = test

        if not test:
            self.state_feature_extractor.train()
            self.policy.train()

            if hasattr(self, "critic"):
                self.critic.train()
                self.critic_optim.zero_grad()
                self.state_feat_val_encoder_optim.zero_grad()

            if hasattr(self, 'traj_feature_extractor'):
                if  self.train_traj_feature and not self.fix_traj_feature:
                    self.traj_feature_extractor.train()
                else:
                    self.traj_feature_extractor.eval()
                if  self.train_traj_sampler:
                    self.traj_feature_sampler.train()
        else:
            torch.no_grad()
            self.policy.eval()
            self.state_feature_extractor.eval()
            if hasattr(self, "critic"): self.critic.eval()
            if hasattr(self, "traj_feature_extractor"): self.traj_feature_extractor.eval()
            if hasattr(self, "traj_feature_sampler"): self.traj_feature_sampler.eval()


    def setup_feature_extractor(self, net_dict, test_time=False):
        """
        Load networks
        """
        if "traj_feature_extractor" in net_dict:
            self.traj_feature_extractor = net_dict["traj_feature_extractor"]["net"]
            self.traj_feature_extractor_opt = net_dict["traj_feature_extractor"]["opt"]
            self.traj_feature_extractor_sch = net_dict["traj_feature_extractor"]["scheduler"]
        else:
            self.traj_feature_extractor = net_dict["state_feature_extractor"]["net"]
        if 'traj_feature_sampler' in net_dict:
            self.traj_feature_sampler = net_dict["traj_feature_sampler"]["net"]
            self.traj_feature_sampler_opt = net_dict["traj_feature_sampler"]["opt"]
            self.traj_feature_sampler_sch = net_dict["traj_feature_sampler"]["scheduler"]

        self.state_feature_extractor = net_dict["state_feature_extractor"]["net"]
        self.state_feature_extractor_optim = net_dict["state_feature_extractor"]["opt"]
        self.state_feature_extractor_scheduler = net_dict["state_feature_extractor"]["scheduler"]
        self.state_feat_encoder_optim = net_dict["state_feature_extractor"][ "encoder_opt" ]
        self.state_feat_encoder_scheduler = net_dict["state_feature_extractor"][ "encoder_scheduler" ]
        self.state_feat_val_encoder_optim = net_dict["state_feature_extractor"][ "val_encoder_opt" ]
        self.state_feat_val_encoder_scheduler = net_dict["state_feature_extractor"][ "val_encoder_scheduler" ]
        self.test_time = test_time

    def  get_mix_ratio(self, update_step):
        """
        Get a mixed schedule for supervised learning and RL
        """
        idx = int((self.update_step > np.array(self.mix_milestones)).sum())
        mix_policy_ratio = get_valid_index(self.mix_policy_ratio_list, idx)
        mix_policy_ratio = min(mix_policy_ratio, self.ddpg_coefficients[4])
        mix_value_ratio  = get_valid_index(self.mix_value_ratio_list, idx)
        mix_value_ratio  = min(mix_value_ratio, self.ddpg_coefficients[3])
        return mix_value_ratio, mix_policy_ratio

    def get_lr(self):
        """
        Get network learning rates
        """
        lrs = {
            "policy_lr": self.policy_optim.param_groups[0]["lr"],
            "feature_lr": self.state_feature_extractor_optim.param_groups[0]["lr"],
        }

        if self.train_traj_feature:
            lrs["traj_feature_lr"] = self.traj_feature_extractor_opt.param_groups[0]["lr"]
        if self.train_traj_sampler:
            lrs["traj_sampler_lr"] = self.traj_feature_sampler_opt.param_groups[0]["lr"]
        if hasattr(self, 'critic_optim'):
            lrs["value_lr"] = self.critic_optim.param_groups[0]["lr"]
            lrs["val_feat_lr"] = self.state_feat_val_encoder_optim.param_groups[0]["lr"]

        headers = ["network", "learning rate"]
        data = [(name, lr) for name, lr in lrs.items()]
        return lrs

    def optimize(self, loss, update_step):
        """
        Backward loss and update optimizer
        """
        self.state_feat_encoder_optim.zero_grad()
        self.policy_optim.zero_grad()
        if self.train_traj_feature:
            self.traj_feature_extractor_opt.zero_grad()
        if self.train_traj_sampler:
            self.traj_feature_sampler_opt.zero_grad()

        loss.backward(retain_graph=self.re_sampler_step)
        self.policy_optim.step()
        if self.train_feature:
            self.state_feat_encoder_optim.step()

        if self.train_traj_feature:
            self.traj_feature_extractor_opt.step()

        if self.train_traj_sampler:
            self.traj_feature_sampler_opt.step()

    def step_scheduler(self, step=None):
        """
        Update network scheduler
        """
        if self.train_traj_sampler:
            self.traj_feature_sampler_sch.step()
        if self.train_traj_feature:
            self.traj_feature_extractor_sch.step()

        if hasattr(self, "critic"):
            self.critic_scheduler.step()
        if hasattr(self, "policy"):
            self.policy_scheduler.step()
        if self.train_feature or self.train_value_feature:
            self.state_feature_extractor_scheduler.step()
            self.state_feat_encoder_scheduler.step()
        if self.train_value_feature and hasattr(self, 'state_feat_val_encoder_scheduler'):
            self.state_feat_val_encoder_scheduler.step()

    def save_model(
        self,
        step,
        output_dir="",
        surfix="latest",
        actor_path=None,
        critic_path=None,
        traj_feat_path=None,
        state_feat_path=None,
    ):
        """
        save model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        actor_path, critic_path, traj_feat_path, traj_sampler_path, state_feat_path = get_model_path(output_dir,
                                                                   self.name, self.env_name, surfix)
        print("Saving models to {} and {}".format(actor_path, critic_path))
        if hasattr(self, "policy"):
            torch.save(
                {
                    "net": self.policy.state_dict(),
                    "opt": self.policy_optim.state_dict(),
                    "sch": self.policy_scheduler.state_dict(),
                },
                actor_path,
            )
        if hasattr(self, "critic"):
            torch.save(
                {
                    "net": self.critic.state_dict(),
                    "opt": self.critic_optim.state_dict(),
                    "sch": self.critic_scheduler.state_dict(),
                },
                critic_path,
            )

        if  hasattr(self, 'traj_feature_extractor_opt'):
            torch.save(
                {
                    "net": self.traj_feature_extractor.state_dict(),
                    "opt": self.traj_feature_extractor_opt.state_dict(),
                    "sch": self.traj_feature_extractor_sch.state_dict(),
                },
                traj_feat_path,
            )
        if  hasattr(self, 'traj_feature_sampler_opt'):
            torch.save(
                {
                    "net": self.traj_feature_sampler.state_dict(),
                    "opt": self.traj_feature_sampler_opt.state_dict(),
                    "sch": self.traj_feature_sampler_sch.state_dict(),
                },
                traj_sampler_path,
            )

        torch.save(
            {
                "net": self.state_feature_extractor.state_dict(),
                "opt": self.state_feature_extractor_optim.state_dict(),
                "encoder_opt": self.state_feat_encoder_optim.state_dict(),
                "sch": self.state_feature_extractor_scheduler.state_dict(),
                "encoder_sch": self.state_feat_encoder_scheduler.state_dict(),
                "val_encoder_opt": self.state_feat_val_encoder_optim.state_dict(),
                "val_encoder_sch": self.state_feat_val_encoder_scheduler.state_dict(),
                "step": step,
            },
            state_feat_path,
        )

    def load_model(
        self, output_dir, surfix="latest", set_init_step=False, reinit_value_feat=False
    ):
        """
        Load saved model
        """
        actor_path, critic_path, traj_feat_path, traj_sampler_path, state_feat_path = get_model_path(output_dir,
                                                                   self.name, self.env_name, surfix)
        if hasattr(self, "policy") and os.path.exists(actor_path):
            net_dict = torch.load(actor_path)
            self.policy.load_state_dict(net_dict["net"])
            self.policy_optim.load_state_dict(net_dict["opt"])
            self.policy_scheduler.load_state_dict(net_dict["sch"])

            if self.reinit_optim and set_init_step:
                for g in self.policy_optim.param_groups:
                    g["lr"] = self.reinit_lr
                self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.policy_optim, milestones=self.policy_milestones, gamma=0.5 )
                self.policy_scheduler.initial_lr = self.reinit_lr
                self.policy_scheduler.base_lrs[0] = self.reinit_lr
                print("reinit policy optim")

            print("load policy weight: {:.3f} from {} !!!!".format(module_max_param(self.policy), actor_path))
            hard_update(self.policy_target, self.policy, self.tau)

        if hasattr(self, "critic") and os.path.exists(critic_path):
            net_dict = torch.load(critic_path)
            self.critic.load_state_dict(net_dict["net"])
            self.critic_optim.load_state_dict(net_dict["opt"])
            self.critic_scheduler.load_state_dict(net_dict["sch"])
            print("load critic weight: {:.3f} !!!!".format(module_max_param(self.critic)))
            hard_update(self.critic_target, self.critic, self.tau)

        if  hasattr(self, 'traj_feature_extractor')  and os.path.exists(traj_feat_path):
            net_dict = torch.load(traj_feat_path)
            self.traj_feature_extractor.load_state_dict(net_dict["net"], strict=False)
            print('load traj feature weight: {:.3f} from {} !!!!'.format(module_max_param(self.traj_feature_extractor), traj_feat_path))
            try:
                self.traj_feature_extractor_opt.load_state_dict(net_dict["opt"])
                self.traj_feature_extractor_sch.load_state_dict(net_dict["sch"])
            except:
                pass

        if  hasattr(self, 'train_traj_sampler') and os.path.exists(traj_sampler_path):
            net_dict = torch.load(traj_sampler_path)
            self.traj_feature_sampler.load_state_dict(net_dict["net"], strict=False)
            print('load traj sampler weight: {:.3f} from {} !!!!'.format(module_max_param(self.traj_feature_sampler), traj_sampler_path))
            try:
                self.traj_feature_sampler_opt.load_state_dict(net_dict["opt"])
                self.traj_feature_sampler_sch.load_state_dict(net_dict["sch"])
            except:
                pass

        if os.path.exists(state_feat_path):
            net_dict = torch.load(state_feat_path)
            if  has_check(self, 'reinit_feat_opt'):
                self.state_feature_extractor.load_state_dict(dict([(n, p) for n, p in net_dict["net"].items() if 'value' not in n ]),strict=False)
            else:
                self.state_feature_extractor.load_state_dict(net_dict["net"] )
            self.state_feature_extractor_optim.load_state_dict(net_dict["opt"])
            self.state_feature_extractor_scheduler.load_state_dict( net_dict["sch"] )
            self.state_feat_encoder_optim.load_state_dict( net_dict["encoder_opt"] )
            self.state_feat_encoder_scheduler.load_state_dict( net_dict["encoder_sch"] )
            if not has_check(self, 'reinit_feat_opt'):
                self.state_feat_val_encoder_optim.load_state_dict(
                net_dict["val_encoder_opt"] )
                self.state_feat_val_encoder_scheduler.load_state_dict(
                net_dict["val_encoder_sch"] )
            print(
                "load feature weight: {} !!!! from: {} step :{}".format(
                module_max_param(self.state_feature_extractor), state_feat_path, net_dict["step"]))

            self.update_step = net_dict["step"]
            self.init_step = self.update_step
            return self.update_step
        return 0
