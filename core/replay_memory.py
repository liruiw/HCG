# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
from torch.utils.data import Dataset
import IPython
import time
import cv2
import random

from core.utils import *
from collections import deque


class BaseMemory(Dataset):
    """Defines a generic experience replay memory module."""

    def __init__(self, buffer_size, args, name="expert"):

        self.cur_idx = 0
        self.cur_episode_idx = 0
        self.total_env_step = 0
        self.is_full = False
        self.name = name
        self.one_sample_timer = AverageMeter()

        for key, val in args.RL_TRAIN.items():
            setattr(self, key, val)

        self.buffer_size = buffer_size
        self.episode_buffer_size = self.buffer_size // self.max_step
        self.episode_max_len = args.RL_MAX_STEP
        self.save_data_name = args.RL_SAVE_DATA_NAME
        self.offset_candidates = combinationSum2(np.arange(1, self.max_step - 1),
                                            self.batch_size, self.batch_sequence_size,
                                            self.batch_sequence_ratio, self.max_step, min_step=self.clip_tail_idx)
        self.batch_sequence_size = int(self.batch_sequence_size * self.batch_sequence_ratio)

        self.attr_names = [
            "action",
            "pose",
            "point_state",
            "target_idx",
            "reward",
            "terminal",
            "timestep",
            "returns",
            "state_pose",
            "collide",
            "grasp",
            "perturb_flags",
            "goal",
            "expert_flags",
            "expert_action",
            'joint_action_delta',
            'curr_joint',
            'expert_joint_action_delta',
            'link_collision_distance',
            'link_collision_dist_vec',
            'expert_joint_plan',
            'plan_goal_change_flags',
            'gaddpg_flags',
            'scene_idx',
            'dataset_state_latent',
            'dataset_traj_latent',
            'dataset_normal_latent'
        ]

        for attr in self.attr_names:
            setattr(self, attr, None)
        (
            self._REW,
            self._LIFTED,
            self._COLLISION,
            self._ONLINE_REW,
            self._ONLINE_LIFTED,
            self._ONLINE_COLLISION,
            self._TEST_REW,
            self._TOTAL_REW,
            self._TOTAL_CNT,
            self._TOTAL_LIFTED,
            self._TOTAL_COLLISION,
        ) = (
            deque([0] * 30, maxlen=200),
            deque([0] * 30, maxlen=200),
            deque([1] * 30, maxlen=200),
            deque([0], maxlen=200),
            deque([0], maxlen=200),
            deque([0], maxlen=200),
            deque([0], maxlen=200),
            0.,
            1.,
            0.,
            1.,
        )
        self.dir = args.RL_DATA_ROOT_DIR
        self.load_obj_performance()
        self.init_buffer()


    def get_total_lifted(self):
        return self._TOTAL_LIFTED

    def get_info(self):
        return (self.upper_idx(), self.get_cur_idx(),  self.is_full,
                self.print_obj_performance(), self.get_total_env_step(),
                self.get_expert_upper_idx(), self.one_sample_timer.avg)

    def update_reward(self, reward, test, explore, target_name):
        # episode reward statistics log
        self._TOTAL_REW += reward
        lifted = reward > 0.5
        collided = reward < 0.
        self._TOTAL_LIFTED += lifted
        self._TOTAL_COLLISION += collided
        self._TOTAL_CNT += 1
        self._REW.append(reward)
        self._LIFTED.append(lifted)
        self._COLLISION.append(collided)

        if target_name != "noexists" and target_name not in self.object_performance:
            self.object_performance[target_name] = [0, 0, 0]
        self.object_performance[target_name][0] += 1
        self.object_performance[target_name][1] += reward

    def load_obj_performance(self):
        self.object_performance = {}

    def reward_info(self):
        return (
            self._TOTAL_REW / self._TOTAL_CNT,
            np.mean(list(self._REW)),
            np.mean(list(self._ONLINE_REW)),
            np.mean(list(self._TEST_REW)),
            np.mean(list(self._COLLISION)),
            np.mean(list(self._LIFTED)),
        )

    def print_obj_performance(self):
        pass

    def __len__(self):
        return self.upper_idx()

    def __getitem__(self, idx):
        return self.sample( )

    def fetchitem(self, idx):
        data = {
                "image_state_batch":  process_image_output(self.image_state[idx]),
                "expert_action_batch": np.float32(self.expert_action[idx]),
                "action_batch": np.float32(self.action[idx]),
                "reward_batch": np.float32(self.reward[idx]),
                "return_batch": np.float32(self.returns[idx]),
                "next_image_state_batch": None,
                "mask_batch": np.float32(self.terminal[idx]),
                "time_batch": np.float32(self.timestep[idx]),
                "point_state_batch": None,
                "next_point_state_batch": None,
                "state_pose_batch": np.float32(self.state_pose[idx]),
                "collide_batch": np.float32(self.collide[idx]),
                "grasp_batch": np.float32(self.grasp[idx]),
                "goal_batch": np.float32(self.goal[idx]),
                'curr_joint': np.float32(self.curr_joint[idx]),
            }
        return data

    def upper_idx(self):
        return max(self.cur_idx, 1) if not self.is_full else len(self.point_state)

    def is_full(self):
        return self.is_full

    def get_cur_idx(self):
        return self.cur_idx

    def get_expert_upper_idx(self):
        upper_idx = self.upper_idx()
        if self.expert_flags is not None and np.sum(self.expert_flags[:upper_idx]) > 0:
            return np.where(self.expert_flags[:upper_idx] >= 1)[0][-1]
        else:
            return 0

    def get_total_env_step(self):
        return self.total_env_step

    def reset(self):
        self.cur_idx = 0
        self.is_full = False

    def recompute_return_with_gamma(self):
        end_indexes  = np.sort(np.unique(self.episode_map))
        copy_returns = self.returns.copy()

        for idx in range(len(end_indexes) - 1):
            start = end_indexes[idx]
            end = end_indexes[idx+1]
            if end > self.cur_idx: continue
            cost_to_go = 0
            for i in range(end - start):
                cur_idx = end + 1
                copy_returns[cur_idx-i-1] = self.reward[cur_idx-i-1] + self.gamma ** i * cost_to_go
                cost_to_go = copy_returns[cur_idx-i-1]
        self.returns = copy_returns

    def sample_offset(self, total, size):
        offset = np.array(random.sample(self.offset_candidates, 1)[0])
        np.random.shuffle(offset)
        return offset

    def sample(self, batch_size=None, batch_idx=None ):
        """Samples a batch of experience from the buffer."""
        batch_size = self.batch_size
        batch_sequence_size = self.batch_sequence_size
        batch_size = int(batch_size * self.online_buffer_ratio) if self.name != 'expert' else batch_size

        s = time.time()
        upper_idx = self.upper_idx()

        if self.cur_idx <= batch_size * 2 or upper_idx <= self.max_step * 2:
            return {}

        if batch_idx is None:
            if  self.full_traj_embedding :
                nonzero_episode_map_idx = np.nonzero(self.unique_episode_map)[0][2:-2]
                if len(nonzero_episode_map_idx) < batch_sequence_size:  return {}
                idx = np.random.choice(nonzero_episode_map_idx, batch_sequence_size, replace=False)
                batch_idx = self.unique_episode_map[idx]
                sample_offset = self.sample_offset(batch_size, batch_sequence_size)
                sample_offset = np.minimum(sample_offset, self.episode_length[idx] - 2)
                sample_offset = np.minimum(sample_offset, batch_idx)
                batch_idx = batch_idx - sample_offset
            else:
                batch_idx = np.random.randint(self.max_step, upper_idx, batch_size)
                if self.fix_batch_gaddpg_latent: # fix gaddpg size
                    gaddpg_batch_idx = np.where(self.gaddpg_flags[:upper_idx].reshape(-1) != 0)[0]
                    non_gaddpg_batch_idx = np.where(self.gaddpg_flags[:upper_idx].reshape(-1) == 0)[0]
                    if len(gaddpg_batch_idx) > len(batch_idx) and len(non_gaddpg_batch_idx) > len(batch_idx):
                        split_gaddpg_batch_size = int(self.gaddpg_batch_ratio * batch_size)
                        split_latent_batch_size = batch_size - split_gaddpg_batch_size
                        gaddpg_batch = np.random.choice(gaddpg_batch_idx, split_gaddpg_batch_size, replace=False)
                        non_gaddpg_batch = np.random.choice(non_gaddpg_batch_idx, split_latent_batch_size, replace=False)
                        batch_idx = np.concatenate((gaddpg_batch, non_gaddpg_batch))
                np.random.shuffle(batch_idx)

        batch_idx = batch_idx.astype(np.int)
        data = self.fetchitem(batch_idx)
        self.post_process_batch(data, batch_idx)
        self.one_sample_timer.update(time.time() - s)
        return data

    def push(self, step_dict):
        """
        Push a single data item to the replay buffer
        """
        if self.action is None:
            self.init_buffer()
        store_idx = self.cur_idx % len(self.point_state)
        if (step_dict["point_state"].shape[1] < 100 or step_dict["point_state"].sum() == 0):
            print('empty state')
            return

        attr_names = self.attr_names[:]
        for name in attr_names:
            if name in step_dict:
                try:
                    getattr(self, name)[store_idx] = step_dict[name]
                except:
                    print('push data error:', name, store_idx, (getattr(self, name).shape, step_dict[name].shape))
        if self.cur_idx >= len(self.episode_map) - 1:
            self.is_full = True
        self.cur_idx = self.cur_idx + 1
        self.total_env_step += 1

        if self.cur_idx >= len(self.point_state) or self.cur_idx < self.buffer_start_idx:
            self.cur_idx = self.buffer_start_idx

    def add_episode(self, episode, explore=False, test=False):
        """
        add an rollout to the dataset
        """
        episode_length = len(episode)
        if episode_length <= 8: return
        if (self.name == 'expert' and not explore) and (
            episode_length <= self.max_step - 3
            or episode[-1]["reward"] < 0.5 \
            or episode[-1]["point_state"].shape[-1] < 2000 \
            or episode[-1]["collide"] \
            or plan_length(episode[0]["expert_joint_plan"]) < self.max_step) :
            return

        if episode_length > 0:
            self.update_reward( episode[-1]["reward"], test, explore, episode[-1]["target_name"])

        for transition in episode:
            self.push(transition)

        if self.cur_idx - episode_length >= 0 and episode_length > 0:
            cost_to_go = 0
            for i in range(episode_length):
                self.returns[self.cur_idx - i - 1] = (
                    self.reward[self.cur_idx - i - 1] + self.gamma ** i * cost_to_go
                )
                cost_to_go = self.returns[self.cur_idx - 1 - i]
            self.episode_map[self.cur_idx - episode_length : self.cur_idx] = self.cur_idx - 1
            self.unique_episode_map[self.cur_episode_idx] = self.cur_idx - 1
            self.episode_length[self.cur_episode_idx] = episode_length
            self.cur_episode_idx = (self.cur_episode_idx + 1) % len(self.unique_episode_map)
            print('push {} buffer episode idx: {} step idx: {}'.format(self.name, self.cur_episode_idx, self.cur_idx))

    def compute_traj(self, data, batch_idx):
        """
        compute a traj from data episode states
        """
        episode_end = self.episode_map[batch_idx].copy()
        curr_traj_time  =   data['curr_traj_time_batch']
        traj_data_names =   ['traj_idx_batch',  'traj_action_batch', 'traj_pose_batch',
                             'traj_expert_action_batch', 'traj_point_state_batch', 'traj_reward_batch',
                             'traj_mask_batch', 'next_traj_action_batch', 'next_traj_point_state_batch',
                             'traj_integer_time_batch',  'traj_joint_batch', 'traj_return_batch',
                             'traj_goal_batch', 'traj_next_goal_batch',  'next_traj_joint_batch', 'traj_gaddpg_batch',
                             'traj_dataset_traj_latent', 'traj_dataset_state_latent', 'traj_dataset_next_state_latent',
                             'sim_traj_point_state_batch', 'sim_traj_joint_batch', 'sim_traj_action_batch', 'sim_traj_pose_batch',
                             'sim_traj_goal_batch', 'sim_traj_idx_batch', 'sim_traj_integer_time_batch', 'traj_dataset_next_traj_latent' ]

        traj_data_names = traj_data_names + ['sparsify_' + name for name in traj_data_names]
        for name in traj_data_names:
            data[name] = []

        if not self.full_traj_embedding:
            for name in traj_data_names:
                if name.replace('traj_', '') in data:
                    data[name] = data[name.replace('traj_', '')].copy()
                else:
                    data[name] = np.zeros_like(data['action_batch'])
            return data

        sampler_require_traj = self.sample_sim_traj
        compute_sim_traj_flag = self.use_simulated_plan and sampler_require_traj and not self.use_offline_latent
        onpolicy_rollout = self.name != 'expert' or not self.use_simulated_plan

        for i in range(len(batch_idx)):
            if episode_end[i] - batch_idx[i] <= 1 or batch_idx[i] < 0 or episode_end[i] < 50:
                if self.full_traj_embedding: print('min idx', (episode_end - batch_idx).min(), episode_end[i] - batch_idx[i], self.name)
                episode_end[i] = batch_idx[i] + 1
            end_idx = episode_end[i] + 1
            batch_idx_offset = self.batch_sequence_size if self.name != 'expert' else 0

            if episode_end[i] - batch_idx[i] <= 0:
                traj_idx_i = [[0 + i, 1, curr_traj_time[i]]]
                traj_joint_state = self.curr_joint[[batch_idx[i], batch_idx[i]]] # dummies
                data['traj_joint_batch'].append(traj_joint_state[1:] )
                data['next_traj_joint_batch'].append(self.curr_joint[[batch_idx[i]+1, batch_idx[i]+1]])
            else:
                batch_idx_offset = self.batch_sequence_size if self.name != 'expert' else 0
                traj_idx_i = [[batch_idx_offset + i, j / float(end_idx - batch_idx[i]), curr_traj_time[i] + (j + 1) / self.max_step] \
                             for j in range(end_idx - batch_idx[i])]
                traj_joint_state = self.curr_joint[batch_idx[i]: end_idx]
                data['traj_joint_batch'].append(traj_joint_state)
                data['next_traj_joint_batch'].append(self.curr_joint[batch_idx[i]+1: end_idx+1])

            traj_batch_i_parallel = np.matmul(se3_inverse(self.state_pose[batch_idx[i]])[None], self.state_pose[batch_idx[i]:end_idx])
            traj_batch_i = pack_pose_rot_first_batch(traj_batch_i_parallel)
            data['traj_idx_batch'].append(traj_idx_i)
            data['traj_action_batch'].append(self.action[batch_idx[i]:end_idx])
            data['traj_expert_action_batch'].append(self.expert_action[batch_idx[i]:end_idx])
            data['traj_goal_batch'].append(self.goal[batch_idx[i]:end_idx])
            data['traj_pose_batch'].append(traj_batch_i)

            data['traj_next_goal_batch'].append(self.goal[batch_idx[i]+1:end_idx+1])
            data['traj_point_state_batch'].append(self.point_state[batch_idx[i]:end_idx])
            data['traj_integer_time_batch'].append((self.timestep[self.episode_map[batch_idx[i]]]) + 1 - self.timestep[batch_idx[i]:end_idx]) #
            data['traj_mask_batch'].append(np.float32(self.terminal[batch_idx[i]:end_idx]))
            data['traj_reward_batch'].append(np.float32(self.reward[batch_idx[i]:end_idx]))
            data['traj_return_batch'].append(np.float32(self.returns[batch_idx[i]:end_idx]))
            data['traj_gaddpg_batch'].append(np.float32(self.gaddpg_flags[batch_idx[i]:end_idx]))
            data['next_traj_action_batch'].append(self.action[batch_idx[i]+1:end_idx+1])
            data['next_traj_point_state_batch'].append(self.point_state[batch_idx[i]+1:end_idx+1])
            data['traj_dataset_traj_latent'].append(self.dataset_traj_latent[batch_idx[i]:end_idx])
            data['traj_dataset_next_traj_latent'].append(self.dataset_traj_latent[batch_idx[i]+1:end_idx+1])
            data['traj_dataset_state_latent'].append(self.dataset_state_latent[batch_idx[i]:end_idx])
            data['traj_dataset_next_state_latent'].append(self.dataset_state_latent[batch_idx[i]+1:end_idx+1])

            if compute_sim_traj_flag: #
                sim_joints = traj_joint_state if onpolicy_rollout else self.expert_joint_plan[batch_idx[i]]
                if plan_length(sim_joints) < 2: sim_joints = traj_joint_state
                sim_states, sim_joints, sim_actions, sim_poses, sim_goals, sim_integer_time, sim_traj_idx = \
                                                                            generate_simulated_expert_trajectory(
                                                                                        data['point_state_batch'][i],
                                                                                        sim_joints,
                                                                                        self.curr_joint[batch_idx[i]],
                                                                                        curr_traj_time[i],
                                                                                        batch_idx_offset + i )

                prev_sim_states = sim_states.copy()
                prev_sim_states[1:] = prev_sim_states[:-1] # shift by 1
                data['sim_traj_point_state_batch'].append(sim_states)
                data['sim_traj_joint_batch'].append(sim_joints)
                data['sim_traj_action_batch'].append(sim_actions)
                data['sim_traj_pose_batch'].append(sim_poses)
                data['sim_traj_goal_batch'].append(sim_goals)
                data['sim_traj_idx_batch'].append(sim_traj_idx)
                data['sim_traj_integer_time_batch'].append(sim_integer_time)

            else:
                for data_name in ['traj_point_state_batch', 'traj_joint_batch', 'traj_action_batch', 'traj_pose_batch',
                                  'traj_goal_batch', 'traj_idx_batch', 'traj_integer_time_batch']:
                    data['sim_' + data_name].append(data[data_name][-1].copy())
            self.sparsify_traj_data(data, traj_data_names, True, episode_end[i], batch_idx[i] )

        for name in traj_data_names:
            if len(data[name]) >= 1:
                data[name] = np.concatenate(data[name], axis=0)
            else:
                data[name] = np.zeros(1)

    def sparsify_traj_data(self, data, data_names, compute_sim_traj_flag, end, start ):
        """ downsample traj data """
        traj_length = len(data[data_names[0]][-1])
        sim_traj_length = len(data['sim_traj_point_state_batch'][-1]) if 'sim_traj_point_state_batch' in data and compute_sim_traj_flag  else 0
        if self.sparsify_traj_ratio > 1   and traj_length > 2 and sim_traj_length > 2:
            downsample_traj_len = int(np.ceil(float(traj_length) / self.sparsify_traj_ratio))
            sim_downsample_traj_len = int(np.ceil(float(sim_traj_length) / self.sparsify_traj_ratio))
            sparsify_traj_idx = np.linspace(0, traj_length - 1, downsample_traj_len).astype(np.int)
            sim_sparsify_traj_idx = np.linspace(0, sim_traj_length - 1, sim_downsample_traj_len).astype(np.int)

        else:
            sparsify_traj_idx = range(len(data[data_names[0]][-1]))
            sim_sparsify_traj_idx = range(len(data['sim_traj_point_state_batch'][-1]))

        if self.sparsify_bc_ratio > 1 and traj_length > 2: #
            downsample_traj_len = int(np.ceil(float(traj_length) / self.sparsify_bc_ratio))
            valid_length = traj_length - 1
            sparsify_bc_idx = np.random.choice( np.arange(1, valid_length), downsample_traj_len,
                                                replace=downsample_traj_len >= valid_length).astype(np.int)
            sparsify_bc_idx[0] = 0  # include current

        else:
            sparsify_bc_idx = range(len(data[data_names[0]][-1])  )

        for name in data_names:
            if len(data[name]) < 1 : continue
            if 'sparsify' not in name:
                if 'sim' in name:
                    data['sparsify_' + name].append(np.stack([data[name][-1][i] for i in sim_sparsify_traj_idx], axis=0))
                else:
                    data['sparsify_' + name].append(np.stack([data[name][-1][i] for i in sparsify_traj_idx], axis=0))

                if self.sparsify_bc_ratio > 1 and 'sim' not in name:
                    data[name][-1] = np.stack([data[name][-1][i] for i in sparsify_bc_idx], axis=0)


    def post_process_batch(self, data, batch_idx):
        """
        Set some data in batch
        """
        end_idx = self.episode_map[batch_idx]
        increment_idx = np.minimum(end_idx, batch_idx + 1).astype(np.int)
        data["next_image_state_batch"] =  process_image_output(self.image_state[increment_idx])
        data["next_goal_batch"] = np.float32(self.goal[increment_idx])
        data["next_expert_action_batch"] = np.float32(self.expert_action[increment_idx])
        data["next_action_batch"] = np.float32( self.action[increment_idx])
        data["next_joint_batch"] = np.float32( self.curr_joint[increment_idx])[:,:7]
        data["goal_joint_batch"] = np.float32( self.curr_joint[end_idx] )[:,:7]
        data["next_return_batch"] = self.returns[increment_idx]
        data["next_point_state_batch"] =  self.point_state[increment_idx]
        data["next_link_collision_batch"] = np.float32(self.link_collision_distance[increment_idx])
        data["dataset_normal_latent"] = np.float32(self.dataset_normal_latent[batch_idx])
        data["point_state_batch"] =  self.point_state[batch_idx]
        data["goal_point_state_batch"] =  self.point_state[self.episode_map[batch_idx]]

        data["curr_traj_time_batch"] = np.float32(data["time_batch"].copy()) / self.max_step
        data["time_batch"] = np.float32(self.timestep[end_idx]) + 1 - data["time_batch"]
        data["expert_flag_batch"] = np.float32(self.expert_flags[batch_idx])
        data["perturb_flag_batch"] = self.perturb_flags[batch_idx] < 1
        data["batch_idx"] = np.uint8(batch_idx)
        data["gaddpg_batch"] = self.gaddpg_flags[batch_idx]
        data['curr_joint'] = data['curr_joint'][:, :7]
        data['dataset_traj_latent'] = self.dataset_traj_latent[batch_idx]
        data['dataset_state_latent'] = self.dataset_state_latent[batch_idx]
        self.compute_traj(data, batch_idx)


    def compute_episode_map(self):
        """ compute episode length and end indexes """
        nonzero_episode_map = self.episode_map[self.episode_map !=0]
        self.unique_episode_map, unique_indexes = np.unique(nonzero_episode_map, return_index=True)
        self.unique_episode_map = nonzero_episode_map[unique_indexes][1:-1]
        self.episode_length     = np.diff(nonzero_episode_map[unique_indexes])[:-1]

        traj_len = np.sum(np.abs(self.expert_joint_plan[nonzero_episode_map[unique_indexes][:-2]+1]).sum(-1) > 0, -1)
        plan_mask = traj_len == self.max_step
        self.unique_episode_map = self.unique_episode_map[plan_mask]
        self.episode_length  = self.episode_length[plan_mask]
        self.cur_episode_idx = 0


    def load(self, data_dir, buffer_size=100000, **kwargs):
        """
        Load data saved offline
        """
        print('load data dir:', data_dir, self.save_data_name)
        if not os.path.exists(data_dir):
            return

        if os.path.exists(os.path.join(data_dir, self.save_data_name)):
            data = np.load( os.path.join(data_dir, self.save_data_name),
                            allow_pickle=True,
                            mmap_mode="r" )
            data_max_idx = min(np.amax(data["episode_map"]), buffer_size)
            if data_max_idx == buffer_size: #
                unique_episode_idx = np.unique(data["episode_map"])
                data_max_idx = unique_episode_idx[unique_episode_idx < data_max_idx][-1]
            print('data max: {} buffer size: {}'.format(data_max_idx, buffer_size))
            for name in self.attr_names + [ "episode_map",  "target_idx" ]:
                s = time.time()
                print("loading {} ...".format(name))
                if  name == "image_state":
                    continue

                if name not in data:
                    print("not in data:", name)
                    continue

                if type(data[name]) is not np.ndarray:
                    setattr(self, name, data[name])
                    print(name, getattr(self, name))
                else:
                    try:
                        getattr(self, name)[:data_max_idx] = data[name][:data_max_idx].copy()
                    except:
                        print('load data fail', name, data_max_idx)
                    print(name + " shape:", getattr(self, name).shape)
                print("load {} time: {:.3f}".format(name, time.time() - s))

            self.cur_idx = min(data_max_idx, buffer_size)
            self.total_env_step = int(data["total_env_step"])
            self.is_full = ( bool(data["is_full"]) and self.cur_idx >= self.buffer_size - self.max_step)
            self.cur_idx = self.upper_idx()
            self.compute_episode_map()
            self.recompute_return_with_gamma()
            expert_upper_idx = self.get_expert_upper_idx()
            pos = np.where(self.returns[: self.cur_idx] > 0.0)[0][1:-1]
            expert_flag = self.expert_flags[: self.cur_idx] == 1

            # Load offline latent
            print( ("======================= name: {} loaded idx: {} env step: {} success: {:.3f} expert ratio: {:.3f}" + \
                    " start idx: {} expert end idx {} min rew: {} max rew: {} traj num: {} traj length mean: {} is_full: {} =======================").format(
                    os.path.join(data_dir, self.save_data_name),
                    self.upper_idx(),
                    self.total_env_step,
                    float(len(pos)) / (self.cur_idx + 1),
                    float(expert_flag.sum()) / (self.cur_idx + 1),
                    self.buffer_start_idx,
                    expert_upper_idx,
                    self.reward.min(),
                    self.reward.max(),
                    len(self.unique_episode_map),
                    self.episode_length.mean(),
                    self.is_full,
                )
            )

    def save(self, save_dir="."):
        """Saves the current buffer to memory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        s = time.time()
        save_dict = {}
        save_attrs = self.attr_names + [
            "episode_map",
            "is_full",
            "cur_idx",
            "total_env_step",
            "target_idx",
        ]
        for name in save_attrs:
            save_dict[name] = getattr(self, name)

        np.savez(os.path.join(save_dir, self.save_data_name), **save_dict)
        print("Saving buffer at {} {}, time: {:.3f}".format(save_dir, self.save_data_name, time.time() - s))

    def init_buffer(self):
        """ initialize buffer fields """
        state_size = (1,)
        action_size = (6,)
        pose_size = (64,)
        point_size = self.uniform_num_pts + 6 + 500
        point_dim  = 4
        self.image_state = np.zeros((self.buffer_size,) + state_size, dtype=np.uint16)
        self.action = np.zeros((self.buffer_size,) + action_size, dtype=np.float32)
        self.expert_action = np.zeros((self.buffer_size,) + action_size, dtype=np.float32)

        self.terminal = np.zeros((self.buffer_size,), dtype=np.float32)
        self.timestep = np.zeros((self.buffer_size,), dtype=np.float32)
        self.reward = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pose = np.zeros((self.buffer_size,) + pose_size, dtype=np.float32)
        self.point_state = np.zeros([self.buffer_size, point_dim, point_size])
        self.collide = np.zeros((self.buffer_size,), dtype=np.float32)
        self.grasp = np.zeros((self.buffer_size,), dtype=np.float32)
        self.state_pose = np.zeros((self.buffer_size, 4, 4), dtype=np.float32)
        self.target_idx = np.zeros((self.buffer_size,), dtype=np.float32)
        self.goal = np.zeros((self.buffer_size, 7), dtype=np.float32)
        self.episode_map = np.zeros((self.buffer_size,), dtype=np.uint32)
        self.expert_flags = np.zeros((self.buffer_size,), dtype=np.float32)
        self.perturb_flags = np.zeros((self.buffer_size,), dtype=np.float32)
        self.plan_goal_change_flags = np.zeros((self.buffer_size,), dtype=np.float32)
        self.gaddpg_flags = np.zeros((self.buffer_size,), dtype=np.float32)

        self.joint_action_delta = np.zeros((self.buffer_size, 9), dtype=np.float32)
        self.curr_joint = np.zeros((self.buffer_size,9), dtype=np.float32)
        self.expert_joint_action_delta = np.zeros((self.buffer_size,9), dtype=np.float32)
        self.link_collision_distance = np.zeros((self.buffer_size, 10), dtype=np.float32)
        self.link_collision_dist_vec = np.zeros((self.buffer_size, 10, 3), dtype=np.float32)
        self.expert_joint_plan  = np.zeros((self.buffer_size, self.max_step, 9), dtype=np.float32)
        self.unique_episode_map = np.zeros((self.episode_buffer_size,), dtype=np.uint16)
        self.episode_length = np.zeros((self.episode_buffer_size,), dtype=np.uint16)
        self.scene_idx = np.zeros((self.buffer_size,), dtype=np.int16)

        self.unique_scene_idx = np.zeros((self.buffer_size,), dtype=np.int16)
        self.dataset_traj_latent  = np.zeros((self.buffer_size, self.policy_traj_latent_size), dtype=np.float32)
        self.dataset_normal_latent = np.zeros((self.buffer_size, self.normal_vae_dim), dtype=np.float32)
        self.dataset_state_latent = np.zeros((self.buffer_size, self.feature_input_dim + 1), dtype=np.float32)

