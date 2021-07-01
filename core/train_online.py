# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime
import numpy as np
import itertools

import torch
from core.bc import BC
from core.ddpg import DDPG
from core.dqn_hrl import DQN_HRL
from core.utils import *
from core.trainer import *

from tensorboardX import SummaryWriter
from env.panda_cluttered_scene import PandaYCBEnv, PandaTaskSpace6D
from env.env_planner import EnvPlanner
from experiments.config import *

import json
import time
import tabulate
import scipy.io as sio
import IPython
import pprint
import glob
import ray
import yaml
import random
import psutil
import GPUtil

def create_parser():
    parser = argparse.ArgumentParser(description='Train Online Args')
    parser.add_argument('--env-name', default="PandaYCBEnv" )
    parser.add_argument('--policy', default="SAC", )
    parser.add_argument('--seed', type=int, default=233, metavar='N')
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--pretrained', type=str, default=None, help='use a pretrained model')

    parser.add_argument('--log', action="store_true", help='log loss')
    parser.add_argument('--model_surfix',  type=str, default='latest')
    parser.add_argument('--save_buffer', action="store_true")
    parser.add_argument('--save_online_buffer', action="store_true")

    parser.add_argument('--config_file', type=str, default="bc.yaml")
    parser.add_argument('--visdom', action="store_true")
    parser.add_argument('--max_load_scene_num', type=int, default=-1)
    parser.add_argument('--load_buffer', action="store_true")
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--save_scene', action="store_true")
    parser.add_argument('--load_scene', action="store_true")
    parser.add_argument('--load_online_buffer', action="store_true", help='load online buffer')
    parser.add_argument('--pretrained_policy_name', type=str, default='BC')

    return parser

def sample_experiment_objects(scene_file=None):
    """
    Sample objects from the json files for replay buffer and environment
    """
    print('scene_file:', scene_file)
    index_file = CONFIG.index_file.split('.json')[0].split('/')[-1]
    index_file = os.path.join(cfg.EXPERIMENT_OBJ_INDEX_DIR, index_file + '.json')

    file_index = json.load(open(index_file))
    if CONFIG.index_split in file_index:
        file_index = file_index[CONFIG.index_split]
    file_dir = [f[:-5].split('.')[0][:-2] if 'json' in f else f for f in file_index ]
    sample_index = np.random.choice(range(len(file_dir)), min(LOAD_OBJ_NUM, len(file_dir)), replace=False).astype(np.int)
    file_dir = [file_dir[idx] for idx in sample_index]

    file_dir = list(set(file_dir))
    print('training object index: {} obj num: {}'.format(index_file, len(file_dir)))
    return file_dir

def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """
    load_from_pretrain = args.pretrained is not None and os.path.exists(os.path.join(args.pretrained, "config.yaml"))

    if load_from_pretrain:
        cfg_folder = args.pretrained
        if os.path.exists(os.path.join(cfg_folder, "config.yaml")):
            cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)

    if args.fix_output_time is None:
        dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    else:
        dt_string = args.fix_output_time

    model_output_dir = os.path.join(cfg.OUTPUT_DIR, dt_string)
    new_output_dir = not os.path.exists(model_output_dir)
    if new_output_dir:
        os.makedirs(model_output_dir)

    if args.config_file is not None:
        script_file = os.path.join(cfg.SCRIPT_FOLDER, args.config_file)
        if os.path.exists(script_file):
            cfg_from_file(script_file)
        cfg.script_name = args.config_file
        os.system( "cp {} {}".format(script_file, os.path.join(model_output_dir, args.config_file) ))

    if new_output_dir:
        os.system("cp {} {}".format( cfg.RL_MODEL_SPEC, os.path.join(model_output_dir, cfg.RL_MODEL_SPEC.split("/")[-1])))
        if load_from_pretrain:
            cfg.pretrained_time = args.pretrained.split("/")[-1]
            migrate_model( args.pretrained, model_output_dir, args.model_surfix, args.pretrained_policy_name, POLICY )
            print("migrate policy...")
        save_cfg_to_file(os.path.join(model_output_dir, "config.yaml"), cfg)

    print("Output will be saved to `{:s}`".format(model_output_dir))
    print("Using config:")
    pprint.pprint(cfg)
    return dt_string

def get_buffer_log():
    """Get gpu and memory usages as well as current performance """
    reward_info, online_reward_info = np.array(ray.get(buffer_id.reward_info.remote())), np.array(ray.get(online_buffer_id.reward_info.remote()))
    return [(reward_info[i], online_reward_info[i]) for i in range(len(reward_info))]

class ActorWrapper(object):
    """
    Wrapper class for actors to do rollouts and save data, to collect data while training
    """
    def __init__(self,  learner_id, buffer_remote_id, online_buffer_remote_id, gaddpg_agent_id, unique_id):
        from env.panda_cluttered_scene import PandaYCBEnv
        self.learner_id = learner_id
        self.buffer_id = buffer_remote_id
        self.unique_id = unique_id
        self.online_buffer_id = online_buffer_remote_id

        self.env = eval(CONFIG.env_name)(**cfg.env_config)
        self.target_obj_list = []
        np.random.seed(args.seed + unique_id)
        self.gaddpg_agent_id = gaddpg_agent_id
        self.reset_env()

        if VISDOM:
            self.vis = Visdom(port=8097)
            self.win_id = self.vis.image(np.zeros([3, int(cfg.RL_IMG_SIZE[0]), int(cfg.RL_IMG_SIZE[1])]))
        self._TOTAL_CNT, self._TOTAL_REW = 1, 0
        self.offset_pose, self.K = get_camera_constant()

    def reset_env(self):
        """
        reset the environment by loading new objects
        """
        from env.panda_cluttered_scene import PandaYCBEnv
        from env.env_planner import EnvPlanner
        self.env = eval(CONFIG.env_name)(**cfg.env_config)
        np.random.seed(args.seed + self.unique_id)

        self.scene_indexes = None
        objects = sample_experiment_objects(self.scene_indexes)  if not CONFIG.shared_objects_across_worker else CONFIG.sampled_objs
        self.env._load_index_objs(objects, obj_root_dir='data/{}'.format(CONFIG.object_dir))
        self.env.reset( data_root_dir=cfg.DATA_ROOT_DIR, enforce_face_target=True)
        self.env_planner = EnvPlanner(self.env)

    def rollout_start(self):
        """
        sample a scene
        """
        self.data_root = 'data/' + SCENE_ROOT.split('/')[-1]
        state, check_scene_flag = sample_scene(self.env, self.env_planner, CONFIG=CONFIG)
        check_scene_flag = check_scene_flag and (self.env_planner.goal_num_full() )
        scene_idx = -1
        return state, check_scene_flag

    def save_scene(self, init_info):
        """
        save scene at local folder
        """
        rew = ray.get(self.buffer_id.get_total_lifted.remote())
        save_scene_name = os.path.join(cfg.RL_SAVE_SCENE_ROOT_DIR, 'scene_{}'.format(int(rew)))
        save_scene(self.env, save_scene_name, init_info)
        print('==== save scene:', save_scene_name)

    def get_flags(self, explore, step, test, expert_traj_length, perturb_traj, milestone_idx):
        """
        get different booleans for the current step
        """
        expert_flags  = float(not explore )
        dart_flags    = CONFIG.dart and \
                        (step > CONFIG.DART_MIN_STEP) and \
                        (step < min(CONFIG.DART_MAX_STEP, expert_traj_length - 8)) and \
                        (np.random.uniform() < CONFIG.DART_RATIO) and not explore and perturb_traj

        perturb_flags = dart_flags
        gaddpg_flags  = False

        if CONFIG.critic_gaddpg :
            if step == 0:
                gaddpg_prob = get_valid_index(CONFIG.gaddpg_ratio_list, milestone_idx)
                if np.random.uniform() < gaddpg_prob:
                    self.gaddpg_threshold_step = int(CONFIG.gaddpg_mean_remaining_step + np.random.normal() * CONFIG.gaddpg_std_remaining_step)
                else:
                    self.gaddpg_threshold_step = -10

            gaddpg_flags = CONFIG.max_step - step <= self.gaddpg_threshold_step
        return expert_flags, perturb_flags, dart_flags, gaddpg_flags

    def vis_rollout(self, simulate, next_state, curr_joint, goal_pose, aux_pred):
        """
        visualize single step in visdom
        """
        goal_involved = CONFIG.train_goal_feature or CONFIG.policy_aux or CONFIG.critic_aux
        hand_cam_intr = get_info(next_state, 'intr', cfg.RL_IMG_SIZE)
        img = draw_grasp_img(next_state[0][1][:3].transpose([2,1,0]), unpack_pose_rot_first(goal_pose),
                             hand_cam_intr, self.offset_pose, (0, 1., 0))
        if goal_involved and len(aux_pred) == 7:
            img = draw_grasp_img(next_state[0][1][:3].transpose([2,1,0]), unpack_pose_rot_first(aux_pred),
                             hand_cam_intr,  self.offset_pose, (1., 0., 0.))
        self.vis.image(img.transpose([2,0,1]), win=self.win_id)

    def gaddpg_action(self,  state, action, episode_steps, max_steps, curr_joint, explore, start_time=0.):
        """
        take gaddpg action
        """
        state = select_target_point(state)
        if state[0][0].shape[1] > 0:
            gaddpg_remain_step = max(max_steps-episode_steps + 1, 1)
            action = ray.get(self.gaddpg_agent_id.select_action.remote(state, remain_timestep=gaddpg_remain_step, curr_joint=curr_joint))[0]
        return action

    def get_plan_encoded_latent(self, state, expert_plan, last_joint):
        """
        encode the expert latent
        """
        expert_traj_data = generate_simulated_expert_trajectory(state[0][0], pad_traj_plan(expert_plan), last_joint)
        update_expert_traj(self.learner_id, expert_traj_data, cfg, remote=True)
        return ray.get(self.learner_id.get_traj_latent.remote())

    def rollout(self, num_episodes=1, explore=False, dagger=False, test=False, noise_scale=1., simulate=False, milestone_idx=0):
        """
        policy rollout and save data
        """
        for _ in range(num_episodes):

            # init scene
            state, check_scene_flag = self.rollout_start()
            if not check_scene_flag or state is None:  return [0]
            cur_episode = []
            step, reward, done = 0., 0., False
            expert_plan, exp_failure, exp_success = [], False, True
            expert_plan, exp_failure, exp_success = self.env_planner.expert_plan(return_success=True, checked=True)
            expert_traj_length = len(expert_plan)
            if expert_traj_length >= EXTEND_MAX_STEP or expert_traj_length < 5 and not explore:
                return [0]

            # attributes
            buffer_id = self.online_buffer_id if ON_POLICY and explore else self.buffer_id
            noise = get_noise_delta(np.zeros(6), CONFIG.action_noise, 'uniform' if explore else 'normal')
            last_joint = np.array(self.env._panda.getJointStates()[0])
            init_info = self.env._get_init_info()
            collided = False
            aux_pred = np.zeros(0)
            grasp = 0
            expert_joint_action = np.zeros(9)
            traj_latent = np.zeros(CONFIG.policy_traj_latent_size)
            traj_normal_latent = np.zeros(CONFIG.normal_vae_dim)
            dataset_state_latent = np.zeros(CONFIG.feature_input_dim + 1)
            first_latent_step = False
            if CONFIG.save_expert_plan_latent_online and not explore:
                traj_latent, _ = self.get_plan_encoded_latent(state, expert_plan, last_joint)

            # rollout
            while not done:

                # plan
                last_joint = np.array(self.env._panda.getJointStates()[0])
                expert_flags, perturb_flags, dart_flags, gaddpg_flags = self.get_flags(
                                             explore, step, test, expert_traj_length, not explore, milestone_idx)
                expert_step =  not explore
                if dart_flags:
                    self.env.random_perturb()
                    rest_expert_plan, exp_failure, exp_success = self.env_planner.expert_plan(
                                                                 return_success=True,
                                                                 step=int(MAX_STEP-step-1),
                                                                 fix_goal=CONFIG.expert_fix_goal )
                    expert_plan = np.concatenate((expert_plan[:int(step)], rest_expert_plan), axis=0)
                    expert_traj_length = len(expert_plan)
                    if CONFIG.save_expert_plan_latent_online and len(rest_expert_plan) > 0:
                        traj_latent, _ = self.get_plan_encoded_latent(state, rest_expert_plan, last_joint)

                remain_timestep = max(expert_traj_length-step, 1)
                goal_pose = self.env._get_relative_goal_pose(nearest=explore)
                if step < len(expert_plan): expert_joint_action = expert_plan[int(step)]
                expert_joint_action_delta = (expert_joint_action-last_joint)
                expert_action = joint_to_cartesian(expert_joint_action, last_joint)
                padded_expert_plan = pad_traj_plan(expert_plan[int(step):], CONFIG.max_step)

                # expert
                if  expert_step:
                    grasp = step == len(expert_plan) - 1
                    action = expert_action

                # agent
                else:
                    gt_traj = None if   not first_latent_step or (step < CONFIG.max_step - CONFIG.gaddpg_max_step \
                                        and np.random.uniform() < CONFIG.online_traj_sample_prob) else 1   #

                    first_latent_step = True
                    action_mean, log_probs, gaddpg_pred, aux_pred  = ray.get(
                                                    self.learner_id.select_action.remote(state,
                                                    goal_state=goal_pose, remain_timestep=remain_timestep,
                                                    curr_joint=last_joint, gt_traj=gt_traj ))

                    if  gt_traj is None: # sample new latent
                        traj_latent, traj_normal_latent = ray.get(self.learner_id.get_traj_latent.remote())

                    if gaddpg_flags: # use gaddpg action
                        action_mean = self.gaddpg_action(state, action, step, len(expert_plan), last_joint, explore)

                    action_mean = action_mean + noise * noise_scale
                    action = action_mean

                # env step
                next_state, reward, done, _ = self.env.step(action, delta=True)
                collided = self.env.obstacle_collided
                curr_joint  = np.array(self.env._panda.getJointStates()[0])

                # extra
                if VISDOM:
                    self.vis_rollout(simulate, next_state, curr_joint, goal_pose, aux_pred)

                if (not explore and step == expert_traj_length - 1) or step == EXTEND_MAX_STEP or done:
                    if not collided:
                        reward, res_obs, _ = self.env.retract(record=True)
                        if VISDOM:
                            for r in res_obs:  self.vis.image(r[0][1][:3].transpose([0,2,1]), win=self.win_id)  #
                    done = True

                # log
                step_dict = {
                            'point_state': state[0][0],
                            'image_state': state[0][1][None],
                            'action': action[None],
                            'reward': reward,
                            'returns': reward,
                            'terminal': done,
                            'timestep': step,
                            'pose': state[2],
                            'target_pose': state[-1][0],
                            'state_pose': state[-1][1],
                            'target_idx': self.env.target_idx,
                            'target_name': self.env.target_name,
                            'collide': collided,
                            'joint_action_delta': (curr_joint-last_joint),
                            'curr_joint': last_joint,
                            'grasp': grasp,
                            'goal': goal_pose,
                            'expert_action': expert_action[None],
                            'expert_joint_action_delta': expert_joint_action_delta,
                            'expert_joint_plan':  padded_expert_plan,
                            'expert_flags': expert_flags,
                            'perturb_flags': perturb_flags,
                            'gaddpg_flags': gaddpg_flags,
                            'dataset_traj_latent': traj_latent,
                            'dataset_normal_latent': traj_normal_latent,
                            'dataset_state_latent': dataset_state_latent
                            }

                cur_episode.append(step_dict)
                step = step + 1.
                state = next_state

            buffer_id.add_episode.remote(cur_episode, explore, test)
            if args.save_scene and reward > 0 and not collided and not explore:
                self.save_scene(init_info)

        return [reward]


@ray.remote(num_cpus=1, num_gpus=0.05)
class ActorWrapper005(ActorWrapper):
    pass

@ray.remote(num_cpus=1, num_gpus=0.10)
class ActorWrapper008(ActorWrapper):
    pass

@ray.remote(num_cpus=1, num_gpus=0.07)
class ActorWrapper006(ActorWrapper):
    pass



def reinit(reset=False):
    print_and_write(None, '============================ Reinit ==========================')
    time.sleep(4)
    CONFIG.sampled_objs = sample_experiment_objects()
    rollouts = [actor.reset_env.remote() for i, actor in enumerate(actors)]
    res = ray.get(rollouts)
    gpu_usage, memory_usage = get_usage()
    gpu_max = float(gpu_usage) / gpu_limit > 0.98
    memory_max = memory_usage >= MEMORY_THRE
    print('==================== Memory: {} GPU: {} ====================='.format(memory_usage, gpu_usage))

    if  reset:
        os.system('nvidia-smi')
        print_and_write(None, '===================== Ray Reinit =================')
        ray.get(learner_id.save_model.remote())
        time.sleep(10)
        ray.shutdown()
        time.sleep(2)
        return get_ray_objects(reinit=True)

    print_and_write(None, '==============================================================')


def get_ray_objects(reinit=False):
    rollout_agent_wrapper = RolloutAgentWrapperGPU1
    gpu_usage, memory_usage = get_usage()
    print('==================== Reset Memory: {} GPU: {} ====================='.format(memory_usage, gpu_usage))

    ray.init(num_cpus=3 * NUM_REMOTES, object_store_memory=object_store_memory, webui_host="0.0.0.0")
    buffer_id = ReplayMemoryWrapper.remote(int(cfg.RL_MEMORY_SIZE), cfg, 'expert')
    if LOAD_MEMORY:
        ray.get(buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, int(cfg.RL_MEMORY_SIZE)))
    if ON_POLICY:
        buffer_size = cfg.ONPOLICY_MEMORY_SIZE if cfg.ONPOLICY_MEMORY_SIZE > 0 else cfg.RL_MEMORY_SIZE
        online_buffer_id = ReplayMemoryWrapper.remote(int(buffer_size), cfg, 'online')
        if args.load_online_buffer:
            ray.get(online_buffer_id.load.remote(cfg.RL_SAVE_DATA_ROOT_DIR, int(buffer_size) ))
    else:
        online_buffer_id = ReplayMemoryWrapper.remote(1, cfg, 'online') # dummy

    if reinit:
        learner_id = agent_wrapper.remote(args, cfg, init_pretrained_path,
                                          input_dim, logdir, True, args.model_surfix, model_output_dir)
        rollout_agent_ids = [rollout_agent_wrapper.remote(args, cfg,  init_pretrained_path,
                                          input_dim, None, True, args.model_surfix, model_output_dir) ]
        gaddpg_agent_id  = get_gaddpg_agent()
    else:
        learner_id = agent_wrapper.remote(args, cfg, pretrained_path, input_dim, None)
        rollout_agent_ids = [rollout_agent_wrapper.remote(args, cfg, init_pretrained_path,
                                input_dim, None, True, args.model_surfix, model_output_dir) ]
        gaddpg_agent_id  = get_gaddpg_agent()

    trainer = TrainerRemote.remote(args, cfg, learner_id, buffer_id, online_buffer_id, logdir, model_output_dir)
    CONFIG.sampled_objs = sample_experiment_objects()
    actors = [actor_wrapper.remote(rollout_agent_ids[0], buffer_id, online_buffer_id, gaddpg_agent_id, actor_idx) for actor_idx in range(NUM_REMOTES)]
    return actors, rollout_agent_ids, learner_id, trainer, buffer_id, online_buffer_id

def log_info():
    actor_name = 'ONLINE' if explore else 'EXPERT'
    rollout_time = time.time() - start_rollout
    gpu_usage, memory_usage = get_usage()
    print_and_write(None, '===== Epoch: {} | Actor: {} | Worker: {} | Explore: {:.4f}  ======'.format(
                                                reinit_count, actor_name, NUM_REMOTES, explore_ratio  ))
    print_and_write(None, ( 'TIME: {:.2f} MEMORY: {:.1f} GPU: {:.0f}  REWARD {:.3f}/{:.3f} ' + \
                            'COLLISION {:.3f}/{:.3f} SUCCESS {:.3f}/{:.3f}\n' + \
                            'DATE: {} BATCH: {}').format(
                            rollout_time, memory_usage, gpu_usage,  buffer_log[1][0],
                            buffer_log[1][1], buffer_log[4][0], buffer_log[4][1],
                            buffer_log[5][0], buffer_log[5][1],
                            datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"), CONFIG.batch_size))
    print_and_write(None, '===========================================================================')
    gpu_max = (float(gpu_usage) / gpu_limit) > 0.98
    memory_max = memory_usage >= MEMORY_THRE
    iter_max = (train_iter + 4) % (reinit_interval) == 0
    return gpu_max, memory_max, iter_max

def choose_setup():
    NUM_REMOTES = int(min((100. - memory_usage) / 5, CONFIG.num_remotes))
    max_memory = 25
    NUM_REMOTES = CONFIG.num_remotes
    CONFIG.batch_size = cfg.OFFLINE_BATCH_SIZE

    if CLUSTER:
        actor_wrapper = ActorWrapper005
        agent_wrapper = AgentWrapperGPU2

    else:
        NUM_REMOTES = CONFIG.num_remotes
        CONFIG.batch_size = cfg.OFFLINE_BATCH_SIZE
        actor_wrapper = ActorWrapper006
        agent_wrapper = AgentWrapperGPU2 # 2

    print('update batch size: {} worker: {} memory: {}'.format(CONFIG.batch_size, NUM_REMOTES, max_memory))
    return actor_wrapper, agent_wrapper, max_memory, NUM_REMOTES

def start_log():
    logdir = '{}/{}/{}_{}'.format(cfg.OUTPUT_DIR,output_time, CONFIG.env_name, POLICY)
    CONFIG.output_time = output_time
    CONFIG.model_output_dir = model_output_dir
    CONFIG.logdir = logdir
    CONFIG.CLUSTER = CLUSTER
    CONFIG.ON_POLICY = ON_POLICY
    pretrained_path = os.path.join(cfg.OUTPUT_DIR, output_time)
    init_pretrained_path = pretrained_path
    print('output_time: {} logdir: {}'.format(output_time, logdir))
    return pretrained_path, logdir, init_pretrained_path

if __name__ == "__main__":
    # config
    parser = create_parser()
    args, other_args = parser.parse_known_args()
    BC  = 'BC' in args.policy
    POLICY = args.policy
    CONFIG = cfg.RL_TRAIN
    CONFIG.RL = False if BC else True
    output_time = setup()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    MAX_STEP = cfg.RL_MAX_STEP
    LOAD_OBJ_NUM = CONFIG.load_obj_num
    DAGGER_RATIO = CONFIG.DAGGER_RATIO
    SAVE_EPISODE_INTERVAL = CONFIG.SAVE_EPISODE_INTERVAL
    LOAD_MEMORY = args.load_buffer or CONFIG.load_buffer
    ON_POLICY = CONFIG.onpolicy
    SAVE_DATA = args.save_buffer
    SAVE_ONLINE_DATA = args.save_online_buffer
    LOAD_SCENE = args.load_scene
    MERGE_EVERY = 1
    EXTEND_MAX_STEP = MAX_STEP + 2

    # cpu and gpu selection
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])

    CLUSTER = check_ngc()
    MEMORY_THRE = 92
    VISDOM = args.visdom
    SCENE_ROOT  =  cfg.RL_SAVE_SCENE_ROOT_DIR
    actor_wrapper, agent_wrapper, max_memory, NUM_REMOTES = choose_setup()

    # hyperparameters
    object_store_memory = int(max_memory * 1e9)
    if CONFIG.index_file ==  'ycb_large.json': CONFIG.reinit_factor = 1000

    reinit_interval = int(LOAD_OBJ_NUM * CONFIG.reinit_factor)
    input_dim = CONFIG.feature_input_dim
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)

    # log
    pretrained_path, logdir, init_pretrained_path = start_log()
    if VISDOM :
        from visdom import Visdom
        vis = Visdom(port=8097 )
        vis.close(None)

    # ray objects
    actors, rollout_agent_id, learner_id, trainer, buffer_id, online_buffer_id = get_ray_objects()
    weights = ray.get(learner_id.get_weight.remote())

    # online training
    os.system('nvidia-smi')
    reinit_count, online_buffer_curr_idx, online_buffer_upper_idx, online_env_step = 0, 0, 0, 0

    for train_iter in itertools.count(1):
        start_rollout = time.time()
        incr_agent_update_step, agent_update_step = ray.get([learner_id.get_agent_incr_update_step.remote(), learner_id.get_agent_update_step.remote()])
        milestone_idx = int((incr_agent_update_step > np.array(CONFIG.mix_milestones)).sum())
        explore_ratio = min(get_valid_index(CONFIG.explore_ratio_list, milestone_idx), CONFIG.explore_cap)
        explore = (np.random.uniform() < explore_ratio) #
        noise_scale = CONFIG.action_noise

        ######################### Rollout and Train
        test_rollout = False
        rollouts = []
        rollouts.extend([actor.rollout.remote(MERGE_EVERY, explore, False, test_rollout, noise_scale, False, milestone_idx) for i, actor in enumerate(actors)])
        rollouts.extend([trainer.train_iter.remote()])
        rollouts.extend([rollout_agent_id_.load_weight.remote(weights) for rollout_agent_id_ in rollout_agent_id])
        rollouts.extend([learner_id.get_weight.remote()])
        res = ray.get(rollouts)
        weights = res[-1]

        ######################### Check Reinit
        buffer_is_full = ray.get(buffer_id.get_info.remote())[2]
        if ON_POLICY: online_buffer_is_full = ray.get(online_buffer_id.get_info.remote())[2]
        buffer_log = get_buffer_log()
        trainer.write_buffer_info.remote(buffer_log)
        trainer.write_external_info.remote( reinit_count=reinit_count,
                                            explore_ratio=explore_ratio)
        gpu_max, memory_max, iter_max = log_info()

        if  iter_max:
            reinit()
            reinit_count += 1

        if  memory_max:
            actors, rollout_agent_id, learner_id, trainer, buffer_id, online_buffer_id = reinit(reset=True)

        ######################### Exit
        if (SAVE_DATA and buffer_is_full):
            ray.get(buffer_id.save.remote(cfg.RL_SAVE_DATA_ROOT_DIR))
            break

        if  ON_POLICY and SAVE_ONLINE_DATA and online_buffer_is_full:
            ray.get(online_buffer_id.save.remote(cfg.RL_SAVE_DATA_ROOT_DIR))
            break

        if agent_update_step >= CONFIG.max_epoch:
            break


