# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Tuning online training scale based on local GPU and memory limits. The code is test on 4
V100 GPU, and 100 GB CPU memory. 2 GPUs are used for actor rollout and the other two for training.

The configs that can be adjusted:
num_remotes, batch_size, RL_MEMORY_SIZE, @ray.remote(num_cpus=*, num_gpus=*)
"""


import os
import os.path as osp
import numpy as np
import math
import tabulate
from easydict import EasyDict as edict
import IPython
import yaml
import torch
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# create output folders

if not os.path.exists(os.path.join(root_dir, 'output')):
    os.makedirs(os.path.join(root_dir, 'output'))
if not os.path.exists(os.path.join(root_dir, 'output_misc')):
    os.makedirs(os.path.join(root_dir, 'output_misc/rl_output_stat'))

__C = edict()
cfg = __C

# Global options
#
__C.script_name = ''
__C.pretrained_time = ''

__C.RNG_SEED = 3
__C.EPOCHS = 200
__C.ROOT_DIR = root_dir + '/'
__C.DATA_ROOT_DIR =  'data/scenes'
__C.ROBOT_DATA_DIR = 'data/robots'
__C.OBJECT_DATA_DIR =  'data/objects'
__C.OUTPUT_DIR = 'output'  # temp_model_output
__C.OUTPUT_MISC_DIR = 'output_misc'
__C.MODEL_SPEC_DIR = "experiments/model_spec"
__C.EXPERIMENT_OBJ_INDEX_DIR = "experiments/object_index"
__C.LOG = True
__C.MAX_SCENE_INDEX = 18000
__C.IMG_SIZE = (112, 112)

__C.RL_IMG_SIZE = (112, 112)
__C.RL_MAX_STEP = 20
__C.RL_DATA_ROOT_DIR = __C.DATA_ROOT_DIR
__C.ONPOLICY_MEMORY_SIZE = -1
__C.RL_MEMORY_SIZE = 100000
__C.OFFLINE_RL_MEMORY_SIZE = 100000
__C.OFFLINE_BATCH_SIZE = 180

__C.RL_SAVE_DATA_ROOT_DIR = 'data/offline_data'
__C.RL_SAVE_SCENE_ROOT_DIR = 'data/online_scenes'
__C.SCRIPT_FOLDER = 'experiments/cfgs'
__C.RL_SAVE_DATA_NAME = 'data_100k_exp.npz'
__C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR, 'rl_pointnet_model_spec.yaml')
__C.RL_TEST_SCENE = 'data/hcg_scenes'


# detailed options
#
__C.RL_TRAIN = edict()

# architecture and network hyperparameter
__C.RL_TRAIN.ONPOLICY_BUFFER_NAME = ''
__C.RL_TRAIN.clip_grad = 1.0
__C.RL_TRAIN.sampler_clip_grad = -1.0
__C.RL_TRAIN.gamma = 0.99
__C.RL_TRAIN.batch_size = 256
__C.RL_TRAIN.updates_per_step = 4
__C.RL_TRAIN.hidden_size = 256

__C.RL_TRAIN.tau = 0.0001
__C.RL_TRAIN.lr = 3e-4
__C.RL_TRAIN.reinit_lr = 1e-4
__C.RL_TRAIN.value_lr = 3e-4
__C.RL_TRAIN.lr_gamma = 0.5
__C.RL_TRAIN.value_lr_gamma = 0.5
__C.RL_TRAIN.head_lr = 3e-4
__C.RL_TRAIN.feature_input_dim = 512
__C.RL_TRAIN.ddpg_coefficients = [0.5, 0.001, 1.0003, 1., 0.2]
__C.RL_TRAIN.value_milestones = [20000, 40000, 60000, 80000]
__C.RL_TRAIN.policy_milestones = [20000, 40000, 60000, 80000]
__C.RL_TRAIN.mix_milestones = [20000, 40000, 60000, 80000]
__C.RL_TRAIN.mix_policy_ratio_list = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
__C.RL_TRAIN.mix_value_ratio_list = [1., 1., 1., 1., 1., 1., 1.]
__C.RL_TRAIN.policy_extra_latent = -1
__C.RL_TRAIN.critic_extra_latent = -1
__C.RL_TRAIN.save_epoch = [3000, 10000, 20000, 40000, 80000, 100000, 140000, 170000, 200000, 230000]
__C.RL_TRAIN.overwrite_feat_milestone = []
__C.RL_TRAIN.load_buffer = False
__C.RL_TRAIN.object_dir  = 'objects'

# algorithm hyperparameter
__C.RL_TRAIN.train_value_feature = True
__C.RL_TRAIN.train_feature = True
__C.RL_TRAIN.reinit_optim = False
__C.RL_TRAIN.off_policy = True
__C.RL_TRAIN.use_action_limit = True
__C.RL_TRAIN.sa_channel_concat = False
__C.RL_TRAIN.use_image = False
__C.RL_TRAIN.dagger = False
__C.RL_TRAIN.use_time = True
__C.RL_TRAIN.RL = True
__C.RL_TRAIN.value_model = False
__C.RL_TRAIN.shared_feature = False
__C.RL_TRAIN.policy_update_gap = 2
__C.RL_TRAIN.self_supervision = False
__C.RL_TRAIN.critic_goal = False
__C.RL_TRAIN.policy_aux = True
__C.RL_TRAIN.train_goal_feature = False
__C.RL_TRAIN.train_traj_feature = False
__C.RL_TRAIN.fix_traj_feature = False

__C.RL_TRAIN.critic_aux = True
__C.RL_TRAIN.policy_goal = False
__C.RL_TRAIN.goal_reward_flag = False
__C.RL_TRAIN.online_buffer_ratio = 0.
__C.RL_TRAIN.onpolicy = False
__C.RL_TRAIN.channel_num = 5
__C.RL_TRAIN.change_dynamics = False
__C.RL_TRAIN.pt_accumulate_ratio = 0.95
__C.RL_TRAIN.dart = False
__C.RL_TRAIN.accumulate_points = True
__C.RL_TRAIN.max_epoch = 120000
__C.RL_TRAIN.action_noise = 0.01
__C.RL_TRAIN.expert_fix_goal = True

# environment hyperparameter
__C.RL_TRAIN.load_obj_num = 25
__C.RL_TRAIN.reinit_factor = 2.
__C.RL_TRAIN.shared_objects_across_worker = False
__C.RL_TRAIN.target_update_interval = 3000
__C.RL_TRAIN.index_split = 'train'
__C.RL_TRAIN.env_name = 'PandaYCBEnv'
__C.RL_TRAIN.index_file = os.path.join(__C.EXPERIMENT_OBJ_INDEX_DIR, 'extra_shape.json')
__C.RL_TRAIN.max_num_pts = 20000
__C.RL_TRAIN.uniform_num_pts = 1024
__C.RL_TRAIN.initial_perturb = True

# exploration worker hyperparameter
__C.RL_TRAIN.num_remotes = 8
__C.RL_TRAIN.init_distance_low = 0.2
__C.RL_TRAIN.init_distance_high = 0.65
__C.RL_TRAIN.explore_ratio = 0.0
__C.RL_TRAIN.explore_cap = 1.0
__C.RL_TRAIN.expert_perturbation_ratio = 1
__C.RL_TRAIN.explore_ratio_list = [0.7]
__C.RL_TRAIN.gaddpg_ratio_list = [1.0]
__C.RL_TRAIN.epsilon_greedy_list = [0.8]
__C.RL_TRAIN.noise_ratio_list = [1.]
__C.RL_TRAIN.noise_type = 'uniform'
__C.RL_TRAIN.expert_initial_state = False
__C.RL_TRAIN.DAGGER_RATIO = 0.6
__C.RL_TRAIN.DART_MIN_STEP = 5
__C.RL_TRAIN.DART_MAX_STEP = 42
__C.RL_TRAIN.DART_RATIO = 0.35
__C.RL_TRAIN.SAVE_EPISODE_INTERVAL = 50
__C.RL_TRAIN.ENV_NEAR = 0.6
__C.RL_TRAIN.ENV_FAR  = 0.8

# misc hyperparameter
__C.RL_TRAIN.log = True
__C.RL_TRAIN.visdom = True
__C.RL_TRAIN.domain_randomization = False
__C.RL_TRAIN.buffer_full_size = -1
__C.RL_TRAIN.buffer_start_idx = 0
__C.RL_TRAIN.scene_sample_check_ik_cnt = 2
__C.RL_TRAIN.scene_sample_inner_cnt = 5
__C.RL_TRAIN.load_scene_recompute_ik = True
__C.RL_TRAIN.max_test_per_obj = 10

__C.RL_TRAIN.joint_point_state_input = True
__C.RL_TRAIN.traj_joint_point_state_input = True
__C.RL_TRAIN.feature_option = 2
__C.RL_TRAIN.st_feature_option = 4
__C.RL_TRAIN.traj_feature_option = 2
__C.RL_TRAIN.traj_feature_extractor_class = "STPointNetFeature"
__C.RL_TRAIN.traj_vae_feature_extractor_class = "PointTrajLatentNet"
__C.RL_TRAIN.state_feature_extractor = "PointNetFeature"
__C.RL_TRAIN.traj_goal_mutual_conditioned = True
__C.RL_TRAIN.traj_latent_size = 64
__C.RL_TRAIN.traj_sampler_latent_size = 1024
__C.RL_TRAIN.full_traj_embedding = False
__C.RL_TRAIN.batch_sequence_size = 4
__C.RL_TRAIN.sparsify_traj_ratio = 5
__C.RL_TRAIN.sparsify_bc_ratio = 6
__C.RL_TRAIN.batch_sequence_ratio = 1
__C.RL_TRAIN.use_simulated_plan = True

__C.RL_TRAIN.traj_net_lr = 1e-3
__C.RL_TRAIN.traj_sampler_net_lr = 3e-4
__C.RL_TRAIN.traj_loss_scale = 1.
__C.RL_TRAIN.latent_loss_scale = 1.
__C.RL_TRAIN.kl_scale = 1.
__C.RL_TRAIN.gaddpg_scale = 1.
__C.RL_TRAIN.traj_feat_recons_loss = True
__C.RL_TRAIN.policy_traj_latent_size = 64
__C.RL_TRAIN.gaddpg_mean_remaining_step = 8
__C.RL_TRAIN.gaddpg_std_remaining_step  = 2.
__C.RL_TRAIN.gaddpg_min_step = 5
__C.RL_TRAIN.gaddpg_max_step = 15
__C.RL_TRAIN.train_traj_sampler = False
__C.RL_TRAIN.use_sampler_latent = False
__C.RL_TRAIN.use_latent_reg_loss = False
__C.RL_TRAIN.sampler_extra_abs_time = False

__C.RL_TRAIN.critic_tanh = False
__C.RL_TRAIN.hrl_offline_latent = False
__C.RL_TRAIN.hrl_latent_action = False
__C.RL_TRAIN.regress_saved_latent_path = ''
__C.RL_TRAIN.use_dataset_policy_feat = False
__C.RL_TRAIN.traj_normal_vae = False
__C.RL_TRAIN.normal_vae_dim = 2
__C.RL_TRAIN.sample_sim_traj = True

__C.RL_TRAIN.save_expert_plan_latent_online = False
__C.RL_TRAIN.debug_latent_type = 'zero'
__C.RL_TRAIN.clip_tail_idx = 2
__C.RL_TRAIN.use_offline_latent = False
__C.RL_TRAIN.test_traj_num = 8
__C.RL_TRAIN.numObjects = 23
__C.RL_TRAIN.dqn = False
__C.RL_TRAIN.critic_mpc = False
__C.RL_TRAIN.critic_gaddpg = True
__C.RL_TRAIN.multi_traj_sample = False
__C.RL_TRAIN.ignore_traj = False
__C.RL_TRAIN.ignore_traj_sampler = False
__C.RL_TRAIN.re_sampler_step = False

__C.RL_TRAIN.gaddpg_prob = 1.
__C.RL_TRAIN.online_traj_sample_prob = 0.05
__C.RL_TRAIN.reinit_feat_opt = True
__C.RL_TRAIN.test_log_sigma_clip = 0.7
__C.RL_TRAIN.value_overwrite_lr = 1.0e-3
__C.RL_TRAIN.data_init_epoch = 1
__C.RL_TRAIN.fix_batch_gaddpg_latent = True
__C.RL_TRAIN.gaddpg_batch_ratio = 0.3
__C.RL_TRAIN.dqn_sample_num = 4


def process_cfg(reset_model_spec=True):
    """
    adapt configs
    """

    if reset_model_spec:
        __C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR , "rl_pointnet_model_spec.yaml" )

    __C.RL_MODEL_SPEC = os.path.join(__C.MODEL_SPEC_DIR , "rl_sampler_large_pointnet_model_spec.yaml")

    # overwrite for cluttered scene
    __C.RL_TRAIN.V =  [  [-0.9351, 0.3518, 0.0428, 0.3037],
                         [0.2065, 0.639, -0.741, 0.132],
                         [-0.2881, -0.684, -0.6702, 1.8803],
                         [0.0, 0.0, 0.0, 1.0]]
    __C.RL_TRAIN.max_step = 50
    __C.RL_TRAIN.uniform_num_pts = 4096
    __C.RL_IMG_SIZE = (224, 224)
    __C.RL_MAX_STEP = 50
    __C.RL_TRAIN.value_model = True

    # expert planner parameters
    __C.omg_config = {
        'traj_init':'grasp',
        'scene_file': '' ,
        'vis': False,
        'increment_iks': True ,
        'top_k_collision': 1200,
        'collision_point_num': 15,
        'uncheck_finger_collision': -1,
        'optim_steps': 10,
        'extra_smooth_steps': 15,
        'ol_alg' :'MD',
        'target_size': 1.0,
        'goal_idx' :-2,
        'traj_delta': 0.05,
        'standoff_dist': 0.08,
        'ik_clearance': 0.06,
        'clearance': 0.04,
        'goal_set_max_num': 50,
        'dist_eps': 2.0,
        'reach_tail_length': 7,
        'ik_parallel': False,
        'traj_max_step': int(__C.RL_MAX_STEP) + 6,
        'root_dir':  root_dir + "/",
        'traj_min_step' :int(__C.RL_MAX_STEP) - 5,
        'timesteps': int(__C.RL_MAX_STEP),
        'dynamic_timestep': False,
        'silent': True,
        'timeout': 1.2,
        'cam_V': __C.RL_TRAIN.V,
        'force_standoff': True,
        'z_upsample': True
    }

    # environment parameters
    __C.env_config = {
        "action_space": 'task6d',
        "data_type": 'RGBDM',
        "expert_step": int(__C.RL_MAX_STEP),
        "numObjects": __C.RL_TRAIN.numObjects,
        "width": __C.RL_IMG_SIZE[0],
        "height": __C.RL_IMG_SIZE[1],
        "img_resize": __C.RL_IMG_SIZE,
        "random_target": False,
        "use_hand_finger_point": True,
        "accumulate_points": __C.RL_TRAIN.accumulate_points,
        "uniform_num_pts": __C.RL_TRAIN.uniform_num_pts,
        "regularize_pc_point_count": True,
        "pt_accumulate_ratio": __C.RL_TRAIN.pt_accumulate_ratio,
        "omg_config": __C.omg_config,
        'use_normal':  False,
        'env_remove_table_pt_input': True,
    }
    compute_input_dim()

def compute_input_dim():
    def has_check(x, prop):
        return hasattr(x, prop) and getattr(x, prop)

    extra_pred_dim = 1
    critic_extra_pred_dim = 0
    num_inputs = 0

    if __C.RL_TRAIN.policy_aux:
        extra_pred_dim = 7

    if __C.RL_TRAIN.critic_aux:
        critic_extra_pred_dim = 7

    if __C.RL_TRAIN.use_time:
        num_inputs += 1

    if has_check(__C.RL_TRAIN, 'critic_gaddpg'):
        critic_extra_pred_dim += 1


    __C.RL_TRAIN.num_input_extra = num_inputs
    __C.RL_TRAIN.extra_pred_dim = extra_pred_dim
    __C.RL_TRAIN.critic_extra_pred_dim = critic_extra_pred_dim
    __C.RL_TRAIN.num_inputs = __C.RL_TRAIN.num_input_extra + __C.RL_TRAIN.hidden_size


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        if not k in b.keys():
            continue

        # the types must match, too
        if type(b[k]) is not type(v):
            continue

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename=None, dict=None, reset_model_spec=True):
    """Load a config file and merge it into the default options."""

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f))

    if not reset_model_spec:
        output_dir = "/".join(filename.split("/")[:-1])
        __C.RL_MODEL_SPEC = os.path.join(
            output_dir, yaml_cfg["RL_MODEL_SPEC"].split("/")[-1]
        )
    if dict is None:
        _merge_a_into_b(yaml_cfg, __C)
    else:
        _merge_a_into_b(yaml_cfg, dict)
    process_cfg(reset_model_spec=reset_model_spec)

def save_cfg_to_file(filename, cfg):
    """Load a config file and merge it into the default options."""
    with open(filename, 'w+') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def cfg_repr(cfg, fmt='plain'):
    def helper(d):
        ret = {}
        for k, v in d.items():
            if isinstance(v, dict):
                ret[k] = helper(v)
            else:
                ret[k] = v
        return tabulate.tabulate(ret.items(), tablefmt=fmt)
    return helper(cfg)
