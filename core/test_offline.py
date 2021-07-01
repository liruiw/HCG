# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime
import numpy as np
import torch
from core.bc import BC
from core.ddpg import DDPG

from core.dqn_hrl import DQN_HRL
from tensorboardX import SummaryWriter
from env.panda_cluttered_scene import PandaYCBEnv, PandaTaskSpace6D
from env.env_planner import EnvPlanner
from experiments.config import *
from core import networks
import glob
from core.utils import *
import json
import scipy.io as sio
import IPython
import pprint
import cv2
import GPUtil
import os

# camera params

def create_parser():
    parser = argparse.ArgumentParser(description= '')
    parser.add_argument('--env-name', default="PandaYCBEnv")
    parser.add_argument('--policy', default="DDPG" )
    parser.add_argument('--seed', type=int, default=123456, metavar='N' )

    parser.add_argument('--test_script_name', type=str, default='', help='test script name')
    parser.add_argument('--pretrained', type=str, default=None, help='test one model')
    parser.add_argument('--test', action="store_true", help='test one model')
    parser.add_argument('--log', action="store_true", help='log')
    parser.add_argument('--render', action="store_true", help='rendering')
    parser.add_argument('--record', action="store_true", help='record video')
    parser.add_argument('--test_episode_num', type=int, default=200, help='number of episodes to test')
    parser.add_argument('--expert', action="store_true", help='generate expert rollout')

    parser.add_argument('--num_runs',  type=int, default=3)
    parser.add_argument('--max_cnt_per_obj',  type=int, default=1)
    parser.add_argument('--model_surfix',  type=str, default='latest', help='surfix for loaded model')
    parser.add_argument('--load_test_scene', action="store_true", help='load pregenerated random scenes')
    parser.add_argument('--config_file',  type=str, default=None)
    parser.add_argument('--output_file',  type=str, default='rollout_success.txt')
    parser.add_argument('--fix_output_time', type=str, default=None)
    parser.add_argument('--use_gaddpg', action="store_true")
    parser.add_argument('--vis_traj_simulate', action="store_true")

    parser.add_argument('--use_sample_latent', action="store_true")
    parser.add_argument('--sample_latent_gap', type=int, default=1)
    parser.add_argument('--critic_mpc', action="store_true")
    parser.add_argument('--multi_traj_sample', action="store_true")
    return parser

def sample_experiment_objects():
    """
    Sample objects from the json files for test
    """
    file = os.path.join(cfg.EXPERIMENT_OBJ_INDEX_DIR, 'ycb_large_scene.json')
    with open(file) as f: file_dir = json.load(f)
    file_dir = file_dir[:TEST_NUM + 30]
    file_dir = list(set(file_dir))
    target_file_dir = [f for f in file_dir if check_filter_name(f)]
    return file_dir, target_file_dir

def setup():
    """
    Set up networks with pretrained models and config as well as data migration
    """
    load_from_pretrain = args.pretrained is not None and os.path.exists(args.pretrained)

    if load_from_pretrain  :
        cfg_folder = args.pretrained
        cfg_from_file(os.path.join(cfg_folder, "config.yaml"), reset_model_spec=False)
        cfg.RL_MODEL_SPEC = os.path.join(cfg_folder, cfg.RL_MODEL_SPEC.split("/")[-1])
        dt_string = args.pretrained.split("/")[-1]

    else:
        if args.fix_output_time is None:
            dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        else:
            dt_string = args.fix_output_time
    net_dict = make_nets_opts_schedulers(cfg.RL_MODEL_SPEC, cfg.RL_TRAIN)

    return net_dict, dt_string


def annotate_images( action, camera_param, traj_pred, last_joint, extra_pred, state,
                     aux_pred, gt_goal, vis_img, hand_cam_intr, camera_hand_offset_pose,
                     goal_involved, remain_timestep=-1, expert_joint=None, extra_text=None):
    """
    annotate on the bullet images
    """
    overhead_vis_img, extr1, intr1  = env.get_overhead_image_observation(*camera_param)
    cur_ef_pose = env._get_ef_pose(mat=True)
    base_pose = env.get_base_pose()
    overhead_extr = extr1.dot((cur_ef_pose))
    traj_offset_pose = None
    points = get_info(state, 'point')
    points = preprocess_points(CONFIG, points[None], last_joint[None], append_pc_time=True)[0].detach().cpu().numpy()
    overhead_vis_img = proj_point_img(overhead_vis_img, intr1, overhead_extr, points,
                                      neg_y=False, traj=traj_pred,
                                      joint_output=False,
                                      last_joint=last_joint,
                                      remain_timestep=remain_timestep,
                                      traj_offset_pose=traj_offset_pose,
                                      gt_goal=gt_goal,
                                      extra_text=extra_text,
                                      model_name=video_surfix)
    return overhead_vis_img, vis_img


def expert_rollout(state, expert_plan):
    """
    simulate and actual rollout expert plan
    """
    expert_traj, expert_overhead_traj, expert_traj_pc, expert_traj_joint_state, expert_traj_time, expert_traj_action, expert_traj_poses, expert_traj_integer_time = [], [], [], [], [], [], [], []
    hand_cam_intr = get_info(state, 'intr', cfg.RL_IMG_SIZE)
    last_joint =  np.array(env._panda.getJointStates()[0])
    expert_traj_data = generate_simulated_expert_trajectory(state[0][0], expert_plan, last_joint)

    # execute expert plan
    for plan_idx in range(len(expert_plan)):
        joint_action = expert_plan[plan_idx]
        goal_state = env._get_relative_goal_pose(mat=True)
        action = joint_to_cartesian(joint_action, last_joint)
        vis_img = get_info(state, 'img', cfg.RL_IMG_SIZE)
        remain_timestep = max(len(expert_plan)-plan_idx, 1)
        last_joint = np.array(env._panda.getJointStates()[0])
        overhead_vis_img, vis_img = annotate_images( action, camera_param, None, last_joint, None, state,
                                                     pack_pose_rot_first(goal_state), pack_pose_rot_first(goal_state), vis_img, hand_cam_intr,
                                                     camera_hand_offset_pose, goal_involved, remain_timestep )
        next_state, reward, done, _ = env.step(action)
        expert_traj.append(vis_img)
        expert_overhead_traj.append(overhead_vis_img)
        state = next_state

    # expert stats
    expert_episode_reward, res_obs, overhead_res_obs = env.retract(True, camera_param)
    exp_lifted = (expert_episode_reward > 0.5)
    collided = env.obstacle_collided and not exp_lifted
    reward = expert_episode_reward
    res_obs = [get_info(r, 'img', cfg.RL_IMG_SIZE) for r in res_obs]
    expert_traj.extend(res_obs)
    expert_overhead_traj.extend(overhead_res_obs)
    overhead_traj, traj = [], []

    return expert_traj_data, overhead_traj, traj, expert_traj, expert_overhead_traj, (exp_lifted, collided, reward)


def log_output(exp_success, exp_failure, episode_reward, collided,reward, exp_lifted, exp_collided,
               exp_reward, episode_steps, cnt, start_time, scene_file):
    """
    log single test episode output
    """
    lifted = (reward > 0.5)
    avg_reward.update(episode_reward)
    avg_lifted.update(lifted)
    avg_collided.update(collided)
    target_obj_list.append(env.target_name)
    traj_lengths.append(episode_steps)

    if env.target_name not in object_performance:
        object_performance[env.target_name] = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]

    object_performance[env.target_name][0].update(episode_reward)
    object_performance[env.target_name][1].update(lifted)
    object_performance[env.target_name][2].update(collided)
    object_performance[env.target_name][3].update(exp_lifted)
    object_performance[env.target_name][4].update(exp_collided)

    print('=======================================================================')
    print(( 'test: {} max steps: {}, episode steps: {}, return: {:.3f} time {:.3f} ' +
            'avg return: {:.3f}/{:.3f} collided: {:.3f} model: {} {} gen: {} epoch: {} script: {}').format(cnt,
            TOTAL_MAX_STEP,  episode_steps, episode_reward, time.time() - start_time,
            avg_reward.avg, avg_lifted.avg, avg_collided.avg, args.pretrained, cfg.script_name,
            args.use_sample_latent, agent.update_step - 1, args.test_script_name))
    print('=======================================================================')
    print('testing script:', args.output_file)

def write_result():
    """
    dump episode statistics to the output
    """
    dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    output_stat_file = os.path.join(cfg.OUTPUT_MISC_DIR, 'rl_output_stat', args.output_file)
    file_handle = open(output_stat_file, 'a+')
    output_text = ''
    output_text += print_and_write(file_handle, '\n')
    output_text += print_and_write(file_handle, "------------------------------------------------------------------")

    output_text += print_and_write(file_handle, 'Test Time: {} Data Root: {}/{} Pretrained: {} Script: {} Index: {} Test: {}'.format(dt_string, cfg.RL_DATA_ROOT_DIR, cfg.RL_SAVE_DATA_NAME, pretrained_model_path, cfg.script_name, CONFIG.index_file, args.test_script_name))
    output_text += print_and_write(file_handle, 'Num of Objs: {} Num of Runs: {}'.format(len(object_performance), NUM_RUNS))
    output_text += print_and_write(file_handle, 'Policy: {} Model Path: {} Step: {}'.format(POLICY,
                    args.pretrained, agent.update_step - 1 ))
    output_text += print_and_write(file_handle, "Test Episodes: {} Avg. Length: {:.3f} Index: {}-{}".format(
                    cnt, np.mean(traj_lengths), scene_indexes[0], scene_indexes[-1]))
    output_text += print_and_write(file_handle, 'Avg. Performance: (Return: {:.3f} +- {:.5f}) (Success: {:.3f} +- {:.5f})  (Collision: {:.3f} +- {:.5f})'.format(
                                    avg_reward.avg, avg_reward.std(), avg_lifted.avg, avg_lifted.std(), avg_collided.avg, avg_collided.std()))
    headers = ['object name', 'count',  'lifted', 'collided', 'exp_lifted','exp_collided']
    object_performance_list = sorted(object_performance.items())
    data = [(name, info[0].count,  int(info[1].sum), int(info[2].sum),  int(info[3].sum), int(info[4].sum))
            for name, info in object_performance_list ]
    obj_performance_str = tabulate.tabulate(data, headers, tablefmt='psql')
    output_text += print_and_write(file_handle, obj_performance_str)
    print('testing script:', args.output_file)

def test(run_iter=0):
    """
    test agent performance on test scenes
    """
    global cnt, total_cnt, object_performance, target_obj_list, exp_exec_success
    k = 1
    data_root = 'data/' + cfg.RL_TEST_SCENE.split('/')[-1]

    while (k < TEST_NUM):

        # sample scene
        start_time = time.time()
        traj, res_obs, overhead_traj = [], [], []
        expert_exec_traj, expert_overhead_traj = [], None
        scene_file = 'scene_{}'.format(int(k))
        k += 1

        state = env.reset(scene_file=scene_file, data_root_dir=data_root, reset_free=True)
        hand_cam_intr = get_info(state, 'intr', cfg.RL_IMG_SIZE)
        cur_ef_pose = env._get_ef_pose(mat=True)
        scene_indexes.append(scene_file.split('/')[-1])
        env_planner.set_planner_scene_file(data_root, scene_file)
        if not check_scene( env, state, cur_ef_pose[:3, :3], object_performance, env_planner, scene_file,
                            run_iter, True, CONFIG, True):  continue

        # expert
        exp_reward, episode_reward, exp_lifted, exp_collided, episode_steps  = 0, 0, 0, 0, 0
        total_cnt = total_cnt + 1
        cnt = cnt + 1
        expert_traj = None
        recons_traj = None
        extra_text = None
        agent.timestep  = torch.zeros(1).cuda()
        exp_failure, exp_success = 0., 1.

        if HAS_PLANNER_INSTALLED:
            expert_plan, exp_failure, exp_success = env_planner.expert_plan(return_success=True, checked=not args.load_test_scene)

        if HAS_PLANNER_INSTALLED and not args.use_sample_latent:
            if args.expert  or CONFIG.full_traj_embedding or CONFIG.train_traj_sampler:
                expert_traj_data,   overhead_traj, traj, expert_traj, expert_overhead_traj, \
                                    (exp_lifted, exp_collided, exp_reward) = expert_rollout(state, expert_plan)
                exp_exec_success = exp_exec_success + exp_lifted
                state =  env.reset(scene_file=scene_file, data_root_dir=data_root, reset_free=True)

            if CONFIG.traj_goal_mutual_conditioned:
                expert_exec_traj, _, expert_pose_traj = update_expert_traj(agent, expert_traj_data, cfg)

        while not args.expert and not env.episode_done and not env.obstacle_collided:
            remain_timestep = max(max_steps-episode_steps, 1)
            goal_state = env._get_relative_goal_pose()
            last_joint = np.array(env._panda.getJointStates()[0])

            # sample latent
            if args.use_sample_latent and args.sample_latent_gap > 0 and (episode_steps % args.sample_latent_gap == 0 and episode_steps < MAX_STEP - 8):
                input_exec_traj = None # None means generating traj latent from sampler
            else:
                input_exec_traj = expert_exec_traj # not-None means executing encoded expert latent plan

            # compute traj and action
            action, traj_pred, extra_pred, aux_pred = agent.select_action(state,
                                                         vis=args.vis_traj_simulate,
                                                         remain_timestep=remain_timestep,
                                                         curr_joint=last_joint,
                                                         gt_traj=input_exec_traj)

            # gaddpg
            if  gaddpg_involved:
                gaddpg_switch_prob = float(agent.gaddpg_pred)
                gaddpg_switch = (gaddpg_switch_prob > 0.7 and remain_timestep < CONFIG.gaddpg_max_step) or (remain_timestep < CONFIG.gaddpg_min_step)
                if gaddpg_switch: #
                    action, aux_pred = gaddpg_action(gaddpg, state, action, episode_steps, max_steps, last_joint, return_goal=True)

            elif args.use_gaddpg and remain_timestep <= CONFIG.gaddpg_mean_remaining_step:
                action, aux_pred = gaddpg_action(gaddpg, state, action, episode_steps, max_steps, last_joint, return_goal=True)

            # vis
            if agent.multi_traj_sample:
                recons_traj = traj_pred

            overhead_vis_img, vis_img = annotate_images( action, camera_param, recons_traj, last_joint, extra_pred, state,
                                                         aux_pred, goal_state, get_info(state, 'img', cfg.RL_IMG_SIZE), hand_cam_intr,
                                                         camera_hand_offset_pose, goal_involved, -1, None, extra_text )

            # step
            next_state, reward, done, env_info = env.step(action, vis=False)
            overhead_traj.append(overhead_vis_img)
            traj.append(vis_img)
            collided = env_info['collided']
            print('run: {} remain step: {} action: {:.3f} rew: {:.2f}'.format(run_iter, remain_timestep, np.abs(action[:3]).sum(), reward))

            if ((episode_steps == TOTAL_MAX_STEP) or done) and not collided:
                reward, res_obs, overhead_res_obs = env.retract(True, camera_param)
                res_obs = [get_info(r, 'img', cfg.RL_IMG_SIZE) for r in res_obs]
                done = False
                traj.extend(res_obs)
                overhead_traj.extend(overhead_res_obs)

            episode_reward += reward
            episode_steps += 1
            state = next_state

        ######################## log
        log_output( exp_success, exp_failure, episode_reward, collided, reward, exp_lifted, exp_collided, exp_reward,
                    episode_steps, cnt, start_time, scene_file)
        if args.record and len(traj) > 5:
            write_video(traj, scene_indexes[-1], overhead_traj, expert_traj, expert_overhead_traj, cnt % MAX_VIDEO_NUM, cfg.RL_IMG_SIZE, cfg.OUTPUT_MISC_DIR,
                            logdir, env.target_name, video_surfix, False, False, False)


def get_video_path():
    video_prefix = 'YCB'
    logdir = '{}/{}/{}_{}'.format(cfg.OUTPUT_DIR, output_time, CONFIG.env_name, POLICY)
    print('output_time: {} logdir: {}'.format(output_time, logdir))

    # Main
    args.output_file = args.output_file.replace('txt', 'script_{}.txt'.format(cfg.script_name))
    video_prefix = video_prefix + '_' + cfg.script_name[:-5]
    print('video output: {} stat output: {}'.format(video_prefix, args.output_file))
    video_surfix = '{}_{}'.format(video_prefix, POLICY)
    video_output_folder = 'output_misc/rl_output_video_{}_{}'.format(video_prefix, POLICY)
    surfix = ''

    video_output_folder = video_output_folder + surfix
    video_surfix = video_surfix + surfix
    mkdir_if_missing(video_output_folder)
    clean_dir(video_output_folder)
    return video_output_folder, video_surfix, logdir

if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    parser = create_parser()
    args, other_args = parser.parse_known_args()
    GPUs = GPUtil.getGPUs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    net_dict, output_time = setup()
    CONFIG = cfg.RL_TRAIN

    # Args
    RENDER = args.render
    TRAIN = not args.test
    TEST_NUM = args.test_episode_num
    MAX_STEP = cfg.RL_MAX_STEP
    TOTAL_MAX_STEP = MAX_STEP + 20
    MAX_TEST_PER_OBJ = args.max_cnt_per_obj
    NUM_RUNS = args.num_runs
    MAX_VIDEO_NUM = 50
    CONFIG.output_time = output_time
    POLICY = 'DQN_HRL' if CONFIG.RL else 'BC'
    CLUSTER = check_ngc()
    MULTI_CAMERA = False
    CONFIG.index_file = 'ycb_large.json'
    CONFIG.test_log_sigma_clip = 0.7

    # Metrics
    input_dim = CONFIG.feature_input_dim
    avg_reward, avg_collided, avg_lifted, exp_lifted = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    cnt, total_cnt, exp_exec_success = 0., 0., 0.
    object_performance = {}
    traj_lengths, scene_indexes, target_obj_list = [], [], []
    model_output_dir = os.path.join(cfg.OUTPUT_DIR, output_time)
    pretrained_path = model_output_dir
    CONFIG.model_output_dir = model_output_dir
    video_output_folder, video_surfix, logdir= get_video_path()
    pretrained_model_path = cfg.pretrained_time if hasattr(cfg, 'pretrained_time') else output_time

    # Agent
    action_space = PandaTaskSpace6D()
    agent = globals()[POLICY](input_dim, action_space, CONFIG)
    agent.setup_feature_extractor(net_dict, True)
    agent.load_model(pretrained_path, surfix=args.model_surfix, set_init_step=False)

    # Environment
    env_config = cfg.env_config
    env_config['renders'] = RENDER
    scene_prefix =  '{}_scene'.format(CONFIG.index_file)
    env = eval(CONFIG.env_name)(**env_config)
    env._load_index_objs(*sample_experiment_objects())
    state = env.reset(save=False, data_root_dir=cfg.DATA_ROOT_DIR, cam_random=0)
    env_planner = EnvPlanner(env)
    camera_hand_offset_pose = se3_inverse(env.cam_offset)
    camera_param = ([-0.51,-0.82,-1.31], 1.5, -33.1, 329.4, 0., 52.)

    max_steps = cfg.RL_MAX_STEP
    goal_involved   = CONFIG.policy_aux
    gaddpg_involved = (CONFIG.critic_gaddpg and CONFIG.dqn) or args.use_gaddpg
    if gaddpg_involved:
        gaddpg = get_gaddpg(pretrained_path)
        agent.gaddpg = gaddpg
        agent.gaddpg.set_mode(True)

    print('start test time:', datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S"))
    for run_iter in range(NUM_RUNS):
        test(run_iter=run_iter)
        avg_lifted.set_mean()
        avg_reward.set_mean()
        avg_collided.set_mean()

    write_result()