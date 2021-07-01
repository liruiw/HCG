# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
import sys
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import random
from tensorboardX import SummaryWriter
from scipy.spatial import cKDTree

import scipy.io as sio
import IPython
import time
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
from torch.optim import Adam

from collections import deque
import tabulate
import cv2
import matplotlib.pyplot as plt
import yaml
import core
import copy
import math
from easydict import EasyDict as edict
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

import colorsys
import psutil
import GPUtil
from core.common_utils import *
import pybullet as p

HAS_PLANNER_INSTALLED = True
try:
    from OMG.ycb_render.robotPose import robot_pykdl
except:
    HAS_PLANNER_INSTALLED = False

# global variables
V = cam_V =  [[-0.9351, 0.3518, 0.0428, 0.3037],
              [0.2065, 0.639, -0.741, 0.132],
              [-0.2881, -0.684, -0.6702, 1.8803],
              [0.0, 0.0, 0.0, 1.0]]


hand_finger_point = np.array([ [ 0.,  0.,  0.   , -0.   ,  0.   , -0.   ],
                               [ 0.,  0.,  0.053, -0.053,  0.053, -0.053],
                               [ 0.,  0.,  0.075,  0.075,  0.105,  0.105]])
anchor_seeds = np.array([
                        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785],
                        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85],
                        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27],
                        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74],
                        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27],
                        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56],
                        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56],
                        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35],
                        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24],
                        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23]
                        ])
renderer = None
robot = None
robot_points = None

panda = None
panda_clients = []

def require_panda(num=1):
    global panda, panda_clients
    if panda is None:
        from env.panda_gripper_hand_camera import Panda
        import pybullet_utils.bullet_client as bc
        panda_clients = [bc.BulletClient(connection_mode=p.DIRECT) for i in range(num)]
        panda = [Panda(stepsize=1./ 1000., base_shift=[-0.05, 0.0, 10.], bullet_client=panda_clients[i]) for i in range(num)] # -0.65
    return panda, panda_clients

def require_robot(new=False):
    if new: return robot_pykdl.robot_kinematics(None, data_path='../../../')
    global robot
    if robot is None:
        robot = robot_pykdl.robot_kinematics(None, data_path='../../../')
    return robot

def require_renderer(large_fov=False, offset=False ):
    global renderer, robot
    if  renderer is None :
        from OMG.ycb_render.ycb_renderer import YCBRenderer

        width, height = 640, 480
        renderer = YCBRenderer(width=width, height=height, offset=offset, gpu_id=0)
        if not large_fov:
            renderer.set_projection_matrix(width, height, width * 0.8, width * 0.8, width / 2, height / 2, 0.1, 6)
            renderer.set_camera_default()
        else:
            renderer.set_fov(90)
        models = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "finger", "finger"]
        obj_paths = ["data/robots/{}.DAE".format(item) for item in models]
        renderer.load_objects(obj_paths)
        robot = require_robot()
    return renderer, robot

def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (8,)).normal_() # 4
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_gaussian(size, truncate_std=None, device=None):
    y = torch.randn(*size).float()
    y = y if device is None else y.to(device)
    if truncate_std is not None:
        truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
    return y

def vis_network_input(xyz, xyz_features):
    for i in range(len(xyzs)):
        renderer, _ = require_renderer()
        vis_point(renderer, xyz_features[i], interact=2)

def get_usage():
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    return gpu_usage, memory_usage


def solve_ik(joints, pose):
    """
    For simulating trajectory
    """
    ik = robot.inverse_kinematics_trac_ik(pose[:3], ros_quat(pose[3:]), seed=joints[:7])
    if ik is not None:
        joints = np.append(np.array(ik), [0, 0.04, 0.04])
    return joints

def bullet_ik(pandas, joints, poses, panda_clients): # simulate
    target_joints = []
    for panda, joint, p_client, pose in zip(pandas, joints, panda_clients, poses):
        panda.reset(np.array(joint).flatten())
        pos, orn = pose[:3], ros_quat(pose[3:])

        target_joints.append(np.array(p_client.calculateInverseKinematics(panda.pandaUid,
                                      panda.pandaEndEffectorIndex, pos, orn)))

    return np.stack(target_joints, axis=0)

def generate_simulated_learner_trajectory(point_state, joints, agent, remain_timestep, max_traj_num=1, vis=False, gaddpg=False):
    """
    use the current point cloud and bullet kinemetics to simulate observation and action for a trajectory
    extract the stored plan in the dataset for plan encoding
    param: 4 x N, 9
    """
    MAX_CLIENT_NUM = max(agent.test_traj_num, 8)
    pandas, panda_clients = require_panda(MAX_CLIENT_NUM)
    num = len(point_state)

    if  num > 1:
        init_joints = joints[0].flatten()
        total_step = int(remain_timestep[0])
    else:
        init_joints = joints.flatten()
        total_step  = remain_timestep

    pandas = pandas[:num]
    panda_clients = panda_clients[:num]
    ef_pose = []
    for panda, p_client in zip(pandas, panda_clients):
        panda.reset(joints=init_joints.flatten())
        pos, orn = p_client.getLinkState(panda.pandaUid, panda.pandaEndEffectorIndex)[:2]
        ef_pose.append(unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]]))
    ef_pose = np.stack(ef_pose, axis=0)
    sim_point_state = point_state[0]  if len(point_state.shape) == 3  and point_state.shape[0] == 1 else point_state
    sim_pose   = np.stack([np.eye(4)] * num, axis=0)

    plan, sim_states, sim_joints, sim_actions, sim_poses = [], [], [], [], [sim_pose]
    agent.train_traj_feature = False # avoid regenerating traj latent

    # rollout
    step_func = agent.batch_select_action if num  > 1 else agent.select_action

    if  has_check(agent, 'vis_traj') and hasattr(agent, 'gaddpg') and gaddpg: step_func = agent.gaddpg_step #
    for episode_steps in range(total_step):
        state = [[sim_point_state, np.zeros(1)], None, None, None]
        action, _, _, aux_pred = step_func(state, remain_timestep=remain_timestep - episode_steps, curr_joint=joints )
        if len(action.shape) == 1:
            action = action[None]

        action_pose = unpack_action_batch(action)
        ef_pose = np.matmul(ef_pose, action_pose) # joints
        joints = bullet_ik(pandas, joints, pack_pose_batch(ef_pose), panda_clients)
        sim_point_state = se3_transform_pc(se3_inverse_batch(action_pose), sim_point_state)
        plan.append(joints)
        sim_actions.append(action)
        sim_states.append(sim_point_state)
        sim_poses.append(np.matmul(sim_poses[-1], action_pose))
        sim_joints.append(joints)

    sim_poses = list(pack_pose_rot_first_batch(np.concatenate(sim_poses, axis=0)).reshape(-1, num, 7).transpose((1,0,2)))
    agent.train_traj_feature = True

    sim_integer_time = np.arange(total_step, 0, -1)
    traj_time_batch  = sim_integer_time[::-1] / float(total_step)
    return sim_poses, traj_time_batch, sim_joints, sim_states, sim_integer_time

def generate_simulated_expert_trajectory(state, plan, curr_joint, curr_traj_time=0, idx=0, vis=False,robot=None):
    """
    use the current point cloud to simulate observation and action for a trajectory
    extract the stored plan in the dataset for plan encoding
    param: 4 x N, T x 9, 9
    """
    arm_collision_point = get_collision_points()
    if robot is None: robot = require_robot()
    curr_ef_pose  = (robot.forward_kinematics_parallel(wrap_value(curr_joint)[None], offset=False)[0][7])
    global_pc =  np.matmul(curr_ef_pose[:3, :3], state[:3, :-500]) + curr_ef_pose[:3, [3]]
    pc_mask  = state[[3]]
    max_len  = len(plan)
    traj_len = plan_length(plan)
    plan     = plan[:traj_len]

    if traj_len == 0:
        return (state[None], plan, np.zeros(6)[None], pack_pose_rot_first(np.eye(4))[None],
                pack_pose_rot_first(np.eye(4))[None], [1.],
                [[idx, 0, curr_traj_time]])

    plan_link_poses  = robot.forward_kinematics_parallel(wrap_values(plan), offset=False)
    sim_poses   = pack_pose_rot_first_batch(np.matmul(se3_inverse(curr_ef_pose)[None], plan_link_poses[:, 7]))
    sim_goals   = pack_pose_rot_first_batch(np.matmul(se3_inverse_batch(\
                    np.concatenate((curr_ef_pose[None], plan_link_poses[:, 7]), axis=0)), plan_link_poses[-1, 7][None]))

    # not used
    inv_ef_pose = se3_inverse_batch(plan_link_poses[:, 7])
    sim_states = np.matmul(inv_ef_pose[:, :3, :3], global_pc[None]) + inv_ef_pose[:, :3, [3]]
    plan_link_poses = np.matmul(inv_ef_pose[:, None], plan_link_poses)
    collision_point = np.matmul(plan_link_poses[...,:3,:3], arm_collision_point.swapaxes(-1, -2)[:,:3]).swapaxes(-1, -2) + \
                                plan_link_poses[...,:3,[3]].swapaxes(-1, -2)
    collision_point = collision_point.reshape([len(plan_link_poses), -1, 3]).swapaxes(-1, -2)

    sim_states  = np.concatenate((sim_states, collision_point), axis=-1) # robot points
    sim_states  = np.concatenate((sim_states, np.tile(pc_mask[None], (len(sim_states), 1, 1))), axis=1)  # mask
    sim_joints  = np.concatenate((curr_joint[None], plan), axis=0)

    sim_actions = np.zeros([len(sim_joints) - 1, 6]) # not used
    sim_traj_idx = [[idx, j / float(traj_len), curr_traj_time + (j + 1) / max_len] for j in range(traj_len + 1)]
    sim_states = np.concatenate((state[None], sim_states),axis=0) #
    plan       = np.concatenate((curr_joint[None], plan),axis=0) #
    sim_poses  = np.concatenate((pack_pose_rot_first(np.eye(4))[None], sim_poses),axis=0)
    sim_actions  = np.concatenate((sim_actions, np.zeros(6)[None]),axis=0)
    sim_integer_time = np.arange(traj_len + 1, 0, -1)
    return (sim_states, plan, sim_actions, sim_poses, sim_goals, sim_integer_time, sim_traj_idx)

def vis_learner_traj(state, joints, agent, remain_timestep):
    """
    visualize rollout using the current traj_feat
    """
    remain_timestep = min(remain_timestep, 45)
    point_state = state[0][0][None]
    poses = robot.forward_kinematics_parallel(wrap_value(joints)[None], offset=False)[0]
    ef_pose = poses[7]
    packed_poses = [pack_pose(pose) for pose in poses]
    sampler_multi_traj = len(agent.traj_feat_target_test) > 1

    # make copy
    traj_feat_copy = agent.traj_feat.clone()
    agent.traj_feat = agent.traj_feat_target_test # restore sampler latent
    max_traj_num = 1

    if sampler_multi_traj:
        max_traj_num = agent.test_traj_num
        joints = np.tile(joints, (max_traj_num, 1))
        point_state = np.tile(point_state, (max_traj_num, 1, 1))
        remain_timestep = torch.ones(max_traj_num).cuda() * remain_timestep
        vis_traj = generate_simulated_learner_trajectory(point_state, joints, agent, remain_timestep, max_traj_num)[0]
        traj_lines = []
        hues = np.linspace(0., 5./6, max_traj_num )
        colors = np.stack([colorsys.hsv_to_rgb(hue, 1.0, 1.0) for hue in hues]) * 255
        lines_color =  (np.repeat(colors, 2, axis=0).astype(np.int)).tolist()
        for i in range(max_traj_num):
            traj_lines.extend(gripper_traj_lines(ef_pose, vis_traj[i]))
        lines = traj_lines
    else:
        vis_traj = generate_simulated_learner_trajectory(point_state, joints, agent, remain_timestep, max_traj_num)[0]
        traj_line, grasp_line = gripper_traj_lines(ef_pose, vis_traj[0])
        lines = [(traj_line[0], traj_line[1]), (grasp_line[0], grasp_line[1])]
        lines_color = [[0, 0, 255], [0, 0, 255]]

    vis_point_state = state[0][0]
    vis_point_state = vis_point_state[:, 6:] # avoid hand point collision
    target_mask = get_target_mask(vis_point_state)
    point_color = get_point_color(vis_point_state)
    vis_point_state = se3_transform_pc(ef_pose, vis_point_state) # base coordinate
    renderer = require_renderer()[0]
    renderer.vis(packed_poses, range(len(poses)),
        shifted_pose=np.eye(4),
        interact=2,
        V=np.array(V),
        visualize_context={
            "white_bg": True,
            "project_point": [vis_point_state[:3]],
            "project_color": [point_color],
            "point_size": [3],
            "reset_line_point": True,
            "static_buffer": True,
            "line": lines,
            "line_color": lines_color,
        }
    )
    agent.traj_feat = traj_feat_copy

def joint_to_cartesian(new_joints, curr_joint):
    """
    Convert joint space action to task space action by fk
    """
    r = require_robot()
    ef_pose = r.forward_kinematics_parallel(wrap_value(curr_joint)[None], offset=False)[0][-3]
    ef_pose_ = r.forward_kinematics_parallel(wrap_value(new_joints)[None], offset=False)[0][-3]
    rel_pose = se3_inverse(ef_pose).dot(ef_pose_)
    action = np.hstack([rel_pose[:3,3], mat2euler(rel_pose[:3,:3])])
    return action

def check_ngc():
    """
    check for using cluster in training
    """
    GPUs = GPUtil.getGPUs()
    gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    return (gpu_limit > 14000)


def plan_length(plan):
    if len(plan) == 0: return plan
    if type(plan) is np.ndarray:
        return np.sum(np.abs(plan).sum(-1) > 0)
    else:
        return torch.sum(torch.abs(plan).sum(-1) > 0)

def pad_traj_plan(plan, max_len=50):
    padded_plan = np.zeros((max_len, 9))
    if len(plan) == 0: return padded_plan
    padded_plan[:len(plan)] = plan
    return padded_plan


def update_net_args(config, spec, net_args):
    net_args["model_scale"] = config.feature_input_dim / 512.
    net_args["group_norm"] = True

    if has_check(config, 'sa_channel_concat'):
        spec["net_kwargs"]["action_concat"] = True

    if has_check(config, 'joint_point_state_input'):
        net_args["extra_latent"] += 7

    if  has_check(config, 'feature_option'):
        net_args["feature_option"] = config.feature_option

    if  has_check(config, 'value_overwrite_lr') and config.value_overwrite_lr > 0:
        spec["opt_kwargs"]["lr"] = config.value_overwrite_lr

def update_traj_net_args(config, spec, net_args):
    net_args["feature_extractor_class"] = config.traj_feature_extractor_class
    net_args["num_inputs"]     = config.traj_latent_size
    net_args["hidden_dim"]     = config.feature_input_dim
    net_args["feat_head_dim"]  = config.traj_latent_size
    net_args["config"]         = config
    net_args["model_scale"]    = config.traj_latent_size / 512.
    net_args["feature_option"] = config.st_feature_option
    net_args["group_norm"] = True
    spec["opt_kwargs"]["lr"] = config.traj_net_lr
    net_args["extra_latent"] += 7

def update_traj_sampler_net_args(config, spec, net_args):
    net_args["num_inputs"]     = config.traj_latent_size
    net_args["hidden_dim"]     = config.feature_input_dim
    net_args["feat_head_dim"]  = config.traj_latent_size
    net_args["config"]         = config
    net_args["output_model_scale"] = config.traj_latent_size / 512.
    net_args["model_scale"]    = config.traj_sampler_latent_size / 512.
    net_args["feature_option"] = config.traj_feature_option
    net_args["group_norm"] = True
    net_args["extra_latent"] += 7 # joint
    spec["opt_kwargs"]["lr"] = config.traj_sampler_net_lr

    if config.sampler_extra_abs_time:
        net_args["extra_latent"] += 1 # time


def make_nets_opts_schedulers(model_spec, config, cuda_device="cuda"):
    specs = yaml.load(open(model_spec).read(), Loader=yaml.SafeLoader)  #
    ret = {}
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    for net_name, spec in specs.items():
        net_args = spec.get("net_kwargs", {})

        if net_name == "state_feature_extractor":
            if has_check(config, 'state_feature_extractor'):
                spec["class"] = config.state_feature_extractor
            net_args["input_dim"] = config.channel_num
            update_net_args(config, spec, net_args)

        if net_name == 'traj_feature_extractor':
            if has_check(config, 'train_traj_feature'):
                net_args["input_dim"] = config.channel_num
                update_traj_net_args(config, spec, net_args)
            else:
                continue
        if net_name == 'traj_feature_sampler':
            if has_check(config, 'train_traj_sampler')  :
                net_args["input_dim"] = config.channel_num
                update_traj_sampler_net_args(config, spec, net_args)
            else:
                continue

        print('net_name:', net_name)
        net_class = getattr(core.networks, spec["class"])
        net = net_class(**net_args)

        net = torch.nn.DataParallel(net).to("cuda")
        d = {
            "net": net,
        }

        if "opt" in spec:
            d["opt"] = getattr(optim, spec["opt"])(
            net.parameters(), **spec["opt_kwargs"]
            )
            if len(config.overwrite_feat_milestone) > 0:
                spec["scheduler_kwargs"]["milestones"] = config.overwrite_feat_milestone
            print("schedule:", spec["scheduler_kwargs"]["milestones"])

            d["scheduler"] = getattr(optim.lr_scheduler, spec["scheduler"])(
                d["opt"], **spec["scheduler_kwargs"]
            )
            if hasattr(net.module, "encoder"):
                d["encoder_opt"] = getattr(optim, spec["opt"])(
                    net.module.encoder.parameters(), **spec["opt_kwargs"]
                )
                d["encoder_scheduler"] = getattr(optim.lr_scheduler, spec["scheduler"])(
                    d["encoder_opt"], **spec["scheduler_kwargs"]
                )
            if hasattr(net.module, "value_encoder"):
                d["val_encoder_opt"] = getattr(optim, spec["opt"])(
                    net.module.value_encoder.parameters(), **spec["opt_kwargs"]
                )
                d["val_encoder_scheduler"] = getattr(
                    optim.lr_scheduler, spec["scheduler"]
                )(d["val_encoder_opt"], **spec["scheduler_kwargs"])
        ret[net_name] = d
    return ret

def get_fc_feat_head(input_dim, dim_list, output_dim, acti_func='nn.ReLU', end_with_act=False):
    model_list = [nn.Linear(input_dim, dim_list[0]), nn.ReLU(True)]
    for i in range(1, len(dim_list)):
        model_list.extend([nn.Linear(dim_list[i-1], dim_list[i]), eval(acti_func)(True)])

    model_list.append(nn.Linear(dim_list[-1], output_dim))
    if end_with_act: model_list.append(eval(acti_func)(True))
    return nn.Sequential(*model_list)

def get_info(state, opt="img", IMG_SIZE=(112, 112)):
    if opt == "img":
        return (state[0][1][:3].T * 255).astype(np.uint8)
    if opt == "intr":
        cam_proj = np.array(state[-2][48:]).reshape([4, 4])
        return projection_to_intrinsics(cam_proj, IMG_SIZE[0], IMG_SIZE[1])[:3, :3]
    if opt == "point":
        return state[0][0]

def get_collision_points():
    """
    load collision points with the order of the link list and end effector
    """
    global robot_points
    if robot_points is None:
        collision_file = 'data/robots/all_collision_pts.npy'
        if not os.path.exists(collision_file):
            collision_pts = []
            links = [
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "hand",
                "finger",
                "finger",
            ]
            for i in range(len(links)):
                file = "data/robots/{}.xyz".format(links[i])
                pts = np.loadtxt(file)
                sample_pts = pts[random.sample(range(pts.shape[0]), 50)]
                collision_pts.append(sample_pts)
            collision_pts  = np.array(collision_pts)
            np.save(collision_file, collision_pts)
        else:
            collision_pts = np.load(collision_file)
        robot_points = collision_pts
    return robot_points

def sample_latent(batch_size, latent_size):
    return torch.randn(batch_size, latent_size).cuda()

def add_extra_text(img, extra_text, text_size=0.3, corner='tl'):
    img = img.copy()
    img_ratio = img.shape[0] / 256
    gap = int(15 * img_ratio)
    width, height = img.shape[:2]
    offset_h = 0 if corner.startswith('t') else height - int(50 * img_ratio)
    offset_w = 0 if corner.endswith('l') else int(width - 30 * img_ratio)
    sign = 1 if corner.startswith('t') else -1

    text_size = 0.3 * img_ratio
    for i, t in enumerate(extra_text): #
        cv2.putText(
            img, t,
            (offset_w, offset_h + sign * (gap + i * gap)),
            cv2.FONT_HERSHEY_DUPLEX,
            text_size, [255,0,0] )  # 0.7

    return img

def write_video(
    traj,
    scene_file,
    overhead_traj=None,
    expert_traj=None,
    overhead_expert_traj=None,
    name=0,
    IMG_SIZE=(112, 112),
    output_dir="output_misc/",
    logdir="policy",
    target_name="",
    surfix="",
    use_pred_grasp=False,
    success=False,
    use_value=False,
    extra_text=None
):
    ratio = 1 if expert_traj is None else 2
    result = "success" if success else "failure"
    video_writer = make_video_writer(
        os.path.join(
            output_dir,
            "rl_output_video_{}/{}_rollout.avi".format(surfix, scene_file),
        ),
        int(ratio * IMG_SIZE[0]),
        int(IMG_SIZE[1]),
    )

    text_color = [255, 0, 0] if use_pred_grasp else [0, 255, 0]
    for i in range(len(traj)):
        img = traj[i][..., [2, 1, 0]]
        if expert_traj is not None:
            idx = min(len(expert_traj) - 1, i)
            img = np.concatenate((img, expert_traj[idx][..., [2, 1, 0]]), axis=1)

        img = img.astype(np.uint8)
        if extra_text is not None:
            img = add_extra_text(img, extra_text)
        video_writer.write(img)

    if overhead_traj is not None:
        width, height = overhead_traj[0].shape[1], overhead_traj[0].shape[0]
        overhead_video_writer = make_video_writer(
            os.path.join( output_dir,
                "rl_output_video_{}/{}_overhead_rollout.avi".format(surfix, scene_file)), int(ratio * width), height )

        for i in range(len(overhead_traj)):
            img = overhead_traj[i][..., [2, 1, 0]]
            if overhead_expert_traj is not None:
                idx = min(len(overhead_expert_traj) - 1, i)
                img = np.concatenate((img, overhead_expert_traj[idx][..., [2, 1, 0]]), axis=1)

            overhead_video_writer.write(img.astype(np.uint8))


def append_pointcloud_time(agent, point_state, time_batch=None, traj=True, train=True):
    if not train:
        if not hasattr(agent, 'timestep'):
            traj_integer_time_batch = torch.Tensor([0]).float().cuda()
        else:
            traj_integer_time_batch = agent.timestep
    else:
        traj_integer_time_batch = time_batch

    if agent.sampler_extra_abs_time:
        traj_time_batch = traj_integer_time_batch.view(-1,1,1).expand(-1, -1, point_state.shape[2])
        point_state  = torch.cat((point_state, traj_time_batch), dim=1)

    return point_state


def preprocess_points(config, state_input, curr_joint, time_batch=None, traj=False, append_pc_time=False):
    """
    process point cloud for network input
    """
    if type(curr_joint) is not torch.Tensor:
        curr_joint = torch.from_numpy(curr_joint).cuda().float()
    if type(state_input) is not torch.Tensor:
        state_input = torch.from_numpy(state_input).cuda().float()
    state_input_batch = state_input.clone()
    curr_joint = curr_joint[:, :7]
    if state_input_batch.shape[-1] > 4500: # robot point included
        state_input_batch = remove_robot_pt(state_input_batch)

    if (not traj and has_check(config, 'joint_point_state_input')) or \
                    (traj and has_check(config, 'traj_joint_point_state_input')):
        curr_joint_bc = curr_joint[...,None].expand(-1, -1, state_input_batch.shape[-1])
        state_input_batch = torch.cat((state_input_batch, curr_joint_bc), dim=1)

    if  append_pc_time and hasattr(config, 'test_mode'):
        state_input_batch  = append_pointcloud_time(config, state_input_batch, time_batch, False, not config.test_mode)

    return state_input_batch


def get_point_color(vis_points):
    tgt_mask = get_target_mask(vis_points)
    obs_mask = get_obs_mask(vis_points)
    near_obs_mask = get_near_obs_mask(vis_points)
    rob_mask = get_robot_mask(vis_points)

    target_color = [0, 255, 0]
    obs_color = [255, 0, 0]
    rob_color = [0, 0, 255]
    near_obs_color = [255, 0, 0]

    pt_color = np.zeros_like(vis_points[:3]).T
    pt_color[tgt_mask] = target_color
    pt_color[obs_mask] = obs_color
    pt_color[rob_mask] = rob_color
    pt_color[near_obs_mask] = near_obs_color
    return pt_color


def sample_ef(target, near=0.2, far=0.50):
    # sample a camera extrinsics
    count = 0
    ik = None
    outer_loop_num = 20
    inner_loop_num = 5
    robot = require_robot()
    for _ in range(outer_loop_num):
        theta = np.random.uniform(low=0, high=1*np.pi/2)
        phi = np.random.uniform(low=np.pi/2, high=3*np.pi/2) # half sphere
        r = np.random.uniform(low=near, high=far) # sphere radius
        pos = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])

        trans = pos + target + np.random.uniform(-0.03, 0.03, 3)
        trans[2] = np.clip(trans[2], 0.2, 0.6)
        trans[1] = np.clip(trans[1], -0.3, 0.3)
        trans[0] = np.clip(trans[0], 0.0, 0.5)
        pos = trans - target

        for i in range(inner_loop_num):
            rand_up = np.array([0, 0, -1])
            rand_up = rand_up / np.linalg.norm(rand_up)
            R = inv_lookat(pos, 2 * pos, rand_up).dot(rotZ(-np.pi/2)[:3, :3])
            quat = ros_quat(mat2quat(R))
            ik = robot.inverse_kinematics(trans, quat, seed=anchor_seeds[np.random.randint(len(anchor_seeds))])
            if  ik is not None:
                break
    return ik



def gripper_traj_lines(start_pose, traj_state, joint_output=False, gripper_along=False ):
    ef_lines = []
    gripper_lines = [np.zeros([3, 0]), np.zeros([3, 0])]
    draw_gripper_traj_line = gripper_along

    if joint_output:
        r = require_robot()
        curr_joint = np.concatenate((traj_state, 0.04 * np.ones((len(traj_state), 2))), axis=-1)
        traj_state = r.forward_kinematics_parallel(wrap_values(curr_joint ), offset=False)[:, 7]

    for grasp_idx, grasp in enumerate(traj_state):
        if not joint_output:
            if grasp_idx == 0:
                grasp = np.eye(4)
            elif len(grasp) == 6:
                grasp = unpack_pose_euler(grasp)
            else:
                grasp = unpack_pose_rot_first(grasp)
        grasp_pose = start_pose.dot(grasp)
        line_starts, line_ends = grasp_gripper_lines(grasp_pose[None])
        gripper_lines[0] = np.concatenate((gripper_lines[0], line_starts[0]), axis=-1)
        gripper_lines[1] = np.concatenate((gripper_lines[1], line_ends[0]), axis=-1)
        ef_lines.append(grasp_pose[:3, 3])

    if not draw_gripper_traj_line:
        gripper_lines[0] = gripper_lines[0][:,-5:]
        gripper_lines[1] = gripper_lines[1][:,-5:]
    if len(ef_lines) > 1:
        ef_lines = [[ef_lines[idx], ef_lines[idx+1]] for idx in range(len(ef_lines) - 1)]
        ef_lines = np.array(ef_lines)
        ef_lines = [ef_lines.T[:, 0], ef_lines.T[:, 1]]
    else:
        ef_lines = []
    return [gripper_lines, ef_lines]

def vis_traj(point, curr_joint, traj_state=None, V=cam_V, interact=2, used_renderer=None, gripper_along=False):
    # visualize traj with renderer
    renderer, robot = require_renderer( )

    if type(point) is torch.Tensor:
        point = point.detach().cpu().numpy()

    point = point[0]
    if point.shape(1) != 4096:
        point = point[:,6:-500] #  remove gripper and robot point
    if type(curr_joint) is torch.Tensor:
        curr_joint = curr_joint.detach().cpu().numpy()
    if len(curr_joint) == 7:
        curr_joint = np.append(curr_joint, [0, 0])

    poses_ = robot.forward_kinematics_parallel(wrap_value(curr_joint), offset=False)[0]
    poses_2 = [pack_pose(pose) for pose in poses_]
    point_color = get_point_color(point)
    point = se3_transform_pc(poses_[7], point)

    if traj_state is not None:
        if type(traj_state) is torch.Tensor:
            traj_state = traj_state.detach().cpu().numpy()

        if type(traj_state) is list and len(traj_state) > 4:
            traj_lines = []
            line_colors = get_mask_colors(len(traj_state) * 2 + 5)[5:]

            for i in range(len(traj_state)):
                traj_lines.extend(gripper_traj_lines(poses_[7], traj_state[i]))

        else:
            gripper_lines, ef_lines = gripper_traj_lines(poses_[7], traj_state, gripper_along=gripper_along)
            line_colors = [[255, 255, 0], [0, 0, 255]]
            traj_lines = [gripper_lines, ef_lines]

        rgb = renderer.vis(poses_2, list(range(10)),
        shifted_pose=np.eye(4),
        interact=interact,
        V=np.array(V),
        visualize_context={
            "white_bg": True,
            "project_point": [point[:3]],
            "project_color": [point_color],
            "static_buffer": True,
            "reset_line_point": True,
            "line": traj_lines,
            "line_color": line_colors,
        }
        )

    else:
        rgb = renderer.vis(poses_2, list(range(10)),
        shifted_pose=np.eye(4),
        interact=interact,
        V=np.array(V),
        visualize_context={
            "white_bg": True,
            "project_point": [point[:3]],
            "project_color": [point_color],
            "static_buffer": True,
            "reset_line_point": True,
            "thickness": [2]
        } )

    return rgb

def vis_point(renderer, point_state, window_name='test', interact=1, curr_joint=None, grasp=None, V=None):
    """visualize single point state """
    if type(point_state) is torch.Tensor:
        point_state = point_state.detach().cpu().numpy()
    vis_points = point_state.copy()
    pt_color = get_point_color(vis_points)

    if V is None:
        V =  [[ 0.3021,  0.668,   0.6801,  0.    ],
             [-0.7739, -0.2447,  0.5841,  0.    ],
             [ 0.5566, -0.7028,  0.4431,  1.1434],
             [ 0.,      0.,      0.,      1.    ]]

    line, line_color = [], []
    cls_indexes, poses = [], []

    if grasp is not None:
        line_starts, line_ends = grasp_gripper_lines(unpack_pose_rot_first(grasp.detach().cpu().numpy())[None])
        line = [(line_starts[0], line_ends[0])]
        line_color = [[255, 255, 0]]

    return renderer.vis( poses, cls_indexes,
    shifted_pose=np.eye(4),
    interact=interact,
    V=np.array(V),
    visualize_context={
        "white_bg": True,
        "project_point": [vis_points[:3] ],
        "project_color": [pt_color],
        "point_size": [3],
        "reset_line_point": True,
        "static_buffer": True,
        "line": line,
        "line_color": line_color,
    }
    )


def compose_state_traj(data_list, CONFIG, step=0):
    """downsampling traj """
    downsample_length = (len(data_list[0]) - step) // int(CONFIG.sparsify_traj_ratio)
    idx = list(np.linspace(step, len(data_list[0]) - 1, downsample_length).astype(np.int))
    torch_data_list = []
    for data_idx, data in enumerate(data_list):
        torch_data_list.append(torch.from_numpy(np.stack([data[i] for i in idx], axis=0)).cuda().float())
    return torch_data_list

def update_expert_traj(agent, expert_data_list, cfg, step=0, remote=False):
    """ compute expert traj latent embedding """
    expert_exec_traj = compose_state_traj(expert_data_list, cfg.RL_TRAIN, step)
    if remote:
        recons_traj = agent.select_traj.remote(None,
                    expert_data_list[0][step][None],
                    None,
                    vis=False,
                    remain_timestep=cfg.RL_MAX_STEP,
                    curr_joint=expert_data_list[1][step][None],
                    gt_traj=expert_exec_traj)    # generate the traj latent
    else:
        recons_traj = agent.select_traj(None,
                    expert_data_list[0][step][None],
                    None,
                    vis=False,
                    remain_timestep=cfg.RL_MAX_STEP,
                    curr_joint=expert_data_list[1][step][None],
                    gt_traj=expert_exec_traj)    # generate the traj latent
    return expert_exec_traj, recons_traj, expert_exec_traj[3].detach().cpu().numpy()

def get_gaddpg(path=None, load_joint_trained_model=False):
    """ get pretrained GA-DDPG Models """
    from core.ddpg import DDPG
    gaddpg_dict = edict()
    gaddpg_dict = edict(yaml.load(open("output/demo_model/config.yaml", "r")))
    net_dict = make_nets_opts_schedulers(gaddpg_dict.RL_MODEL_SPEC, gaddpg_dict.RL_TRAIN)

    gaddpg = DDPG(512, PandaTaskSpace6D(), gaddpg_dict.RL_TRAIN)
    gaddpg.setup_feature_extractor(net_dict, True)
    gaddpg.load_model('output/demo_model')
    gaddpg.set_mode(True)
    return gaddpg


def proj_point_img(img, K, offset_pose, points=None, color=(255, 0, 0),
                   vis=False, neg_y=True, traj=None, joint_output=False, last_joint=None,
                   remain_timestep=-1,  gt_goal=None, traj_offset_pose=None, extra_text=None,
                   model_name=None, vis_traj_gradient_color=False):
    # draw traj lines, goal / actions predictions and texts in image plane
    target_mask = get_target_mask(points)
    obs_mask = get_obs_mask(points)
    robot_mask = get_robot_mask(points)
    colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]
    img = img.copy()

    # point first
    for i, mask in enumerate([target_mask, obs_mask, robot_mask]):
        points_i = points[:, mask]
        points_xyz = points_i[:3]
        xyz_points = offset_pose[:3, :3].dot(points_xyz) + offset_pose[:3, [3]]
        if neg_y: xyz_points[:2] *= -1
        x, y, valid_idx_mask = valid_3d_to_2d(K, xyz_points, img)
        img[y[valid_idx_mask], x[valid_idx_mask]] = colors[i]

    if traj_offset_pose is None: traj_offset_pose = offset_pose
    if traj is not None and traj[0] is not None:
        if (remain_timestep == -1 or len(traj) == remain_timestep) and type(traj) is not list:
            line_colors = [[255, 255, 0], [0, 255, 255]]
            traj_lines = gripper_traj_lines(traj_offset_pose, traj, joint_output)
        else:
            if type(traj) is list:
                traj_num = len(traj)
                remain_timestep = len(traj[0])
                traj[0] = traj[0][:,:7]
                traj = np.concatenate(traj, axis=0)
            else:
                traj_num = int(len(traj) / remain_timestep)
                remain_timestep = int(remain_timestep)

            traj_lines = []
            hues = np.linspace(0., 5./6, traj_num )
            colors = np.stack([colorsys.hsv_to_rgb(hue, 1.0, 1.0) for hue in hues]) * 255
            line_colors = np.repeat(colors,  2, axis=0)

            for i in range(traj_num):
                traj_lines.extend(gripper_traj_lines(traj_offset_pose, traj[i*remain_timestep:(i+1)*remain_timestep ] ))

        for line_i, lines in enumerate(traj_lines):
            lines = np.array(lines)
            if len(lines) == 0: continue
            if neg_y: lines[:, :2] *= -1
            p_xyz = np.matmul(K, lines)
            x, y = (p_xyz[:,0] / p_xyz[:,2]).astype(np.int), (p_xyz[:,1] / p_xyz[:,2]).astype(np.int)
            x = np.clip(x, 0, img.shape[0] - 1)
            y = np.clip(y, 0, img.shape[1] - 1)

            for i in range(x.shape[1]):
                # avoid clipping issues
                color = line_colors[line_i]
                color = (int(color[0]), int(color[1]), int(color[2]))
                if np.abs(x[0, i] - x[1, i]) > 100 or np.abs(y[0, i] - y[1, i]) > 100:
                    continue
                if line_i == 1 and len(traj_lines) > 4 and not vis_traj_gradient_color:
                    cv2.line(img, (x[0, i], y[0, i]), (x[1, i], y[1, i]), color, 2)
                else:
                    cv2.line(img, (x[0, i], y[0, i]), (x[1, i], y[1, i]), color, 1)

    if extra_text is not None:
        img = add_extra_text(img, extra_text, 1.5) # 0.7
    return img

def draw_grasp_img(img, pose, K, offset_pose, color=(0, 0, 255), vis=False, neg=True ):
    img_cpy = img.copy()
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]

    hand_anchor_points = grasp_points_from_pose(pose, offset_pose)
    if neg: hand_anchor_points[:2] *= -1
    p_xyz = K.dot(hand_anchor_points)
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    x = np.clip(x, 0, img.shape[0] - 1)
    y = np.clip(y, 0, img.shape[1] - 1)
    for i in range(len(line_index[0])):
        pt1 = (x[line_index[0][i]], y[line_index[0][i]])
        pt2 = (x[line_index[1][i]], y[line_index[1][i]])
        cv2.line(img_cpy, pt1, pt2, color, 2)

    return img_cpy

def remove_robot_pt(points):
    return points[..., :-500]

def reparameterize(mu, logsigma, truncated=True, fix_eps=None):
    std = torch.exp(logsigma)
    if truncated:
        eps = sample_gaussian(std.shape, truncate_std=2.).cuda()
    else:
        eps = torch.randn_like(std)
    if fix_eps is not None:
        eps = fix_eps
    return mu + eps * std

def has_check(x, prop):
    return hasattr(x, prop) and getattr(x, prop)

def check_scene(env, state=None, start_rot=None, object_performance=None, planner=None,
                scene_name=None, run_iter=0, check_ik=False, CONFIG=None, load_test_scene=False):
    """
    check if a scene is valid by its distance, view, hand direction, target object state, and object counts
    """

    if  load_test_scene :  return name_check(env, object_performance, run_iter)
    MAX_TEST_PER_OBJ = CONFIG.max_test_per_obj
    pose_flag  = pose_check(env, state, start_rot, CONFIG)
    name_flag  = name_check(env, object_performance, run_iter, MAX_TEST_PER_OBJ)
    collision_flag = not env.collided
    check_flag = pose_flag and name_flag and collision_flag
    if check_flag and check_ik:
        goal_validity = planner.expert_plan(return_success=True, check_scene=True)
        if not goal_validity:
            return False
    return check_flag

def sample_scene(env, planner, object_performance=None, scene_file=None, run_iter=0, CONFIG=None, timeout=6.):
    """
    sample scenes with ik filtering
    """
    state = None
    MAX_TEST_PER_OBJ = CONFIG.max_test_per_obj
    start_time = time.time()
    outer_cnt, inner_cnt = CONFIG.scene_sample_check_ik_cnt, CONFIG.scene_sample_inner_cnt
    if CONFIG.index_file == 'filter_shapenet.json':
        inner_cnt *= 3

    for _ in range(outer_cnt):
        for _ in range(inner_cnt):
            if time.time() - start_time > timeout:
                return  state, False
            flag = not test_cnt_check(env, object_performance, run_iter, MAX_TEST_PER_OBJ)

            if flag: break

            state = env.reset(  scene_file=None, init_joints=rand_sample_joint(env, None, CONFIG.ENV_NEAR, CONFIG.ENV_FAR),
                                reset_free=True, enforce_face_target=True )
            cur_ef_pose = env._get_ef_pose(mat=True)
            flag = check_scene(env, state, cur_ef_pose[:3, :3], object_performance,
                               planner, scene_file, run_iter, False, CONFIG)

            if flag: break
        if flag and check_scene(env, state, cur_ef_pose[:3, :3], object_performance,
                                planner, scene_file, run_iter, True, CONFIG):
            break

    return  state, flag


def select_target_point(state, target_pt_num=1024):
    """get target point cloud from scene point cloud """
    point_state = state[0][0]
    target_mask = get_target_mask(point_state)
    point_state = point_state[:4, target_mask]
    gripper_pc  = point_state[:4, :6]
    point_num  = min(point_state.shape[1], target_pt_num)
    obj_pc = regularize_pc_point_count(point_state.T, point_num, False).T
    point_state = np.concatenate((gripper_pc, obj_pc), axis=1)
    return [(point_state, state[0][1])] + state[1:]


def gaddpg_action(gaddpg, state, action, episode_steps, max_steps, curr_joint, return_goal=False):
    """apply GA-DDPG action """
    state = select_target_point(state)

    if state[0][0].shape[1] > 0:
        gaddpg_remain_step = max(min(max_steps-episode_steps + 1, 30), 1)
        print('use gaddpg remaining step: {}...'.format(gaddpg_remain_step))
        action, _, _, aux_pred = gaddpg.select_action(state, remain_timestep=gaddpg_remain_step, curr_joint=curr_joint)
        if return_goal:
            return action, aux_pred
        return action
    return np.zeros(6), np.ones(7) * 0.01


class PandaTaskSpace6D():
    def __init__(self):
        self.high = np.array([0.1,   0.1,  0.1,  np.pi/6,  np.pi/6,  np.pi/6])
        self.low  = np.array([-0.1, -0.1, -0.1, -np.pi/6, -np.pi/6, -np.pi/6])
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])

class RobotMLP(nn.Module):
    """ simple Pointnet-Like MLP """
    def __init__(self, in_channels, out_channels, dim=1, gn=False, gn_num=16):
        super(RobotMLP, self).__init__()
        if dim == 1:
            conv = nn.Conv1d
            if gn:
                bn = lambda k: nn.GroupNorm(gn_num, k)
            else:
                bn = nn.BatchNorm1d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(oc),
                nn.ReLU(True),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs, masks=None):
        inp_shape = inputs.shape
        inputs = inputs.view(-1, inputs.shape[-2], inputs.shape[-1])
        x = self.layers(inputs)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(len(x), -1)
        return x

def clean_dir(dst_dir):
    if os.path.exists(dst_dir):
        os.system('rm -rf {}/*'.format(dst_dir))


def get_usage_and_success():
    """Get gpu and memory usages as well as current performance """
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    return memory_usage, gpu_usage


def get_model_path(output_dir, name, env_name, surfix):
    actor_path = "{}/{}_actor_{}_{}".format(
        output_dir, name, env_name, surfix )
    critic_path = "{}/{}_critic_{}_{}".format(
        output_dir, name, env_name, surfix )
    traj_feat_path = "{}/{}_traj_feat_{}_{}".format(
        output_dir, name, env_name, surfix )
    traj_sampler_path = "{}/{}_traj_sampler_{}_{}".format(
        output_dir, name, env_name, surfix )
    state_feat_path = "{}/{}_state_feat_{}_{}".format(
        output_dir, name, env_name, surfix )
    return actor_path, critic_path, traj_feat_path, traj_sampler_path, state_feat_path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_loss_info_dict():
   return {     'bc_loss': deque([0], maxlen=50),
                'policy_grasp_aux_loss': deque([0], maxlen=50),
                'critic_gaddpg_loss': deque([0], maxlen=100),
                'critic_loss': deque([0], maxlen=100),
                'kl_loss': deque([0], maxlen=50),
                'sampler_grasp_aux_loss': deque([0], maxlen=50),
                'sampler_bc_loss': deque([0], maxlen=50),
                'traj_latent_loss': deque([0], maxlen=50),
                'gaddpg_loss':  deque([0], maxlen=50),
                'reward_mask_num': deque([0], maxlen=5),
                'expert_reward_mask_num': deque([0], maxlen=5),
                'value_mean': deque([0], maxlen=5),
                'return_mean': deque([0], maxlen=5),
                'gaddpg_pred_mean': deque([0], maxlen=5),
                'traj_grad': deque([0], maxlen=5),
                'traj_param': deque([0], maxlen=5),
                'policy_param': deque([0], maxlen=5),
                'sampler_mean': deque([0], maxlen=5),
                'traj_num': deque([0], maxlen=5),
                'sampler_logsigma': deque([0], maxlen=5),
                'policy_grad': deque([0], maxlen=5),
                'feat_grad': deque([0], maxlen=5),
                'feat_param': deque([0], maxlen=5),
                'val_feat_grad': deque([0], maxlen=5),
                'val_feat_param': deque([0], maxlen=5),
                'critic_grad': deque([0], maxlen=5),
                'critic_param': deque([0], maxlen=5),
                'train_batch_size': deque([0], maxlen=5),
             }


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def create_bottleneck(input_size, latent_size):
    logmu = nn.Linear(input_size, latent_size)
    logvar = nn.Linear(input_size, latent_size)
    return nn.ModuleList([logmu, logvar])


def get_policy_class(policy_net_name, args):
    policy = getattr(core.networks, policy_net_name)(
        args.num_inputs,
        args.action_dim,
        args.hidden_size,
        args.action_space,
        extra_pred_dim=args.extra_pred_dim,
        config=args,
    ).to('cuda')
    policy_optim = Adam(
       policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=1e-5 )
    policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        policy_optim, milestones=args.policy_milestones, gamma=args.lr_gamma)
    policy_target = getattr(core.networks, policy_net_name)(
        args.num_inputs,
        args.action_dim,
        args.hidden_size,
        args.action_space,
        extra_pred_dim=args.extra_pred_dim,
        config=args,
    ).to('cuda')
    return policy, policy_optim, policy_scheduler, policy_target


def get_critic(args):
    model = core.networks.ResidualQNetwork if has_check(args, 'dense_critic') else core.networks.QNetwork
    critic = model(
            args.critic_num_input,
            args.critic_value_dim,
            args.hidden_size,
            extra_pred_dim=args.critic_extra_pred_dim,
        ).cuda()

    critic_optim = Adam(
        critic.parameters(), lr=args.value_lr, eps=1e-5, weight_decay=1e-5 )
    critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        critic_optim,
        milestones=args.value_milestones,
        gamma=args.value_lr_gamma,
    )
    critic_target = model(
        args.critic_num_input,
        args.critic_value_dim,
        args.hidden_size,
        extra_pred_dim=args.critic_extra_pred_dim,
    ).cuda()
    return critic, critic_optim, critic_scheduler, critic_target