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

import scipy.io as sio
import IPython
import time
from torch import nn
from collections import deque
import tabulate
import torch.nn.functional as F
import cv2
import yaml
import torch
import ray

import core
import copy
import math
from easydict import EasyDict as edict
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

import psutil
import GPUtil
import itertools
import glob

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.sum_2 = 0.
        self.count_2 = 0.
        self.means = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val * n
        self.count_2 += n

    def set_mean(self):
        self.means.append(self.sum_2 / self.count_2)
        self.sum_2 = 0.
        self.count_2 = 0.

    def std(self):
        return np.std(np.array(self.means) + 1e-4)

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)

def module_max_param(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data))
                for name, param in module.named_parameters()])
    return max_data


def module_max_gradient(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_grad = np.amax(
        [(maybe_max(param.grad)) for name, param in module.named_parameters()]
    )
    return max_grad

def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    R = np.stack([side, up, -forward], axis=-1)
    return R

def process_image_input(state):
    state[:, :3] *= 255
    if state.shape[1] >= 4:
        state[:, 3] *= 5000
    if state.shape[1] == 5:
        state[:, -1][state[:, -1] == -1] = 50
    return state.astype(np.uint16)

def process_image_output(sample):
    sample = sample.astype(np.float32).copy()
    n = len(sample)
    if len(sample.shape) <= 2:
        return sample

    sample[:, :3] /= 255.0
    if sample.shape[0] >= 4:
        sample[:, 3] /= 5000
    sample[:, -1] = sample[:, -1] != 0
    return sample

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def get_valid_index(arr, index):
    return arr[min(len(arr) - 1, index)]


def fc(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.BatchNorm1d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes), nn.LeakyReLU(0.1, inplace=True)
        )

def deg2rad(deg):
    if type(deg) is list:
        return [x/180.0*np.pi for x in deg]
    return deg/180.0*np.pi

def rad2deg(rad):
    if type(rad) is list:
        return [x/np.pi*180 for x in rad]
    return rad/np.pi*180

def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def projection_to_intrinsics(mat, width=224, height=224):
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def view_to_extrinsics(mat):
    pose = np.linalg.inv(np.array(mat).reshape([4, 4]).T)
    return np.linalg.inv(pose.dot(rotX(np.pi)))


def concat_state_action_channelwise(state, action):
    """
    concate the action in the channel space
    """
    action = action.unsqueeze(2)
    state = torch.cat((state, action.expand(-1, -1, state.shape[2])), 1)
    return state

def safemat2quat(mat):
    quat = np.array([1,0,0,0])
    try:
        quat = mat2quat(mat)
    except:
        pass
    quat[np.isnan(quat)] = 0
    return quat


def migrate_model(in_model, out_model, surfix="latest", in_policy_name="BC", out_policy_name="BC"):
    files = [
        "actor_PandaYCBEnv_{}".format(surfix),
        "state_feat_PandaYCBEnv_{}".format(surfix),
        "traj_feat_PandaYCBEnv_{}".format(surfix),
        "traj_sampler_PandaYCBEnv_{}".format(surfix),
        "critic_PandaYCBEnv_{}".format(surfix),
    ]
    config_file = glob.glob(in_model + '/*.yaml')
    config_file = [f for f in config_file if 'bc' in f]
    in_policy_name = "BC" if len(config_file) >= 1 else "DQN_HRL"
    for file in files:
        cmd = "cp {}/{}_{} {}/{}_{}".format(
                in_model, in_policy_name, file, out_model, out_policy_name, file)
        if os.path.exists('{}/{}_{}'.format(in_model, in_policy_name, file)):
            os.system(cmd)
            print(cmd)

def get_info(state, opt="img", IMG_SIZE=(112, 112)):
    if opt == "img":
        return (state[0][1][:3].T * 255).astype(np.uint8)
    if opt == "intr":
        cam_proj = np.array(state[-2][48:]).reshape([4, 4])
        return projection_to_intrinsics(cam_proj, IMG_SIZE[0], IMG_SIZE[1])[:3, :3]
    if opt == "point":
        return state[0][0]



def make_gripper_pts(points, color=(1, 0, 0)):
    line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

    cur_gripper_pts = points.copy()
    cur_gripper_pts[1] = (cur_gripper_pts[2] + cur_gripper_pts[3]) / 2.0
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(cur_gripper_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_index)
    line_set.colors = o3d.utility.Vector3dVector(
        [color for i in range(len(line_index))]
    )
    return line_set

def _cross_matrix(x):
    """
    cross product matrix
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def a2e(q):
    p = np.array([0, 0, 1])
    r = _cross_matrix(np.cross(p, q))
    Rae = np.eye(3) + r + r.dot(r) / (1 + np.dot(p, q))
    return mat2euler(Rae)

def get_camera_constant():
    K = np.eye(3)
    K[0,0]=K[0,2]=K[1,1]=K[1,2] = 56.

    offset_pose = np.zeros([4, 4])
    offset_pose[0,1]=-1.
    offset_pose[1,0]=offset_pose[2,2]=offset_pose[3,3]=1.
    offset_pose[2,3]=offset_pose[1,3]=-0.036
    return offset_pose, K

def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def se3_inverse_batch(RT):
    R = RT[:, :3, :3]
    T = RT[:, :3, 3].reshape((-1, 3, 1))
    RT_new = np.tile(np.eye(4, dtype=np.float32), (len(R), 1, 1))
    RT_new[:, :3, :3] = R.transpose(0, 2, 1)
    RT_new[:, :3, 3] = -1 * np.matmul(R.transpose(0, 2, 1), T).reshape(-1, 3)
    return RT_new

def save_scene(env, file_name, save_info=None):
    if save_info is None:
        obj_names, obj_poses = env.get_env_info()
        init_joints =  np.array(env._panda.getJointStates()[0])
    else:
        obj_names, obj_poses, target_idx, init_joints, goals, reach_grasps = save_info
    scene_mat = {}
    scene_mat['path'] = ['data/objects/' + name.split('/')[-1].strip() for name in obj_names]
    scene_mat['pose'] = [pose for pose in obj_poses]
    scene_mat['init_joints'] = init_joints
    scene_mat['target_idx'] = target_idx
    scene_mat["goals"] = goals
    scene_mat["reach_grasps"] = reach_grasps
    sio.savemat(file_name + '.mat', scene_mat)

def save_traj(env, file_name):
    scene_mat = sio.loadmat(file_name + '.mat')
    scene_mat['traj'] = env.cur_plan
    sio.savemat(file_name + '.mat', scene_mat)

def backproject_camera_target(im_depth, K, im_mask, im_normal):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    # create 4 channel point clouds

    im_mask   = im_mask.flatten()
    im_normal = im_normal.reshape(-1, 3)
    mask = (depth != -1) * (im_mask != -1)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R )
    X[1] *= -1  # flip y OPENGL
    X = X[:, mask]
    X = np.concatenate((X, im_mask[None, mask], im_normal.T[:, mask]), axis=0)
    return X

def compute_normal(depth, smooth=False):
    if smooth:
        zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    else:
        zy, zx = np.gradient(depth)

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
    return normal


def get_hand_anchor_index_point():
    hand_anchor_points = np.array(
        [
            [0, 0, 0],
            [0.00, -0.00, 0.058],
            [0.00, -0.043, 0.058],
            [0.00, 0.043, 0.058],
            [0.00, -0.043, 0.098],
            [0.00, 0.043, 0.098],
        ]
    )
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]
    return hand_anchor_points, line_index

def grasp_gripper_lines(pose):
    hand_anchor_points, line_index = get_hand_anchor_index_point()
    hand_points = (
        np.matmul(pose[:, :3, :3], hand_anchor_points.T) + pose[:, :3, [3]]
    )
    hand_points = hand_points.transpose([1, 0, 2])
    p1 = hand_points[:, :, line_index[0]].reshape([3, -1])
    p2 = hand_points[:, :, line_index[1]].reshape([3, -1])
    return [p1], [p2]

def tensor_stat(x):
    return ([round(i, 2) for i in x.min(-1)[0].tolist()], [round(i, 2) for i in x.mean(-1).tolist()],
           [round(i, 2) for i in x.max(-1)[0].tolist()])



def valid_3d_to_2d(K, xyz_points, img, filter_depth=True):
    p_xyz = K.dot(xyz_points)
    x, y = (p_xyz[0] / p_xyz[2]).astype(np.int), (p_xyz[1] / p_xyz[2]).astype(np.int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1) * (p_xyz[2] > 0.03)
    return x, y, valid_idx_mask



def grasp_points_from_pose(pose, offset_pose):
    hand_anchor_points = np.array(
        [
            [0, 0, 0],
            [0.00, -0.00, 0.068],
            [0.00, -0.043, 0.068],
            [0.00, 0.043, 0.068],
            [0.00, -0.043, 0.108],
            [0.00, 0.043, 0.108],
        ]
    )
    line_index = [[0, 1, 1, 2, 3], [1, 2, 3, 4, 5]]

    hand_anchor_points = pose[:3, :3].dot(hand_anchor_points.T) + pose[:3, [3]]
    hand_anchor_points = (
        offset_pose[:3, :3].dot(hand_anchor_points) + offset_pose[:3, [3]]
    )
    return  hand_anchor_points



def huber_loss(x, y, scale=5e-1, outlier=3.0):
    """
    smooth l1 loss with scale and outlier
    """
    diff = torch.abs(x - y)
    diff = diff[diff < outlier]  # remove outlier
    flag = diff < scale

    diff[flag] = 0.5 * diff[flag] ** 2
    diff[~flag] = scale * (diff[~flag] - 0.5 * scale)
    return diff.mean()



def get_noise_delta(action, noise_level, noise_type="uniform"):
    normal = noise_type != "uniform"

    if type(action) is not np.ndarray:
        if normal:
            noise_delta = torch.randn_like(action) * noise_level / 3.0
        else:
            noise_delta = (torch.rand_like(action) * 3 - 6) * noise_level
        noise_delta[:, 3:] *= 5

    else:
        if normal:
            noise_delta = np.random.normal(size=(6,)) * noise_level / 3.0
        else:
            noise_delta = np.random.uniform(-3, 3, size=(6,)) * noise_level
        noise_delta[3:] *= 5  # radians
    return noise_delta


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def rotZ(rotz):
    RotZ = np.array(
        [
            [np.cos(rotz), -np.sin(rotz), 0, 0],
            [np.sin(rotz), np.cos(rotz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return RotZ


def rotY(roty):
    RotY = np.array(
        [
            [np.cos(roty), 0, np.sin(roty), 0],
            [0, 1, 0, 0],
            [-np.sin(roty), 0, np.cos(roty), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotY


def rotX(rotx):
    RotX = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rotx), -np.sin(rotx), 0],
            [0, np.sin(rotx), np.cos(rotx), 0],
            [0, 0, 0, 1],
        ]
    )
    return RotX


def se3_transform_pc(pose, point):
    if point.shape[1] == 3:
        return np.matmul(pose[..., :3, :3], point) + pose[..., :3, [3]]
    else:
        point_ = point.copy()
        point_[...,:3,:] = np.matmul(pose[..., :3, :3], point[...,:3,:]) + pose[..., :3, [3]]
        return point_

def mkdir_if_missing(dst_dir):
    try:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
    except:
        pass

def unpack_pose(pose, rot_first=False):
    unpacked = np.eye(4)
    if rot_first:
        unpacked[:3, :3] = quat2mat(pose[:4])
        unpacked[:3, 3] = pose[4:]
    else:
        unpacked[:3, :3] = quat2mat(pose[3:])
        unpacked[:3, 3] = pose[:3]
    return unpacked

def pack_pose(pose, rot_first=False):
    packed = np.zeros(7)
    if rot_first:
        packed[4:] = pose[:3, 3]
        packed[:4] = safemat2quat(pose[:3, :3])
    else:
        packed[:3] = pose[:3, 3]
        packed[3:] = safemat2quat(pose[:3, :3])
    return packed


def unpack_pose_rot_first(pose):
    unpacked = np.eye(4)
    unpacked[:3, :3] = quat2mat(pose[:4])
    unpacked[:3, 3] = pose[4:]
    return unpacked

def pack_pose_rot_first(pose):
    packed = np.zeros(7)
    packed[4:] = pose[:3, 3]
    packed[:4] = safemat2quat(pose[:3, :3])
    return packed

def pack_pose_rot_first_batch(pose):
    n = len(pose)
    packed = np.zeros([n, 7])
    packed[:, 4:] = pose[:, :3, 3]
    rotmat_th = torch.from_numpy(pose[:, :3, :3])
    packed[:, :4] = mat_to_quat(rotmat_th).detach().numpy()
    return packed

def pack_pose_batch(pose):
    n = len(pose)
    packed = np.zeros([n, 7])
    packed[:, :3] = pose[:, :3, 3]
    rotmat_th = torch.from_numpy(pose[:, :3, :3])
    packed[:, 3:] = mat_to_quat(rotmat_th).detach().numpy()
    return packed

def pack_pose_rot_first_batch_th(pose):
    n = len(pose)
    packed = torch.zeros([n, 7]).cuda()
    packed[:, 4:] = pose[:, :3, 3]
    rotmat_th =  pose[:, :3, :3]
    packed[:, :4] = mat_to_quat(rotmat_th)
    return packed

def pack_pose_euler(pose):
    packed = np.zeros(6)
    packed[3:] = pose[:3, 3]
    packed[:3] = mat2euler(pose[:3, :3])
    return packed

def unpack_pose_euler(pose):
    unpacked = np.eye(4)
    unpacked[:3, 3] = pose[:3]
    unpacked[:3, :3] = euler2mat(pose[3], pose[4], pose[5])
    return unpacked

def inv_pose(pose):
    return pack_pose(se3_inverse(unpack_pose(pose)))

def relative_pose(pose1, pose2):
    return pack_pose(se3_inverse(unpack_pose(pose1)).dot(unpack_pose(pose2)))

def relative_pose_rot_firstz(pose1, pose2):
    return pack_pose_rot_first(se3_inverse(unpack_pose_rot_first(pose1).dot(
                se3_inverse(unpack_pose_rot_first(pose2)))))

def robot_point_from_joint(joints):
    arm_collision_point = get_collision_points()
    robot = require_robot()
    plan_link_poses  = robot.forward_kinematics_parallel(wrap_value(joints)[None], offset=False)[0]
    inv_ef_pose = se3_inverse(plan_link_poses[7])
    plan_link_poses = np.matmul(inv_ef_pose[None], plan_link_poses)
    collision_point = np.matmul(plan_link_poses[...,:3,:3], arm_collision_point.swapaxes(-1, -2)[:,:3]).swapaxes(-1, -2) + \
                                          plan_link_poses[...,:3,[3]].swapaxes(-1, -2)
    return collision_point.reshape([-1, 3]).T

def relative_action_pose(pose1, pose2):
    # delta where pose1 * delta = pose2
    return pack_pose_euler(se3_inverse(unpack_pose_rot_first(pose1)).dot(unpack_pose_rot_first(pose2)))

def unpack_action(action):
    pose_delta = np.eye(4)
    pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
    pose_delta[:3, 3] = action[:3]
    return pose_delta

def unpack_action_batch(actions):
    pose_deltas = []
    for action in actions:
        pose_delta = np.eye(4)
        pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
        pose_delta[:3, 3] = action[:3]
        pose_deltas.append(pose_delta)
    return np.stack(pose_deltas, axis=0)


def unpack_action_pose(pose):
    # delta where pose1 * delta = pose2
    return pack_pose_euler((unpack_pose_rot_first(pose)))


def compose_pose(pose1, pose2):
    return pack_pose(unpack_pose(pose1).dot(unpack_pose(pose2)))

def safe_div(dividend, divisor, eps=1e-8):  # mark
    return dividend / (divisor + eps)

def wrap_value(value):
    if value.shape[0] <= 7:
        return rad2deg(value)
    value_new = np.zeros(value.shape[0] + 1)
    value_new[:7] = rad2deg(value[:7])
    value_new[8:] = rad2deg(value[7:])
    return value_new

def wrap_values(value):

    value_new = np.zeros([value.shape[0], value.shape[1] + 1])
    value_new[:,:7] = rad2deg(value[:,:7])
    value_new[:,8:] = rad2deg(value[:,7:])
    return value_new


def inv_relative_pose(pose1, pose2, decompose=False):
    """
    pose1: b2a
    pose2: c2a
    relative_pose:  b2c
    shape: (7,)
    """

    from_pose = np.eye(4)
    from_pose[:3, :3] = quat2mat(pose1[3:])
    from_pose[:3, 3] = pose1[:3]
    to_pose = np.eye(4)
    to_pose[:3, :3] = quat2mat(pose2[3:])
    to_pose[:3, 3] = pose2[:3]
    relative_pose = se3_inverse(to_pose).dot(from_pose)
    return relative_pose


def ros_quat(tf_quat):  # wxyz -> xyzw
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


def tf_quat(ros_quat):  # xyzw -> wxyz
    quat = np.zeros(4)
    quat[0] = ros_quat[-1]
    quat[1:] = ros_quat[:-1]
    return quat

def tf_quats(ros_quat):  # xyzw -> wxyz
    quat = np.zeros_like(ros_quat)
    quat[:,0] = ros_quat[:,-1]
    quat[:,1:] = ros_quat[:,:-1]
    return quat

def soft_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def half_soft_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        if target_name[:7] in ["linear1", "linear2", "linear3"]:
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def half_hard_update(target, source, tau):
    for (target_name, target_param), (name, param) in zip(
        target.named_parameters(), source.named_parameters()
    ):
        if target_name[:7] in ["linear4", "linear5", "linear6"]:
            target_param.data.copy_(param.data)

def hard_update(target, source, tau=None):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



def distance_by_translation_point(p1, p2):
    """
    Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] > npoints:
        if use_farthest_point:
            pc = torch.from_numpy(pc).cuda()[None].float()
            new_xyz = (
            gather_operation(
                pc.transpose(1,2).contiguous(), furthest_point_sample(pc[...,:3].contiguous(), npoints)
            ) .contiguous() )
            pc = new_xyz[0].T.detach().cpu().numpy()

        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False )
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def check_filter_name(target_name):
    return np.sum([name in target_name for name in ['pitcher' ]]) == 0 and target_name != 'noexists'


def get_robot_mask(point):
    return point[3] == 2

def get_target_mask(point):
    return point[3] == 0

def get_obs_mask(point):
    return np.logical_or(point[3] == 1, point[3] == 3)

def get_near_obs_mask(point):
    return point[3] == 3

def pose_check(env, state=None, start_rot=None, CONFIG=None):

    dist = np.linalg.norm( env._get_target_relative_pose('tcp')[:3, 3])
    dist_flag = dist > CONFIG.init_distance_low and dist < CONFIG.init_distance_high

    z = start_rot[:3, 0] / np.linalg.norm(start_rot[:3,0])
    hand_dir_flag = z[-1] > -0.3
    if state is not None:
        target_mask = get_target_mask(env.curr_observ_point)
        obs_mask = get_obs_mask(env.curr_observ_point)
        robot_mask  = get_robot_mask(env.curr_observ_point)
        pt_flag = (target_mask.sum() > 5)  and (robot_mask.sum() < 8000) and \
                  np.sum(env.placed_objects) > 2 and state[0][0].shape[-1] > 2000
    else:
        pt_flag = True
    return hand_dir_flag and dist_flag and pt_flag

def test_cnt_check(env, object_performance=None, run_iter=0, MAX_TEST_PER_OBJ=10):
    if object_performance is not None:
        full_flag = env.target_name not in object_performance or \
                    object_performance[env.target_name][0].count < (run_iter + 1) * MAX_TEST_PER_OBJ
    else:
        full_flag = True
    return full_flag

def name_check(env, object_performance=None, run_iter=0, MAX_TEST_PER_OBJ=10):
    target_obj_flag = env.target_name != 'noexists'
    full_flag = test_cnt_check(env, object_performance, run_iter, MAX_TEST_PER_OBJ)
    name_flag  = check_filter_name(env.target_name)
    return name_flag and full_flag and target_obj_flag


def rand_sample_joint(env, init_joints=None, near=0.4, far=0.8):
    """
    randomize initial joint configuration
    """
    init_joints=None
    for _ in range(10):
        init_joints_ = env.randomize_arm_init(near, far)
        init_joints = init_joints_ if init_joints_ is not None else init_joints
        if init_joints is not None:
            env.reset_joint(init_joints)
            start_rot = env._get_ef_pose(mat=True)
            z = start_rot[:3, 0] / np.linalg.norm(start_rot[:3,0])
            hand_dir_flag = z[-1] > -0.3
            if hand_dir_flag:
                break

    return init_joints

def get_control_point_tensor(batch_size, use_torch=True, device="cpu", rotz=False):
    """
    Outputs a tensor of shape (batch_size x 6 x 3).
    use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.array([[ 0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ],
       [ 0.053, -0.   ,  0.075],
       [-0.053,  0.   ,  0.075],
       [ 0.053, -0.   ,  0.105],
       [-0.053,  0.   ,  0.105]], dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])
    if rotz:
        control_points = np.matmul(control_points, rotZ(np.pi / 2)[:3, :3])
    if use_torch:
        return torch.tensor(control_points).to(device).float()

    return control_points.astype(np.float32)

def object_points_from_rot_and_trans( grasp_eulers,
                                      grasp_translations,
                                      obj_pc=None,
                                      device="cpu"
                                      ):
    rot = tc_rotation_matrix(grasp_eulers[:, 0],
                             grasp_eulers[:, 1],
                             grasp_eulers[:, 2],
                             batched=True)

    # inverse transform
    obj_pc = obj_pc - grasp_translations.unsqueeze(-1).expand(-1, -1, obj_pc.shape[2])
    obj_pc = torch.matmul(rot.permute(0, 2, 1), obj_pc)
    return obj_pc

def transform_control_points(
    gt_grasps,
    batch_size,
    mode="qt",
    device="cpu",
    t_first=False,
    rotz=False,
    control_points=None ):
    """
    Transforms canonical points using gt_grasps.
    mode = 'qt' expects gt_grasps to have (batch_size x 7) where each
      element is catenation of quaternion and translation for each
      grasps.
    mode = 'rt': expects to have shape (batch_size x 4 x 4) where
      each element is 4x4 transformation matrix of each grasp.
    """
    assert mode == "qt" or mode == "rt", mode
    grasp_shape = gt_grasps.shape
    if grasp_shape[-1] == 7:
        assert len(grasp_shape) == 2, grasp_shape
        assert grasp_shape[-1] == 7, grasp_shape
        if control_points is None:
            control_points = get_control_point_tensor(
                batch_size, device=device, rotz=rotz
            )
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps, 1).repeat(1, num_control_points, 1)

        if t_first:
            gt_q = gt_grasps[:, :, 3:]
            gt_t = gt_grasps[:, :, :3]
        else:
            gt_q = gt_grasps[:, :, :4]
            gt_t = gt_grasps[:, :, 4:]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert len(grasp_shape) == 3, grasp_shape
        assert grasp_shape[1] == 4 and grasp_shape[2] == 4, grasp_shape
        if control_points is None:
            control_points = get_control_point_tensor(
                batch_size, device=device, rotz=rotz
            )
        shape = control_points.shape
        ones = torch.ones(
            (shape[0], shape[1], 1), dtype=torch.float32, device=control_points.device
        )
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))

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


def clean_dir(dst_dir): # dangerous
    if os.path.exists(dst_dir):
        os.system('rm -rf {}/*'.format(dst_dir))


def tc_rotation_matrix(az, el, th, batched=False):
    if batched:
        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1.0, 0.0, 0.0], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))

def control_points_from_rot_and_trans(
    grasp_eulers, grasp_translations, device="cpu", grasp_pc=None
):
    rot = tc_rotation_matrix(
        grasp_eulers[:, 0], grasp_eulers[:, 1], grasp_eulers[:, 2], batched=True
    )
    if grasp_pc is None:
        grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], device=device, rotz=True)

    grasp_pc = torch.matmul(grasp_pc.float(), rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1], -1)
    return grasp_pc

def combinationSum2(candidates, target, max_len, ratio, max_step, min_step=2):
    """
    :type candidates: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    res = [s for s in list(itertools.combinations(candidates, max_len)) if sum(s) == target and min(s) > min_step]
    res = [r * int(ratio) for r in res]
    return res

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def mat_to_quat(rotation_matrix, eps= 1e-8) :
    """Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (x, y, z, w) format.
    Args:
        rotation_matrix (torch.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        torch.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input size must be a (*, 3, 3) tensor. Got {}".format(
                rotation_matrix.shape))

    def safe_zero_division(numerator , denominator ) :
        eps = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)


    if not rotation_matrix.is_contiguous():
        rotation_matrix_vec = rotation_matrix.reshape(
            rotation_matrix.shape[0], 9)
    else: # *rotation_matrix.shape[:-2]
        rotation_matrix_vec = rotation_matrix.view(
            rotation_matrix.shape[0], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(
        rotation_matrix_vec, chunks=9, dim=-1)

    trace = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qw, qx, qy, qz], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion = torch.where(
        trace > 0., trace_positive_cond(), where_1)
    return quaternion


def get_color_mask(object_index, nc=None):
    """"""
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255

    return color_mask

def get_mask_colors(num):
    return (get_color_mask(np.arange(num) + 1)).tolist()

