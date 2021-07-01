# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import random
import os
import time
import sys

import pybullet as p
import numpy as np
import IPython

from env.panda_gripper_hand_camera import Panda
from transforms3d.quaternions import *
import scipy.io as sio
from core.utils import *
import json
from itertools import product

BASE_LINK = -1
MAX_DISTANCE = 0.000


def get_num_joints(body, CLIENT=None):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_links(body, CLIENT=None):
    return list(range(get_num_joints(body, CLIENT)))

def get_all_links(body, CLIENT=None):
    return [BASE_LINK] + list(get_links(body, CLIENT))

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE, CLIENT=None):
    closest_points = p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)
    return len(closest_points) != 0


def any_link_pair_collision(body1, body2, links1=None, links2=None, CLIENT=None, **kwargs):
    if links1 is None:
        links1 = get_all_links(body1, CLIENT)
    if links2 is None:
        links2 = get_all_links(body2, CLIENT)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue

        if pairwise_link_collision(body1, link1, body2, link2, CLIENT=CLIENT, **kwargs):
            return True
    return False

def body_collision(body1, body2, max_distance=MAX_DISTANCE, CLIENT=None):
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, body2, links1, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

class PandaTaskSpace6D():
    def __init__(self):
        self.high = np.array([0.06,   0.06,  0.06,  np.pi/6,  np.pi/6,  np.pi/6])
        self.low  = np.array([-0.06, -0.06, -0.06, -np.pi/6, -np.pi/6, -np.pi/6])
        self.shape = [6]
        self.bounds = np.vstack([self.low, self.high])

class PandaYCBEnv():
    """
    Class for franka panda environment with YCB objects.
    """

    def __init__(self,
                 renders=False,
                 maxSteps=100,
                 random_target=False,
                 blockRandom=0.5,
                 cameraRandom=0,
                 action_space='configuration',
                 use_expert_plan=True,
                 accumulate_points=False,
                 use_hand_finger_point=False,
                 use_arm_point=False,
                 expert_step=20,
                 data_type='RGB',
                 filter_objects=[],
                 img_resize=(224, 224),
                 regularize_pc_point_count=False,
                 egl_render=False,
                 width=224,
                 height=224,
                 uniform_num_pts=1024,
                 numObjects=7,
                 termination_heuristics=True,
                 domain_randomization=False,
                 pt_accumulate_ratio=0.95,
                 env_remove_table_pt_input=False,
                 omg_config=None,
                 use_normal=False):

        self._timeStep = 1. / 1000.
        self._observation = []
        self._renders = renders
        self._maxSteps = maxSteps

        self._env_step = 0
        self._resize_img_size = img_resize

        self._p = p
        self._window_width = width
        self._window_height = height
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._numObjects =  numObjects

        self._accumulate_points = accumulate_points
        self._use_expert_plan = use_expert_plan
        self._expert_step = expert_step
        self._use_hand_finger_point = use_hand_finger_point
        self._use_normal = use_normal

        self._action_space = action_space
        self._env_remove_table_pt_input = env_remove_table_pt_input

        self._pt_accumulate_ratio = pt_accumulate_ratio
        self._domain_randomization = domain_randomization

        self._termination_heuristics = termination_heuristics
        self._omg_config = omg_config
        self._regularize_pc_point_count = regularize_pc_point_count
        self._uniform_num_pts = uniform_num_pts
        self.observation_dim = (self._window_width, self._window_height, 3)

        self.init_constant()
        self.connect()

    def init_constant(self):
        """
        Initialize constants for the environment setup
        """
        self._shift = [0.8, 0.8, 0.8]
        self._max_episode_steps = 50
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.data_root_dir = os.path.join(self.root_dir, 'data/scenes')

        self.retracted = False
        self._standoff_dist = 0.08
        self._max_distance = 0.5

        self.cam_offset = np.eye(4)
        self.cam_offset[:3, 3]  = (np.array([0.036, 0, 0.036]))   # camera offset
        self.cam_offset[:3, :3] = euler2mat(0,0,-np.pi/2)
        self.cur_goal = np.eye(4)

        self.state_uid = None
        self.placed_object_target_idx = 0
        self.target_idx = 0
        self.objects_loaded = False
        self.parallel = False
        self.curr_acc_points = np.zeros([4, 0]) if not self._use_normal else np.zeros([7, 0])
        self.connected = False
        self.action_dim = 6
        self.hand_finger_points = hand_finger_point
        self.action_space =  PandaTaskSpace6D()
        self.arm_collision_point = get_collision_points()

    def connect(self):
        """
        Connect pybullet.
        """
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)

            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(1., 180.0, -41.0, [-0.35, -0.58, -0.88])
        else:
            self.cid = p.connect(p.DIRECT )

        self.connected = True

    def disconnect(self):
        """
        Disconnect pybullet.
        """
        p.disconnect()
        self.connected = False

    def reset(self, save=False, init_joints=None, scene_file=None,
                        data_root_dir=None, cam_random=0,
                        reset_free=False, enforce_face_target=False ):
        """
        Environment reset called at the beginning of an episode.
        """
        self.episode_done = False
        self.retracted = False
        self.under_retract = False
        self.object_grasped = False

        if data_root_dir is not None:
            self.data_root_dir = data_root_dir
        self._cur_scene_file = scene_file
        self.near_point = 0

        if reset_free:
            return self.cache_reset(scene_file, init_joints, enforce_face_target )

        self.disconnect()
        self.connect()

        # Set the camera
        look = [0.1 - self._shift[0], 0.2 - self._shift[1], 0 - self._shift[2]]
        distance = 2.5
        pitch = -56
        yaw = 245
        roll = 0.
        fov = 20.
        aspect = float(self._window_width) / self._window_height
        self.near = 0.1
        self.far = 10
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        self._light_position = np.array([-1.0, 0, 2.5])

        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setGravity(0,0,-9.81)
        p.stepSimulation()

        # Set table and plane
        plane_file = os.path.join(self.root_dir,  'data/objects/floor/model_normalized.urdf')
        table_file = os.path.join(self.root_dir,  'data/objects/table/models/model_normalized.urdf')

        self.obj_path = [plane_file, table_file]
        self.plane_id = p.loadURDF(plane_file, [0 - self._shift[0], 0 - self._shift[1], -.82 - self._shift[2]])
        self.table_pos = np.array([0.5 - self._shift[0], 0.0 - self._shift[1], -.82 - self._shift[2]])
        self.table_id = p.loadURDF(table_file, self.table_pos[0], self.table_pos[1], self.table_pos[2], 0.707, 0., 0., 0.707)

        # Intialize robot and objects
        if init_joints is None:
            self._panda = Panda(stepsize=self._timeStep, base_shift=self._shift)
        else:
            self._panda = Panda(stepsize=self._timeStep, init_joints=init_joints, base_shift=self._shift)
            for _ in range(30): p.stepSimulation()

        if not self.objects_loaded:
            self._objectUids = self.cache_objects()

        if  scene_file is None or not os.path.exists(os.path.join(self.data_root_dir, scene_file + '.mat')):
            self._randomly_place_objects(self._get_random_object(self._numObjects), scale=1)
        else:
            self.place_objects_from_scene(scene_file)

        self._objectUids += [self.plane_id, self.table_id]
        self._env_step = 0
        self.grasp_step = 20
        self.collided = False
        self.grasped = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        self.curr_acc_points = np.zeros([4, 0]) if not self._use_normal else np.zeros([7, 0])
        self.inv_cam_offset = se3_inverse(self.cam_offset)
        self.collision_check()
        self.check_self_collision()

        return None

    def step(self, action, delta=False, repeat=None, config=False, vis=False):
        """
        Environment step.
        """
        repeat = 150
        s = time.time()
        action = self.process_action(action, delta, config)
        if self.object_grasped: action[-2:] = 0
        self._panda.setTargetPositions(action)
        for _ in range(int(repeat)):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

        observation = self._get_observation(vis=False)
        test_termination_obs =  observation[0][1]
        depth = test_termination_obs[[3]].T
        mask = test_termination_obs[[4]].T
        self.collision_check()
        done = self._termination(depth.copy(), mask, observation[0][0], self._termination_heuristics)
        reward = self._reward()

        info = { 'grasp_success': reward,
                 'point_num':self.curr_acc_points.shape[1],
                 'collided': self.obstacle_collided,
                 'cur_ef_pose': self._get_ef_pose(mat=True) }

        self._env_step += 1
        return observation, reward, done, info

    def _get_observation(self, pose=None, vis=False, acc=True ):
        """
        Get observation
        """
        object_pose = self._get_target_relative_pose('ef')
        ef_pose = self._get_ef_pose('mat')
        s = time.time()

        joint_pos, joint_vel = self._panda.getJointStates()
        near, far = self.near, self.far
        view_matrix, proj_matrix = self._view_matrix, self._proj_matrix
        extra_overhead_camera = False
        camera_info = tuple(view_matrix) + tuple(proj_matrix)
        hand_cam_view_matrix, hand_proj_matrix, lightDistance, lightColor, lightDirection, near, far = self._get_hand_camera_view(pose)
        camera_info += tuple(hand_cam_view_matrix.flatten()) + tuple(hand_proj_matrix)
        _, _, rgba, depth, mask = p.getCameraImage(  width=self._window_width,
                                                     height=self._window_height,
                                                     viewMatrix=tuple(hand_cam_view_matrix.flatten()),
                                                     projectionMatrix=hand_proj_matrix,
                                                     physicsClientId=self.cid,
                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # transform depth from NDC to actual depth
        depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16)
        mask[mask >= 0] += 1  # transform mask to have target id 0
        target_idx = self.target_idx + 4

        # remap mask
        # sky -1, floor 1, table 2, robot id 3, others ++
        # -> sky floor -1, target: 0, obstacle 1, robot 2
        mask[mask == target_idx] = 0
        mask[np.logical_or(mask == -1, mask == 1)] = -1 # sky and floor
        mask[mask == 2] = -1 # remove table point
        mask[np.logical_or(mask == 2, mask > 3)] = 1
        mask[mask == 3] = 2

        if False:
            fig = plt.figure(figsize=(16.4, 4.8))
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow((rgba[..., :3]  ).astype(np.uint8))
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(depth)
            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(mask)
            plt.show()

        obs = np.concatenate([rgba[..., :3], depth[...,None], mask[...,None] ], axis=-1)
        obs = self.process_image(obs[...,:3], obs[...,[3]], obs[...,[4]] , tuple(self._resize_img_size))
        intrinsic_matrix = projection_to_intrinsics(hand_proj_matrix, self._window_width, self._window_height)
        point_state = backproject_camera_target(obs[3].T, intrinsic_matrix, obs[4].T, obs[-3:].T)
        point_state[:3] = self.cam_offset[:3,:3].dot(point_state[:3]) + self.cam_offset[:3, [3]]

        point_state[1] *= -1
        point_state = self.process_pointcloud(point_state, rgba[..., :3], vis, acc)
        obs = (point_state, obs[:5])
        pose_info = (object_pose, ef_pose)

        return [obs, joint_pos, camera_info, pose_info]


    def retract(self, record=False, overhead_view_param=None):
        """
        Move the arm to lift the object.
        """
        overhead_observations = self.append_overhead_view(overhead_view_param, [])
        self.under_retract = True
        cur_joint = np.array(self._panda.getJointStates()[0])
        cur_joint[-2:] = 0 # close finger
        observations = [self.step(cur_joint, repeat=300, config=True, delta=False, vis=False)[0]]
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]

        for i in range(5):
            pos = (pos[0], pos[1], pos[2] + 0.03)
            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                                               self._panda.pandaEndEffectorIndex, pos))
            jointPoses[-2:] = 0.0
            obs = self.step(jointPoses, config=True, delta=False)[0]
            if record:
                observations.append(obs)
                overhead_observations = self.append_overhead_view(overhead_view_param, overhead_observations)

        self.grasp_step = self._env_step
        self.retracted = True
        self.under_retract = False
        rew = self._reward(terminal=True)

        self.object_grasped = rew == 1
        self.episode_done = True
        if record:
            return (rew, observations, overhead_observations)
        return rew

    def _reward(self, terminal=False):
        """
        Calculates the reward for the episode.
        """
        reward = 0
        if self.obstacle_collided:
            reward = -1
        if terminal:
            if self.target_lifted():
                print('target {} grasp success!'.format(self.target_name))
                reward = 1 #
            else:
                print('target {} grasp failed !'.format(self.target_name))
        return reward

    def _termination(self, depth_img, mask_img, point_state, use_depth_heuristics=False):
        """
        Target depth heuristics for determining if grasp can be executed.
        The threshold is based on depth in the middle of the camera and the finger is near the bottom two sides
        """

        depth_heuristics = False
        nontarget_mask = mask_img[...,0] != 0

        if use_depth_heuristics and not self.object_grasped:
            depth_img = depth_img[...,0]
            depth_img[nontarget_mask] = 10

            # hard coded region
            depth_img_roi = depth_img[int(38. * self._window_height / 64):,
                            int(24. * self._window_width / 64):int(48 * self._window_width / 64)]
            depth_img_roi_ = depth_img_roi[depth_img_roi < 0.1]
            if depth_img_roi_.shape[0] > 1:
                self.near_point = (depth_img_roi_ < 0.045).sum()
                depth_heuristics = self.near_point > 30
            done = depth_heuristics

        return self._env_step >= self._maxSteps or done or self.target_fall_down() or self.obstacle_collided

    def cache_objects(self):
        """
        Load all YCB objects and set up
        """
        obj_path = os.path.join(self.root_dir, self.obj_root_dir)
        objects  = self.obj_indexes
        obj_path = [os.path.join(obj_path,  objects[i]) for i in self._all_obj]

        pose = np.zeros([len(obj_path), 3])
        pose[:, 0] = -0.5 - np.linspace(0, 4, len(obj_path))
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        objects_paths = [p_.strip() + '/' for p_ in obj_path]
        objectUids = []

        self.target_obj_indexes = [self._all_obj.index(idx) for idx in self._target_objs]
        self.obj_path = objects_paths + self.obj_path
        self.placed_object_poses = []

        for i, name in enumerate(objects_paths):
            trans = pose[i] + np.array(pos)
            self.placed_object_poses.append((trans.copy(), np.array(orn).copy()))
            uid = self._add_mesh(os.path.join(self.root_dir, name, 'model_normalized.urdf'), trans, orn)
            objectUids.append(uid)
            p.setCollisionFilterPair(uid, self.plane_id, -1, -1, 0)

        self.objects_loaded = True
        self.placed_objects = [False] * len(self.obj_path)
        return objectUids

    def cache_reset(self, scene_file, init_joints, enforce_face_target):
        """
        Reset the loaded objects around to avoid loading multiple times
        """

        self.place_back_objects()
        self._panda.reset(init_joints)
        if scene_file is None or not os.path.exists(os.path.join(self.data_root_dir, scene_file + '.mat')):
            self._randomly_place_objects(self._get_random_object(self._numObjects), scale=1)
        else:
            self.place_objects_from_scene(scene_file, self._objectUids)

        self._env_step = 0
        self.retracted = False
        self.collided = False
        self.obstacle_collided = False
        self.obj_names, self.obj_poses = self.get_env_info()
        self.init_target_height = self._get_target_relative_pose()[2, 3]
        self.curr_acc_points = np.zeros([4, 0]) if not self._use_normal else np.zeros([7, 0])

        observation = self.enforce_face_target() if enforce_face_target else self._get_observation()
        self.collision_check()
        self.check_self_collision()
        return observation

    def place_objects_from_scene(self, scene_file, objectUids=None):
        """
        Place objects with poses based on the scene file
        """
        if self.objects_loaded:
            objectUids = self._objectUids

        scene = sio.loadmat(os.path.join(self.data_root_dir, scene_file + '.mat'))
        poses = scene['pose']
        path = scene['path']
        target_idx = int(scene['target_idx'])

        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        new_objs = objectUids is None
        objects_paths = [p_.strip() + '/' for p_ in path]

        for i, name in enumerate(objects_paths[:-2]):

            pose = poses[i]
            trans = pose[:3, 3] + np.array(pos)
            orn = ros_quat(mat2quat(pose[:3, :3]))
            full_name = os.path.join(self.root_dir, name)
            if full_name not in self.obj_path: continue

            k = self.obj_path.index(full_name) if self.objects_loaded else i
            self.placed_objects[k] = True
            p.resetBasePositionAndOrientation(objectUids[k], trans, orn)
            p.resetBaseVelocity(objectUids[k], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        self.target_idx = self.obj_path.index(os.path.join(self.root_dir, objects_paths[target_idx]))
        self.target_name = objects_paths[target_idx].split('/')[-2]

        if 'proj_matrix' in scene:
            self._proj_matrix = tuple(scene['proj_matrix'][0])
        if 'view_matrix' in scene:
            self._view_matrix = tuple(scene['view_matrix'][0])
        if 'init_joints' in scene:
            self.reset_joint(scene['init_joints'])

        print('==== loaded scene: {} target: {} idx: {}'.format(scene_file.split('/')[-1],
                self.target_name, self.target_idx))
        self._check_object_wait()
        return objectUids


    def update_curr_acc_points(self, new_points):
        """
        Update accumulated points in world coordinate
        """
        if self.object_grasped: return
        pos, rot = self._get_ef_pose()
        ef_pose  = unpack_pose(np.hstack((pos, tf_quat(rot))))
        new_points[:3] = se3_transform_pc(ef_pose, new_points[:3])
        if self._use_normal:
            new_points[-3:] = ef_pose[:3,:3].dot(new_points[-3:])
        self.curr_acc_points = np.concatenate((new_points, self.curr_acc_points), axis=1) #


    def reset_joint(self, init_joints):
        if init_joints is not None:
          self._panda.reset(np.array(init_joints).flatten())

    def process_action(self, action, delta=False, config=False):
        """
        Process different action types
        """
        if  config or len(action) != 6:
            delta_joint =   delta
            if len(action) == 7:
                if not delta_joint:
                    action = np.concatenate((action, [0.04,0.04]))
                else:
                    action = np.concatenate((action, [0.,0.]))

        else:
            # transform to local coordinate
            cur_ef = np.array(self._panda.getJointStates()[0])[-3]
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]

            pose = np.eye(4)
            pose[:3, :3] = quat2mat(tf_quat(orn))
            pose[:3, 3] = pos

            pose_delta = np.eye(4)
            pose_delta[:3, :3] = euler2mat(action[3], action[4], action[5])
            pose_delta[:3, 3] = action[:3]

            new_pose = pose.dot(pose_delta)
            orn = ros_quat(mat2quat(new_pose[:3, :3]))
            pos = new_pose[:3, 3]

            jointPoses = np.array(p.calculateInverseKinematics(self._panda.pandaUid,
                                  self._panda.pandaEndEffectorIndex, pos, orn))
            jointPoses[-2:] = 0.04
            action = jointPoses
        return action

    def _get_hand_camera_view(self, cam_pose=None):
        """
        Get hand camera view
        """
        if cam_pose is None:
            pos, orn = p.getLinkState(self._panda.pandaUid, 10)[:2]
            cam_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cam_pose_mat = unpack_pose(cam_pose)

        fov = 90
        aspect = float(self._window_width) / (self._window_height)
        hand_near = 0.035
        hand_far =  2
        hand_proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, hand_near, hand_far)
        hand_cam_view_matrix = se3_inverse(cam_pose_mat.dot(rotX(-np.pi/2).dot(rotZ(-np.pi)))).T # z backward

        lightDistance = 2.0
        lightDirection = self.table_pos - self._light_position
        lightColor = np.array([1., 1., 1.])
        light_center = np.array([-1.0, 0, 2.5])
        return  hand_cam_view_matrix, hand_proj_matrix, lightDistance, \
                lightColor, lightDirection, hand_near, hand_far

    def target_fall_down(self):
        """
        Check if target has fallen down
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height < -0.03:
            return True
        return False

    def target_lifted(self):
        """
        Check if target has been lifted
        """
        end_height = self._get_target_relative_pose()[2, 3]
        if end_height - self.init_target_height > 0.04:
            return True
        return False

    def place_from_aabb(self, obj_idx, trans, table_height):
        """
        compute bounding boxes for placement height
        """
        orn = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])

        p.resetBasePositionAndOrientation(
            self._objectUids[obj_idx],
            [trans[0], trans[1], 1],
            [orn[0], orn[1], orn[2], orn[3]],
        )
        lower, upper =  p.getAABB(self._objectUids[obj_idx])
        z_init = table_height + 0.5 * (upper[2] - lower[2])
        p.resetBasePositionAndOrientation(
            self._objectUids[obj_idx],
            [trans[0], trans[1], z_init - self._shift[2]],
            [orn[0], orn[1], orn[2], orn[3]],
        )  # xyzw
        p.resetBaseVelocity(
            self._objectUids[obj_idx], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        )
        self.placed_objects[obj_idx] = True

    def _randomly_place_objects(self, urdfList, scale=1, poses=None):
        """
        Randomize positions of each object urdf.
        """
        xpos = 0.6 + 0.2 * (self._blockRandom * random.random() - 0.5)
        ypos = 0.5 * self._blockRandom * (random.random() - 0.5)
        target_m_pos = np.array([xpos, ypos])
        xpos = xpos - self._shift[0]
        ypos = ypos - self._shift[1]
        table_lower, table_upper = p.getAABB(self.table_id)

        self.target_idx = self.obj_path.index(urdfList[0])
        self.target_name = urdfList[0].split('/')[-2]
        table_height = table_upper[2] + self._shift[2]
        self.place_from_aabb(self.target_idx, [xpos, ypos], table_height)
        self.placed_objects[self.target_idx] = True
        for _ in range(200):
            p.stepSimulation()

        #### candidate grid positions
        x_min, x_max =  0.23, 0.65
        y_min, y_max = -0.23, 0.23
        xv, yv  = np.meshgrid(np.linspace(x_min + 0.04, x_max - 0.05, 3), np.linspace(y_min + 0.03, y_max - 0.03, 3), indexing='ij')
        xv, yv  = xv.flatten(), yv.flatten()
        target_m = np.argmin(np.abs(np.subtract(np.stack((xv, yv), axis=-1), target_m_pos)).sum(axis=-1))
        candidate_prob = 1.
        m = 0

        for i, name in enumerate(urdfList[1:]):
            obj_idx = self.obj_path.index(name)
            if self.placed_objects[obj_idx] or m >= len(xv) - 1:
                continue

            m += 1
            # use grid positions to sample objects
            delta_x = self._blockRandom * 0.05 * (random.random() - 0.5)
            delta_y = self._blockRandom * 0.05 * (random.random() - 0.5)
            xpos_ = xv[m] - delta_x - self._shift[0]
            ypos_ = yv[m] - delta_y - self._shift[1]

            xy = np.array([[xpos_, ypos_]])
            self.place_from_aabb(obj_idx, [xpos_, ypos_ ],  table_height)
            p.stepSimulation()
            points = p.getContactPoints(self._objectUids[obj_idx])

            if len(points) > 0:
                self.placed_objects[obj_idx] = False
                self.place_back_object(obj_idx)
                m -= 1
                continue

            p.stepSimulation()
        s = time.time()
        self._check_object_wait()
        return []


    def _check_object_wait(self, timeout=1000, tol=0.01):
        """
        wait until all objects rest
        """
        objects_resting = False
        total_step = 0
        while not objects_resting and total_step < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                p.stepSimulation()
            total_step += 60

            # check whether all objects are resting
            objects_resting = True
            for obj_idx, flag in enumerate(self.placed_objects):
                if flag and np.linalg.norm(p.getBaseVelocity(self._objectUids[obj_idx])) > tol:
                    objects_resting = False
                    break

    def _get_random_object(self, num_objects ):
        """
        Randomly choose an object urdf from the selected objects
        """
        self.target_idx = self._all_obj.index(
            self._target_objs[np.random.randint(0, len(self._target_objs))]
        )  #
        obstacle = np.random.choice(
            range(len(self._all_obj)), min(self._numObjects - 1, len(self._all_obj)), replace=False
        ).tolist()
        selected_objects = [self.target_idx] + obstacle
        selected_objects_filenames = [
            self.obj_path[selected_object] for selected_object in selected_objects
        ]
        return selected_objects_filenames

    def _load_index_objs(self, file_dir, target_file_dir=None, obj_root_dir='data/objects'):
        """
        Load object indexes
        """
        if target_file_dir is not None:
            target_obj_list = [file_dir.index(file) for file in target_file_dir]
        else:
            target_obj_list   = range(len(file_dir))
        self.obj_root_dir = obj_root_dir
        self._target_objs = target_obj_list
        self._all_obj = range(len(file_dir))
        self.obj_indexes = file_dir


    def enforce_face_target(self):
        """
        Move the gripper to face the target
        """
        target_forward = self._get_target_relative_pose('ef')[:3, 3]
        target_forward = target_forward / np.linalg.norm(target_forward)
        r = a2e(target_forward)
        action = np.hstack([np.zeros(3), r])
        obs = self.step(action, repeat=200, vis=False)[0]
        self.curr_acc_points = np.zeros([4, 0])
        return obs


    def randomize_arm_init(self, near=0.35, far=0.50):
        """
        randomize initial joint configuration from a hemisphere looking at target
        """
        target_pos = self._get_target_relative_pose('base')[:3, 3]
        target_forward = np.array([0.4, 0, target_pos[-1]])#
        init_joints = sample_ef(target_forward, near=near, far=far)

        if init_joints is not None:
            return list(init_joints) + [0, 0.04, 0.04]
        return None


    def random_perturb(self, t_delta=0.03, r_delta=0.2):
        """
        Random perturb of arm
        """
        t = np.random.uniform(-t_delta, t_delta, size=(3,))
        r = np.random.uniform(-r_delta, r_delta, size=(3,))
        action = np.hstack([t, r])
        obs = self.step(action, repeat=150, vis=False)[0]
        return obs

    def check_self_collision(self):
        """
        Check robot self collision
        """
        link_list = list(range(7))

        for link1, link2 in product(link_list, link_list):
            if np.abs(link1 - link2) <= 1: continue

            if pairwise_link_collision(self._panda.pandaUid, link1, self._panda.pandaUid, link2, CLIENT=self.cid):
                self.collided = True


    def collision_check(self):
        """
        Check collision against all links
        """
        if self.under_retract:
            return

        if any_link_pair_collision(self._objectUids[self.target_idx], self._panda.pandaUid, CLIENT=self.cid):
            if self._accumulate_points and self.curr_acc_points.shape[1] > self._uniform_num_pts:
                target_mask = get_target_mask(self.curr_acc_points)
                if len(target_mask.shape) > 1:
                    reset_pt_num = min(300, target_mask.shape[1])
                else:
                    reset_pt_num = 0
                self.curr_acc_points =   np.concatenate(
                                         (regularize_pc_point_count(self.curr_acc_points[:,target_mask].T, reset_pt_num).T,
                                         self.curr_acc_points[:, ~target_mask]), axis=-1)

            self.collided = True
            return

        for idx, uid in enumerate(self._objectUids):
            if idx == self.target_idx: continue
            if self.placed_objects[idx] or idx >= len(self._objectUids) - 2:
                if any_link_pair_collision(self._objectUids[idx], self._panda.pandaUid, CLIENT=self.cid) \
                                            and self._env_step < self._expert_step:
                    # tolerance in the grasping stage.
                    self.obstacle_collided = True
                    self.collided = True
                    self.episode_done = True
                    print('collided with obstacle')

    def process_image(self, color, depth, mask, size=None):
        """
        Normalize RGBDM
        """
        color = color.astype(np.float32) / 255.0
        mask  = mask.astype(np.float32)
        depth = depth.astype(np.float32) / 5000
        if size is not None:
            color = cv2.resize(color, size)
            mask  = cv2.resize(mask, size)
            depth = cv2.resize(depth, size)
        obs = np.concatenate([color, depth[...,None], mask[...,None]], axis=-1)
        obs = obs.transpose([2, 1, 0])
        return obs

    def process_pointcloud(self, point_state, color, vis, acc_pt=True, use_farthest_point=False):
        """
        Process point cloud input, aggregating  and downsampling points
        """
        s = time.time()
        total_point_num = 2000
        target_portion = 0.1
        max_observe_point = int(total_point_num * (self._pt_accumulate_ratio**self._env_step))
        max_target_observe_point = int(total_point_num * target_portion * (self._pt_accumulate_ratio**self._env_step))
        point_state = point_state[:4]
        self.curr_observ_point = point_state

        point_state = point_state[:, ~get_robot_mask(point_state)]
        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            # restrict maximum number of added points
            point_state = self._bias_target_pc_regularize(point_state, max_observe_point, max_target_observe_point )

        pos, rot = self._get_ef_pose()
        ef_pose = se3_inverse(unpack_pose(np.hstack((pos, tf_quat(rot)))))

        if self._accumulate_points and acc_pt:
            self.update_curr_acc_points(point_state)
            point_state = self.curr_acc_points.copy()
            point_state  = se3_transform_pc(ef_pose, point_state)

        if self._regularize_pc_point_count and point_state.shape[1] > 0:
            point_state = self._bias_target_pc_regularize(point_state, self._uniform_num_pts )

        # dummies
        robot_point_state = np.ones((4, 6)) * -1 if not self._use_normal else np.ones((7, 6)) * -1
        robot_point_state[:3] = self.hand_finger_points
        point_state = np.concatenate((robot_point_state, point_state), axis=1)
        robot_arm_point_state = np.ones([4, 500]) * 2
        point_state = np.concatenate((point_state, robot_arm_point_state), axis=1)
        return point_state[:4]


    def _bias_target_pc_regularize(self, point_state, total_point_num, target_pt_num=1024, use_farthest_point=True ):
        """
        farthest point downsampling with fixed target number of point
        """
        target_mask = get_target_mask(point_state)
        target_pt = point_state[:, target_mask]
        nontarget_pt = point_state[:, ~target_mask]

        if target_pt.shape[1] > 0:
            target_pt = regularize_pc_point_count(target_pt.T, target_pt_num, use_farthest_point).T
        if nontarget_pt.shape[1] > 0:
            effective_target_pt_num = min(target_pt_num, target_pt.shape[1])
            nontarget_pt = regularize_pc_point_count(nontarget_pt.T, total_point_num - effective_target_pt_num, use_farthest_point).T

        return np.concatenate((target_pt, nontarget_pt), axis=1)

    def get_overhead_image_observation(self, look, distance, pitch, yaw, roll, fov):
        """
        Global overhead view
        """
        width = 320
        height = 240
        aspect = float(width) / height
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, self.near, self.far)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)

        _, _, rgba, depth, mask = p.getCameraImage(width=width,
                                                   height=height,
                                                   viewMatrix=view_matrix,
                                                   projectionMatrix=proj_matrix,
                                                   physicsClientId=self.cid,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL )
        far, near = self.far, self.near
        depth = (far * near / (far - (far - near) * depth) * 5000).astype(np.uint16)
        obs = np.concatenate([rgba[..., :3], depth[...,None], mask[...,None]], axis=-1)
        target_idx = self.target_idx + 4
        obs = np.concatenate([rgba[..., :3] / 255.0, depth[...,None] / 5000., mask[...,None]], axis=-1)
        obs = rgba[..., :3]
        extrinsic_matrix = view_to_extrinsics(view_matrix)
        intrinsic_matrix = projection_to_intrinsics(proj_matrix, width, height)
        return obs, extrinsic_matrix, intrinsic_matrix

    def append_overhead_view(self, overhead_view_param, overhead_observations):
        if overhead_view_param is not None:
            if type(overhead_view_param) is list:
                overhead_observation =  [self.get_overhead_image_observation(*param)[0] for param in overhead_view_param]
            else:
                overhead_observation = self.get_overhead_image_observation(*overhead_view_param)[0]
            overhead_observations.append(overhead_observation)
        return overhead_observations


    def _get_relative_ef_pose(self):
        """
        Get all obejct poses with respect to the end effector
        """
        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                pos, orn = p.getBasePositionAndOrientation(uid) # to target
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, ef_pose))
        return poses

    def _get_relative_goal_pose(self, rotz=False, mat=False, nearest=False):
        """
        Get the relative pose from current to the goal
        """

        pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        ef_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        pos, orn = p.getBasePositionAndOrientation(self._objectUids[self.target_idx]) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        cur_goal_mat = unpack_pose(obj_pose).dot(self.cur_goal)
        cur_goal = pack_pose(cur_goal_mat)
        if mat:
            return inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)) if rotz else inv_relative_pose(cur_goal, ef_pose)
        if rotz:
            return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose).dot(rotZ(np.pi/2)))
        return pack_pose_rot_first(inv_relative_pose(cur_goal, ef_pose))

    def _get_ef_pose(self, mat=False):
        """
        end effector pose in world frame
        """
        if not mat:
            return p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        else:
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
            return unpack_pose(list(pos) + [orn[3], orn[0], orn[1], orn[2]])

    def get_base_pose(self):
        """
        robot base pose in world frame
        """
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        base_pose = unpack_pose(base_pose)
        return base_pose

    def _get_target_relative_pose(self, option='base'):
        """
        Get target obejct poses with respect to the different frame.
        """
        if option == 'base':
            pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        elif option == 'ef':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
        elif option == 'tcp':
            pos, orn = p.getLinkState(self._panda.pandaUid, self._panda.pandaEndEffectorIndex)[:2]
            rot = quat2mat(tf_quat(orn))
            tcp_offset = rot.dot(np.array([0,0,0.13]))
            pos = np.array(pos) + tcp_offset

        pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        uid = self._objectUids[self.target_idx]
        pos, orn = p.getBasePositionAndOrientation(uid) # to target
        obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        return inv_relative_pose(obj_pose, pose)

    def _get_init_info(self):
        return [self.obj_names, self.obj_poses, self.placed_object_target_idx,
                np.array(self._panda.getJointStates()[0]),
                self.planner_goal_set, self.planner_reach_grasps]

    def get_env_info(self, scene_file=None):
        """
        Return object names and poses of the current scene
        """
        pos, orn = p.getBasePositionAndOrientation(self._panda.pandaUid)
        base_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
        poses = []
        obj_dir = []

        for idx, uid in enumerate(self._objectUids):
            if self.placed_objects[idx] or idx >= len(self._objectUids) - 2:
                pos, orn = p.getBasePositionAndOrientation(uid) # center offset of base
                obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
                poses.append(inv_relative_pose(obj_pose, base_pose))
                obj_dir.append('/'.join(self.obj_path[idx].split('/')[:-1]).strip())
                if idx == self.target_idx:
                    self.placed_object_target_idx = len(obj_dir) - 1

        return obj_dir, poses

    def _add_mesh(self, obj_file, trans, quat, scale=1):
        """
        Add a mesh with URDF file.
        """
        return p.loadURDF(obj_file, trans, quat, globalScaling=scale, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    def place_back_object(self, idx):
        """
        place back a single object
        """
        p.resetBasePositionAndOrientation(self._objectUids[idx],
                                          self.placed_object_poses[idx][0],
                                          self.placed_object_poses[idx][1])
        self.placed_objects[idx] = False
        p.stepSimulation()

    def place_back_objects(self):
        """
        place object back from table
        """
        for idx, obj in enumerate(self._objectUids):
            if self.placed_objects[idx]:
                p.resetBasePositionAndOrientation(obj, self.placed_object_poses[idx][0], self.placed_object_poses[idx][1])
            self.placed_objects[idx] = False

if __name__ == '__main__':
    pass
