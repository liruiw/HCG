# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pybullet as p
import numpy as np
import IPython

try:
    from OMG.omg.config import cfg as planner_cfg
    from OMG.omg.core import PlanningScene
except:
    pass
from core.utils import *

class EnvPlanner():
    """
    Class for franka panda environment with YCB objects.
    """

    def __init__(self, env=None):
        self.env = env
        self._planner_setup = False
        self.env.planner_reach_grasps = []
        self.env.planner_goal_set = []
        self.planner_change_goal = False

    def plan_to_conf(self, step, start_conf, goal_conf, vis=False):
        """
        Plan to a fix configuration
        """
        planner_cfg.timesteps = step # 20
        planner_cfg.get_global_param(step)
        self.planner_scene.traj.start = start_conf
        dummy_goal_set = planner_cfg.goal_set_proj
        setattr(planner_cfg, 'goal_set_proj', False)
        planner_cfg.get_global_param()
        self.planner_scene.traj.end = goal_conf
        self.planner_scene.update_planner()
        info = self.planner_scene.step()
        setattr(planner_cfg, 'goal_set_proj', dummy_goal_set)
        planner_cfg.get_global_param()
        plan = self.planner_scene.planner.history_trajectories[-1]
        return plan


    def setup_expert_scene(self):
        """
        Load all meshes once and then update pose
        """
        # parameters
        print('set up expert scene ...')
        for key, val in self.env._omg_config.items():
            setattr(planner_cfg, key, val)

        planner_cfg.get_global_param(planner_cfg.timesteps)
        planner_cfg.get_global_path()

        # load obstacles
        self.planner_scene = PlanningScene(planner_cfg)
        self.planner_scene.traj.start = np.array(self.env._panda.getJointStates()[0])
        self.planner_scene.env.clear()

        obj_names, obj_poses = self.env.get_env_info(self.env._cur_scene_file)
        object_lists = [name.split('/')[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]

        for i, name in enumerate(self.env.obj_path[:-2]):
            name = name.split('/')[-2]
            trans, orn = self.env.placed_object_poses[i]
            self.planner_scene.env.add_object(name, trans, tf_quat(orn), compute_grasp=True)

        self.planner_scene.env.add_plane(np.array([0.05, 0, -0.17]), np.array([1,0,0,0])) # never moved
        self.planner_scene.env.add_table(np.array([0.55, 0, -0.17]), np.array([0.707, 0.707, 0., 0]))
        self.planner_scene.env.combine_sdfs()
        self._planner_setup = True

    def expert_plan_grasp(self, step=-1, check_scene=False, checked=False, fix_goal=False):
        """
        Generate expert to grasp the target object
        """
        joint_pos = self.env._panda.getJointStates()[0]
        self.planner_scene.traj.start = np.array(joint_pos)
        self.planner_scene.env.set_target(self.env.obj_path[self.env.target_idx].split('/')[-2]) #scene.env.names[0])
        info = []
        if step > 0: # continued plan
            self.get_planning_scene_target().compute_grasp = False
            planner_cfg.timesteps = step # 20
            if fix_goal:
                planner_cfg.goal_idx = self.planner_scene.traj.goal_idx
                planner_cfg.ol_alg = "Baseline"
            planner_cfg.get_global_param(planner_cfg.timesteps)
            self.planner_scene.reset(lazy=True)
            self.prev_planner_goal_idx = self.planner_scene.traj.goal_idx

            if  not check_scene:
                info = self.planner_scene.step()

            planner_cfg.timesteps = self.env._expert_step # set back
            planner_cfg.get_global_param(planner_cfg.timesteps)
            if  fix_goal: # reset
                planner_cfg.goal_idx = -1
                planner_cfg.ol_alg = "MD"
            self.planner_change_goal = self.prev_planner_goal_idx != self.planner_scene.traj.goal_idx
            self.prev_planner_goal_idx = self.planner_scene.traj.goal_idx

        else:
            if checked:
                self.get_planning_scene_target().compute_grasp = False

            self.planner_scene.reset(lazy=True)
            if not check_scene:
                info = self.planner_scene.step()
            self.planner_change_goal = False
            self.prev_planner_goal_idx = -1

        return info

    def expert_plan(self, step=-1, return_success=False, check_scene=False, checked=False, fix_goal=True,
                    joints=None, ef_pose=None, pts=None, vis=False):
        """
        Run OMG planner for the current scene
        """
        if not self._planner_setup:
            self.setup_expert_scene()

        obj_names, obj_poses = self.env.get_env_info(self.env._cur_scene_file)
        object_lists = [name.split('/')[-1].strip() for name in obj_names]
        object_poses = [pack_pose(pose) for pose in obj_poses]
        exists_ids = []
        placed_poses = []
        placed_obj_names = []
        plan = []

        if self.env.target_idx == -1 or self.env.target_name == 'noexists':
            if not return_success:
                return [], np.zeros(0)
            return [], np.zeros(0), False

        for i, name in enumerate(object_lists[:-2]):  # buffer
            self.planner_scene.env.update_pose(name, object_poses[i])
            idx = self.env.obj_path[:-2].index(os.path.join(self.env.root_dir, 'data/objects/' + name + '/'))
            exists_ids.append(idx)
            trans, orn = self.env.placed_object_poses[idx]
            placed_poses.append(np.hstack([trans, ros_quat(orn)]))
            placed_obj_names.append(name)

        planner_cfg.disable_collision_set = [name.split('/')[-2] for idx, name in enumerate(self.env.obj_path[:-2])
                                             if idx not in exists_ids]

        info = self.expert_plan_grasp(step, check_scene, checked, fix_goal)

        if planner_cfg.vis: self.planner_scene.fast_debug_vis(collision_pt=False)

        for i, name in enumerate(placed_obj_names): # reset
            self.planner_scene.env.update_pose(name, placed_poses[i])

        if check_scene:
            return len(self.planner_scene.traj.goal_set) >= 15

        if  hasattr(self.planner_scene, 'planner') and len(self.planner_scene.planner.history_trajectories) >= 1: #
            plan = self.planner_scene.planner.history_trajectories[-1]
            if not hasattr(self.env, 'robot'):
                self.env.robot = require_robot()
            ef_pose = self.env.get_base_pose().dot(self.env.robot.forward_kinematics_parallel(
                                wrap_value(plan[-1])[None], offset=False)[0][-3]) # world coordinate

            pos, orn = p.getBasePositionAndOrientation(self.env._objectUids[self.env.target_idx]) # to target
            obj_pose = list(pos) + [orn[3], orn[0], orn[1], orn[2]]
            self.env.cur_goal = se3_inverse(unpack_pose(obj_pose)).dot(ef_pose)


        success = info[-1]['terminate'] if len(info) > 1 else False
        failure = info[-1]['failure_terminate'] if len(info) > 1 else True
        self.env.planner_reach_grasps = self.get_planning_scene_target().reach_grasps
        self.env.planner_goal_set = self.planner_scene.traj.goal_set

        if not return_success:
            return plan, failure
        return plan, failure, success

    def set_planner_scene_file(self, scene_path, scene_file):
        planner_cfg.traj_init = "scene"
        planner_cfg.scene_path = scene_path
        planner_cfg.scene_file = scene_file

    def get_planning_scene_target(self):
        return self.planner_scene.env.objects[self.planner_scene.env.target_idx]

    def _get_init_info(self):
        """
        Get environment information
        """
        return [self.obj_names, self.obj_poses, self.placed_object_target_idx,
                np.array(self._panda.getJointStates()[0]),
                self.planner_scene.traj.goal_set,
                self.get_planning_scene_target().reach_grasps]

    def goal_num_full(self):
        """
        Check if the scene has enough goals
        """
        return not self._planner_setup or len(self.planner_scene.traj.goal_set) >= 6
