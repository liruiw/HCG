# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Edited from https://github.com/bryandlee/franka-pybullet/blob/master/src/panda.py

import pybullet as p
import numpy as np
import IPython
import os

class Panda:
    def __init__(self, stepsize=1e-3, realtime=0, init_joints=None, base_shift=[0,0,0], bullet_client=None):
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.control_mode = "position" 
        self.client = p if bullet_client is None else bullet_client
        self.position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self.position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        f_max = 250
        self.max_torque = [f_max,f_max,f_max,f_max,f_max,f_max,f_max,150,150,150,150] 
           
        # connect pybullet
        self.client.setRealTimeSimulation(self.realtime)

        # load models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.client.setAdditionalSearchPath(current_dir + "/models")
        self.robot = self.client.loadURDF("panda/panda_gripper_hand_camera.urdf",
                                useFixedBase=True,
                                flags=self.client.URDF_USE_SELF_COLLISION)
        self._base_position = [-0.05 - base_shift[0], 0.0 - base_shift[1], -0.65 - base_shift[2]]
        self.pandaUid = self.robot        

        # robot parameters
        self.dof = self.client.getNumJoints(self.robot)
        c = self.client.createConstraint(self.robot,
                       8,
                       self.robot,
                       9,
                       jointType=self.client.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
        self.client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_torque = []
        self.pandaEndEffectorIndex = 7
        self._joint_min_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0, 0, 0, 0])
        self._joint_max_limit = np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0, 0.04, 0.04, 0])

        for j in range(self.dof):
            self.client.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_info = self.client.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)
        self.reset(init_joints)
        

    def reset(self, joints=None):
        self.t = 0.0        
        self.control_mode = "position"
        self.client.resetBasePositionAndOrientation(self.pandaUid, self._base_position,
                                      [0.000000, 0.000000, 0.000000, 1.000000])
        if joints is None:
            self.target_pos = [  
                    0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0, 0.04, 0.04]
             
            self.target_pos = self.standardize(self.target_pos)
            for j in range(self.dof):
                self.target_torque[j] = 0.
                self.client.resetJointState(self.robot,j,targetValue=self.target_pos[j])
        
        else:
            joints = self.standardize(joints)
            for j in range(self.dof):
                self.target_pos[j] = joints[j]
                self.target_torque[j] = 0.
                self.client.resetJointState(self.robot,j,targetValue=self.target_pos[j])
        self.resetController()
        self.setTargetPositions(self.target_pos)
      
    def step(self):
        self.t += self.stepsize
        self.client.stepSimulation()

    def resetController(self):
        self.client.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=self.client.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def standardize(self, target_pos):
        if type(target_pos) is np.ndarray and len(target_pos.shape) == 2:
            target_pos = target_pos[0]

        if len(target_pos) == 9:
            if type(target_pos) == list:
                target_pos.insert(7, 0)
            else:
                target_pos = np.insert(target_pos, 7, 0)
        target_pos = np.array(target_pos)
        if len(target_pos) == 10:
            target_pos = np.append(target_pos, 0)
       
        target_pos = np.minimum(np.maximum(target_pos, self._joint_min_limit), self._joint_max_limit)
        return target_pos

    def setTargetPositions(self, target_pos):
        self.target_pos = self.standardize(target_pos)
        self.client.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=self.client.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)
 
    def getJointStates(self):
        joint_states = self.client.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        if len(joint_pos) == 11:
            del joint_pos[7], joint_pos[-1]
            del joint_vel[7], joint_vel[-1]
        return joint_pos, joint_vel 
 
if __name__ == "__main__":
    robot = Panda(realtime=1)
    while True:
        pass
