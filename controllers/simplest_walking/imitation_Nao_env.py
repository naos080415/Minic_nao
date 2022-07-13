import sys
import numpy as np

from controller import Supervisor, Node
from gym.spaces import Box
from gym.utils import seeding
import math

from utils.RobotUtils import RobotFunc
from utils.MathUtils import GetQuaternionFromAxisAngle, GetEulerFromOrientation, GetQuaternionFromEuler, GetAxisAngleFromQuaternion
from utils.MathUtils import Scaler

import random

class robotisImitationEnv():
    def __init__(self, timestep=None, trajectorypath="./ref-Data/trajectory.csv", init_actionpath='./ref-Data/initAct.csv', ref_actionpath="./ref-Data/refAct.csv"):
        self.supervisor = Supervisor()

        if timestep is None:
            self.timestep = int(self.supervisor.getBasicTimeStep())
        else:
            self.timestep = timestep

        # self.trajectory = np.loadtxt(trajectorypath, delimiter=',', dtype=np.float32)
        self.init_action = np.loadtxt(init_actionpath, delimiter=',', dtype=np.float32)
        self.ref_action = np.loadtxt(ref_actionpath, delimiter=',', dtype=np.float32)

        self.setup_agent()      # motor, sensor setting

        self.numObs, self.numAct = 3, 10
        # Lower and maximum values on observation space
        lowObs = -np.inf * np.ones(self.numObs)
        maxObs = np.inf * np.ones(self.numObs)
        self.observation_space = Box(low=lowObs, high=maxObs, dtype=np.float32)

        # Lower and maximum values on action space
        lowAct = -1.0 * np.ones(self.numAct)
        maxAct = 1.0 * np.ones(self.numAct)
        self.action_space = Box(low=lowAct, high=maxAct, dtype=np.float32)

        self.com_pos = np.array([0, 0, 0])
        self.com_pos_his = np.array([0, 0, 0])
        
        self.phase = 0

        self.cnt = 0


    def setup_agent(self):
        """
        This method initializes the motors, 
        storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.robot = self.supervisor.getFromDef("ROBOT")

        self.translation_field = self.robot.getField("translation")
        self.rotation_field = self.robot.getField("rotation")

        self.leftfoot = self.supervisor.getFromDef("ROBOT.LeftFoot")
        self.rightfoot = self.supervisor.getFromDef("ROBOT.RightFoot")
        self.GetLeftFootPosition = self.leftfoot.getField("translation")
        self.GetRightFootPosition = self.rightfoot.getField("translation")
        
        self.motorNames = RobotFunc.getMotorNames()
        self.motorList, self.motorLimitList = RobotFunc.getMotors(robot=self.supervisor, num=None)

        self.positionSensors = RobotFunc.getPositionSensors(robot=self.supervisor, timestep=self.timestep, num=None)

    # def reset(self):
    #     self.phase = 0
    # 
    #     # 初期状態にセット
    #     cnt = 0
    #     while self.supervisor.step(self.timestep) != -1:
    #         init_act = self.init_action[cnt]
    #         self.apply_action(init_act)
    # 
    #         cnt += 1
    #         if cnt >= self.init_action.shape[0]:
    #             break
    # 
    # 
    #     return self.get_observations()

    def reset(self):
        self.phase = 0
        self.counter = 0
        
        init_pos = [7.038719910171904e-07, -1.9018715709164663e-07, 0.3056947598600072]
        init_rotation = [-0.0014797995121209266, 0.000984208412616663, 0.9999984207623552, 1.5627820795176646]

        # 初期状態にセット
        cnt = 0
        while self.supervisor.step(self.timestep) != -1:
            init_act = self.init_action[cnt]
            self.apply_action(action=init_act, init_flg=True)

            cnt += 1
            if cnt >= self.init_action.shape[0]:
                break

        # TODO:モータが目標角度まで動き終わったかをチェックする

        self.translation_field.setSFVec3f(list(init_pos))        
        self.rotation_field.setSFRotation(list(init_rotation))

        # 初期状態にセット
        cnt = 0
        while self.supervisor.step(self.timestep) != -1:
            cnt += 1
            if cnt > 10:
                break

        self.supervisor.simulationResetPhysics()

        return self.get_observations()

    def step(self, action):
        self.apply_action(action)

        while self.supervisor.step(self.timestep) != -1:
            break

        self.cnt += 1
        self.phase += 1

        if self.phase >= len(self.ref_action):
            self.phase = 0


        next_observations = self.get_observations()
        rewards = self.get_reward(action)
        done = self.is_done()
        info = self.get_info()

        self.com_pos_his = np.array(self.com_pos)
        return next_observations, rewards, done, info

    def apply_action(self, action, init_flg=False):
        if init_flg:
            for i, ac in enumerate(action):
                ac = float(ac)
                self.motorList[i].setPosition(ac)

        else:
            ref_act = self.get_ref_action()

            target = ref_act 

            for i in range(len(self.motorList)):
                ac = float(target[i])
                self.motorList[i].setPosition(ac)

    def get_reward(self, action):
        return 0

    def is_done(self):
        return False

    def get_info(self):
        return {}

    def get_observations(self):
        obs = self.get_sim_observations()

        return obs

    def get_sim_observations(self):
        observation = []
        motor_position = []

        # x,y,z pos; x, y, z, w quaternion
        base_position = self.translation_field.getSFVec3f()
        base_orientation = self.rotation_field.getSFRotation()
        orientation = GetQuaternionFromAxisAngle(base_orientation)
        base_orientation = GetEulerFromOrientation(orientation[3], orientation[0], orientation[1], orientation[2])
        observation.append(base_position[0])
        observation.append(base_position[1])
        observation.append(base_position[2])
        observation.append(base_orientation[0])
        observation.append(base_orientation[1])
        observation.append(base_orientation[2])

        print(base_position)
        # linear velocity--0,1,2; angular velocity--3,4,5
        base_velocity = self.robot.getVelocity()
        observation.append(base_velocity[0])
        observation.append(base_velocity[1])
        observation.append(base_velocity[2])
        observation.append(base_velocity[3])
        observation.append(base_velocity[4])
        observation.append(base_velocity[5])

        self.com_pos = self.robot.getCenterOfMass()
        observation.append(self.com_pos[0])
        observation.append(self.com_pos[1])
        observation.append(self.com_pos[2])

        motor_position = RobotFunc.getValuePositionSensor(self.positionSensors)
        observation.extend(motor_position)

        return np.array(observation)

    def get_ref_obs(self):
        ref_obs = self.trajectory[self.phase]

        return ref_obs

    def get_ref_action(self):
        ref_act = self.ref_action[self.phase]
        return ref_act


    def close(self):
        pass


if __name__ == '__main__':
    env = robotisImitationEnv(timestep=None)
