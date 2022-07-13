import sys
import numpy as np

from controller import Supervisor, Node
import gym
from gym.spaces import Box
from gym.utils import seeding
import math
import random

from utils.RobotUtils import RobotFunc
from utils.MathUtils import GetQuaternionFromAxisAngle, GetEulerFromOrientation, GetQuaternionFromEuler, GetAxisAngleFromQuaternion

class robotisImitationEnv(gym.Env):
    def __init__(self, timestep=None, trajectorypath="./ref-Data/trajectory.csv", init_actionpath='./ref-Data/initAct.csv', ref_actionpath="./ref-Data/refAct.csv"):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here. 
        For the observation we are using a time window of 5, to store the last 5 episode observations.
        
        ref-obs
            Num	Observation                                  Min           Max
            0   Robot position x-axis                       -inf          +inf     
            1   Robot position y-axis                       -inf          +inf
            2   Robot position z-axis                       -inf          +inf
            3   Robot roll angle                            -inf          +inf
            4   Robot pitch angle                           -inf          +inf
            5   Robot yaw angle                             -inf          +inf
            6   Robot linear velocity x-axis                -inf          +inf
            7   Robot linear velocity y-axis                -inf          +inf
            8   Robot linear velocity z-axis                -inf          +inf
            9   Robot angular velocity x-axis               -inf          +inf
           10   Robot angular velocity y-axis               -inf          +inf
           11   Robot angular velocity z-axis               -inf          +inf
           12   Robot center-of-mass position x-axis        -inf          +inf     
           13   Robot center-of-mass position y-axis        -inf          +inf     
           14   Robot center-of-mass position z-axis        -inf          +inf     
           15   Motor position HeadPitch            
           16   Motor position HeadYaw
           17   Motor position LAnklePitch
           18   Motor position LAnkleRoll
           15   Motor position LElbowRoll
           15   Motor position LElbowYaw
           15   Motor position LHipPitch
           15   Motor position LHipRoll
           15   Motor position LHipYawPitch
           15   Motor position LKneePitch
           15   Motor position LShoulderPitch
           15   Motor position LShoulderRoll
           15   Motor position LWristYaw
           15   Motor position RAnklePitch
           15   Motor position RAnkleRoll
           15   Motor position RElbowRoll
           15   Motor position RElbowYaw
           15   Motor position RHipPitch
           15   Motor position RHipRoll
           15   Motor position RHipYawPitch
           15   Motor position RKneePitch
           15   Motor position RShoulderPitch
           15   Motor position RShoulderRoll
           15   Motor position RWristYaw

            # 1   Robot position y-axis               -inf          +inf
            # 2   Robot position z-axis               -inf          +inf
            # 4	LeftAnkle                           -inf          +inf
            # 5	LeftCrus                            -inf          +inf
            # 6	LeftFemur                           -inf          +inf
            # 7	RightAnkle                          -inf          +inf
            # 8	RightCrus                           -inf          +inf
            # 9	RightFemur                          -inf          +inf

        Action(Forward/Backward walking task):
            Num	BodyPost      Min       Max      Desc
            # 0	LeftAnkle   -2.356     +2.356    Set the motor position from -2.356 to +2.356
            # 1	LeftCrus    -2.356     +2.356    Set the motor position from -2.356 to +2.356
            # 2	LeftFemur   -2.356     +2.356    Set the motor position from -2.356 to +2.356
            # 3	RightAnkle  -2.356     +2.356    Set the motor position from -2.356 to +2.356
            # 4	RightCrus   -2.356     +2.356    Set the motor position from -2.356 to +2.356
            # 5	RightFemur  -2.356     +2.356    Set the motor position from -2.356 to +2.356
        Action(Left/Right moving task):
            Num	BodyPost      Min       Max      Desc
            0	LeftAnkle    -1.0      +1.0      Set the motor position from -2.356 to +2.356
            1	LeftCrus     -1.0      +1.0      Set the motor position from -2.356 to +2.356
            2	LeftFemur    -1.0      +1.0      Set the motor position from -2.356 to +2.356
            3	RightAnkle   -1.0      +1.0      Set the motor position from -2.356 to +2.356
            4	RightCrus    -1.0      +1.0      Set the motor position from -2.356 to +2.356
            5	RightFemur   -1.0      +1.0      Set the motor position from -2.356 to +2.356

        Reward(Forward/Backward walking task):

        Reward(Left/Right moving task):

        Starting State:
            [0, 0, ..., 0]

        Episode Termination:
            Robot y axis smaller than 0.50 cm
            Robot walked more that 15 m
        """
        self.supervisor = Supervisor()

        if timestep is None:
            self.timestep = int(self.supervisor.getBasicTimeStep())
        else:
            self.timestep = timestep

        self.trajectory = np.loadtxt(trajectorypath, delimiter=',', dtype=np.float32)
        self.init_action = np.loadtxt(init_actionpath, delimiter=',', dtype=np.float32)
        self.ref_action = np.loadtxt(ref_actionpath, delimiter=',', dtype=np.float32)

        self.setup_agent()      # motor, sensor setting

        self.numObs, self.numAct = 39 * 2 + 1, 10
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
        self.counter = 0

        # logger用の変数
        self.logging_reward = []
        self.logging_robotPos = [0] * 3


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

        rl_controlled_motor = ['LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
                            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']

        self.joint_index = [] 
        for i, name in enumerate(self.motorNames):
            if name in rl_controlled_motor:
                self.joint_index.append(i)

        self.motor_position = np.zeros(len(self.motorList))

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

        self.phase += 1
        if self.phase >= self.ref_action.shape[0]:
            self.phase = 0
            self.counter += 1

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
            dt = self.timestep / 1000

            for i, ac in enumerate(action):
                dtheta = self.motorList[self.joint_index[i]].getMaxVelocity() * ac * dt
                ac = float(ref_act[self.joint_index[i]] + dtheta)
                self.motorList[self.joint_index[i]].setPosition(ac)


    def get_reward(self, action):
        sim_obs = self.get_observations()
        ref_obs = self.get_ref_obs() 
        joint_penalty = 0

        for i in range(len(self.joint_index)):
            error = 1/len(self.joint_index) * (ref_obs[15 + self.joint_index[i]]-sim_obs[15 + self.joint_index[i]])**2
            joint_penalty += error*30 

        com_penalty = (ref_obs[14] - sim_obs[14])**2 + (0 - sim_obs[12])**2     # TODO
        orientation_penalty = (ref_obs[3] - sim_obs[3])**2 + (ref_obs[4] - sim_obs[4])**2 + (ref_obs[5] - sim_obs[5])**2
        imitation_reward = 0.6*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-orientation_penalty)
        #sim_reward = 50 * min((self.com_pos[2] - self.com_pos_his[2]),0.02)

        vel = self.com_pos[1] - self.com_pos_his[1]
        vel_error = 100000*(vel - 0.01)**2
        sim_reward = np.exp(-vel_error)

        AxisAngle = self.rotation_field.getSFRotation()
        orientation = GetQuaternionFromAxisAngle(AxisAngle)
        euler = GetEulerFromOrientation(orientation[3], orientation[0], orientation[1], orientation[2])
        if (math.degrees(euler[1])>20 or math.degrees(euler[1])<-20):
            sim_reward -=1

        self.logging_reward = [0.6*np.exp(-joint_penalty), 0.3*np.exp(-com_penalty), 0.1*np.exp(-orientation_penalty), sim_reward]
        """
        sim_reward+=np.exp(-abs(euler[1])/30)-1
        """
        total_reward = 0.62*imitation_reward+0.38*sim_reward
        # total_reward = 1*imitation_reward+0*sim_reward
        """
        if self.is_done():
            total_reward = 0
        else:
            total_reward += 0.2
        """
        return total_reward

    def is_done(self):
        done = False
        base_position = self.translation_field.getSFVec3f()
        robot_position = self.robot.getPosition()

        AxisAngle = self.rotation_field.getSFRotation()
        orientation = GetQuaternionFromAxisAngle(AxisAngle)
        euler = GetEulerFromOrientation(orientation[3], orientation[0], orientation[1], orientation[2])

        if base_position[2] < 0.28:
            done = True
        if (math.degrees(euler[0]) > 45 or math.degrees(euler[0]) < -45):
            done = True
        elif (math.degrees(euler[1]) > 45 or math.degrees(euler[1]) < -45):
            done = True
        
        return done

    def get_info(self):
        return {}

    def get_observations(self):
        obs = self.get_sim_observations()
        ref_obs = self.get_ref_obs() 

        phase = np.array([self.phase])

        return np.concatenate([obs, ref_obs, phase])

    def get_sim_observations(self):
        observation = []

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

        self.motor_position = RobotFunc.getValuePositionSensor(self.positionSensors)
        observation.extend(self.motor_position)
        
        self.logging_robotPos = base_position

        return np.array(observation)

    def get_ref_obs(self):
        if self.counter == 0:
            ref_obs = self.trajectory[self.phase]
        elif self.counter == 1:
            ref_obs = self.trajectory[self.phase + self.ref_action.shape[0]]
        else:
            ref_obs = np.copy(self.trajectory[self.phase + self.ref_action.shape[0]*2])
            ref_obs[1] += (self.trajectory[self.ref_action.shape[0]*3-1, 1]- self.trajectory[self.ref_action.shape[0]*2-1, 1])* (self.counter-2)
            ref_obs[13] += (self.trajectory[self.ref_action.shape[0]*3-1, 13]- self.trajectory[self.ref_action.shape[0]*2-1, 13])* (self.counter-2)
        return ref_obs

    def get_ref_action(self):
        ref_act = self.ref_action[self.phase]
        return ref_act


    def close(self):
        pass


if __name__ == '__main__':
    env = robotisImitationEnv(timestep=None)
