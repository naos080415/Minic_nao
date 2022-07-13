import numpy as np
import math

class RobotFunc(object):
    def __init__(self):
        uncontrolled_motorNames = ["LPhalanx1", "LPhalanx2", "LPhalanx3", "LPhalanx4",
                                   "LPhalanx5", "LPhalanx6", "LPhalanx7", "LPhalanx8", 
                                   "RPhalanx1", "RPhalanx2", "RPhalanx3", "RPhalanx4",
                                   "RPhalanx5", "RPhalanx6", "RPhalanx7", "RPhalanx8"]


    def getMotorNames(name_type='all'):
        if name_type == 'all':
            return ["HeadPitch", "HeadYaw", "LAnklePitch", "LAnkleRoll",
                    "LElbowRoll", "LElbowYaw", "LHipPitch", "LHipRoll",
                    "LHipYawPitch", "LKneePitch", "LShoulderPitch", "LShoulderRoll",
                    "LWristYaw", "RAnklePitch", "RAnkleRoll", "RElbowRoll",
                    "RElbowYaw", "RHipPitch", "RHipRoll", "RHipYawPitch",
                    "RKneePitch", "RShoulderPitch", "RShoulderRoll", "RWristYaw"]

    def setMotorLimit(motorName):
        lower = -float('inf')
        upper = float('inf')

        return [lower, upper]

    def getMotors(robot, num):
        """
        Get 6 (legs-all) or 4 (leg)  motors from the robot model.
        """
        # Get the motors names
        motorNames = RobotFunc.getMotorNames('all')

        motorList = []
        motorlimitList = []
        for motorName in motorNames:
            motor = robot.getMotor(motorName)	 # Get the motor handle
            # motor.setPosition(float('inf'))  # Set starting position
            # motor.setVelocity(0.0)  # Zero out starting velocity
            motorList.append(motor)  # Append motor to motorList

            motorlimitList.append(RobotFunc.setMotorLimit(motorName))

        return motorList, motorlimitList

    def getPositionSensors(robot, timestep, num):
        # Get the motors names
        motorNames = RobotFunc.getMotorNames('all')

        positionSensorList = []
        for motor_name in motorNames:
            positionSensorName = motor_name + 'S'
            positionSensor = robot.getDevice(positionSensorName)
            positionSensor.enable(timestep)
            positionSensorList.append(positionSensor)
        return positionSensorList

    def getValuePositionSensor(positionSensorList):
        psValue = []
        for i in positionSensorList:
            psValue.append(i.getValue())

        return psValue

    def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
        """
        Normalizes value to a specified new range by supplying the current range.
        :param value: value to be normalized
        :type value: float
        :param minVal: value's min value, value ∈ [minVal, maxVal]
        :type minVal: float
        :param maxVal: value's max value, value ∈ [minVal, maxVal]
        :type maxVal: float
        :param newMin: normalized range min value
        :type newMin: float
        :param newMax: normalized range max value
        :type newMax: float
        :param clip: whether to clip normalized value to new range or not, defaults to False
        :type clip: bool, optional
        :return: normalized value ∈ [newMin, newMax]
        :rtype: float
        """

        value = float(value)
        minVal = float(minVal)
        maxVal = float(maxVal)
        newMin = float(newMin)
        newMax = float(newMax)

        if clip:
            return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
        else:
            return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax
