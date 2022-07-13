import numpy as np

from utils.RobotUtils import RobotFunc

# ref_act : Forwards50.motion のPose46 ~ Pose71
ref_act = [
    # LHipRoll LHipPitch LKneePitch LAnklePitch LAnkleRoll RHipRoll RHipPitch RKneePitch RAnklePitch RAnkleRoll
    [-0.007, -0.566, 1.034, -0.468, 0.007, - 0.007, -0.233, 0.935, -0.702, 0.007],
    [-0.055, -0.53, 1.029, -0.499, 0.055, - 0.055, -0.181, 0.892, -0.712, 0.055],
    [-0.095, -0.49, 1.014, -0.524, 0.095, - 0.095, -0.126, 0.839, -0.714, 0.095],
    [-0.121, -0.452, 0.993, -0.541, 0.126, - 0.127, -0.097, 0.83, -0.732, 0.127],
    [-0.105, -0.417, 0.982, -0.565, 0.157, - 0.166, -0.194, 1.018, -0.824, 0.166],
    [-0.118, -0.413, 0.99, -0.578, 0.17, - 0.187, -0.338, 1.197, -0.859, 0.187],
    [-0.125, -0.42, 1.008, -0.588, 0.177, - 0.199, -0.509, 1.335, -0.826, 0.199],
    [-0.125, -0.436, 1.039, -0.603, 0.177, - 0.198, -0.716, 1.369, -0.653, 0.198],
    [-0.118, -0.435, 1.047, -0.612, 0.17, - 0.184, -0.764, 1.278, -0.515, 0.184],
    [-0.105, -0.418, 1.038, -0.62, 0.157, - 0.164, -0.739, 1.148, -0.409, 0.164],
    [-0.124, -0.37, 1.006, -0.636, 0.128, - 0.128, -0.657, 1.009, -0.352, 0.128],
    [-0.099, -0.334, 0.988, -0.654, 0.099, - 0.099, -0.628, 1.006, -0.378, 0.099],
    [-0.04, -0.279, 0.964, -0.685, 0.04, - 0.04, -0.596, 1.028, -0.432, 0.04],
    [0.007, -0.233, 0.935, -0.702, -0.007, 0.007, -0.566, 1.034, -0.468, -0.007],
    [0.054, -0.181, 0.893, -0.712, -0.054, 0.054, -0.53, 1.029, -0.499, -0.054],
    [0.095, -0.126, 0.839, -0.714, -0.095, 0.095, -0.49, 1.014, -0.524, -0.095],
    [0.141, -0.11, 0.869, -0.759, -0.141, 0.121, -0.437, 0.986, -0.55, -0.138],
    [0.166, -0.194, 1.018, -0.824, -0.166, 0.105, -0.417, 0.982, -0.565, -0.157],
    [0.194, -0.422, 1.275, -0.852, -0.194, 0.122, -0.415, 0.998, -0.583, -0.174],
    [0.202, -0.591, 1.372, -0.781, -0.202, 0.126, -0.426, 1.02, -0.593, -0.179],
    [0.198, -0.716, 1.369, -0.653, -0.198, 0.125, -0.437, 1.04, -0.603, -0.177],
    [0.184, -0.764, 1.278, -0.515, -0.184, 0.118, -0.436, 1.047, -0.612, -0.17],
    [0.152, -0.712, 1.087, -0.375, -0.152, 0.109, -0.404, 1.029, -0.625, -0.149],
    [0.128, -0.657, 1.009, -0.352, -0.128, 0.123, -0.37, 1.006, -0.636, -0.128],
    [0.099, -0.629, 1.006, -0.378, -0.099, 0.099, -0.334, 0.988, -0.654, -0.099],
    [0.04, -0.596, 1.028, -0.432, -0.04, 0.04, -0.279, 0.964, -0.685, -0.04]]


def Converting_refAct_to_CSV(filename='./ref-Data/Act.csv'):
    global ref_act
    motorNames = RobotFunc.getMotorNames(name_type='all')
    controlled_motorNames = ["LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch",
                             "LAnkleRoll", "RHipRoll", "RHipPitch", "RKneePitch",
                             "RAnklePitch", "RAnkleRoll"]

    act = []

    for ref_act_step in ref_act:
        act_tmp = []
        for motorName in motorNames:
            if motorName in controlled_motorNames:
                index = controlled_motorNames.index(motorName)
                act_tmp.append(ref_act_step[index])
            else:
                # ロボットの上半身の腕の位置などを初期化する
                if motorName in ["LElbowRoll", "LElbowYaw"]:
                    act_tmp.append(-1.396)
                elif motorName in ["RElbowRoll", "RElbowYaw"]:
                    act_tmp.append(1.396)
                elif motorName in ["LShoulderPitch", "RShoulderPitch"]:
                    act_tmp.append(1.745)
                elif motorName in ["LShoulderRoll"]:
                    act_tmp.append(0.3450)
                elif motorName in ["RShoulderRoll"]:
                    act_tmp.append(-0.3450)
                else:
                    act_tmp.append(0.0)
        act.append(act_tmp)

    act = np.array(act)
    np.savetxt(filename, act, delimiter=',')


def main():
    Converting_refAct_to_CSV()


if __name__ == '__main__':
    main()
