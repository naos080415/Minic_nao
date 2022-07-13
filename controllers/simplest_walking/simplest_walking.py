from imitation_Nao_env import robotisImitationEnv
import numpy as np

from utils.RobotUtils import RobotFunc

# init_act : Forwards50.motion のPose1 ~ Pose33
init_act = [
    # LHipRoll,LHipPitch,LKneePitch,LAnklePitch,LAnkleRoll,RHipRoll,RHipPitch,RKneePitch,RAnklePitch,RAnkleRoll
    [0,-0.611,1.222,-0.611,0,0,-0.611,1.222,-0.611,0],
    [0,-0.608,1.215,-0.608,0,0,-0.608,1.215,-0.608,0],
    [0,-0.572,1.144,-0.572,0,0,-0.572,1.144,-0.572,0],
    [0,-0.543,1.086,-0.543,0,0,-0.543,1.086,-0.543,0],
    [0,-0.525,1.05,-0.525,0,0,-0.525,1.05,-0.525,0],
    [0,-0.525,1.05,-0.525,0,0,-0.525,1.05,-0.525,0],
    [0,-0.525,1.05,-0.525,0,0,-0.525,1.05,-0.525,0],
    [-0.001,-0.525,1.05,-0.525,0.001,-0.001,-0.525,1.05,-0.525,0.001],
    [-0.003,-0.524,1.05,-0.525,0.003,-0.003,-0.524,1.05,-0.525,0.003],
    [-0.004,-0.524,1.05,-0.525,0.004,-0.004,-0.524,1.05,-0.525,0.004],
    [-0.008,-0.524,1.049,-0.526,0.008,-0.008,-0.524,1.049,-0.526,0.008],
    [-0.01,-0.523,1.049,-0.526,0.01,-0.01,-0.523,1.049,-0.526,0.01],
    [-0.014,-0.522,1.049,-0.527,0.014,-0.014,-0.522,1.049,-0.527,0.014],
    [-0.021,-0.521,1.048,-0.527,0.021,-0.021,-0.521,1.048,-0.527,0.021],
    [-0.027,-0.52,1.048,-0.528,0.027,-0.027,-0.52,1.048,-0.528,0.027],
    [-0.034,-0.518,1.047,-0.529,0.034,-0.034,-0.518,1.047,-0.529,0.034],
    [-0.044,-0.516,1.045,-0.53,0.044,-0.044,-0.516,1.045,-0.53,0.044],
    [-0.063,-0.51,1.041,-0.531,0.063,-0.063,-0.51,1.041,-0.531,0.063],
    [-0.08,-0.505,1.036,-0.531,0.08,-0.08,-0.505,1.036,-0.531,0.08],
    [-0.101,-0.498,1.029,-0.531,0.101,-0.101,-0.498,1.029,-0.531,0.101],
    [-0.136,-0.482,1.013,-0.531,0.136,-0.136,-0.482,1.013,-0.531,0.136],
    [-0.157,-0.468,0.999,-0.531,0.157,-0.157,-0.468,0.999,-0.531,0.157],
    [-0.173,-0.454,0.987,-0.533,0.173,-0.174,-0.459,0.996,-0.537,0.174],
    [-0.15,-0.44,0.984,-0.544,0.19,-0.197,-0.52,1.119,-0.599,0.197],
    [-0.144,-0.442,0.999,-0.557,0.196,-0.211,-0.605,1.244,-0.639,0.211],
    [-0.146,-0.452,1.033,-0.582,0.198,-0.223,-0.743,1.374,-0.631,0.223],
    [-0.142,-0.454,1.051,-0.597,0.195,-0.219,-0.803,1.378,-0.575,0.219],
    [-0.134,-0.447,1.056,-0.609,0.187,-0.205,-0.815,1.313,-0.498,0.205],
    [-0.121,-0.427,1.045,-0.618,0.174,-0.184,-0.777,1.202,-0.425,0.184],
    [-0.128,-0.379,1.011,-0.632,0.145,-0.147,-0.678,1.037,-0.36,0.147],
    [-0.119,-0.344,0.99,-0.646,0.119,-0.119,-0.634,1,-0.366,0.119],
    [-0.065,-0.293,0.97,-0.677,0.065,-0.065,-0.604,1.022,-0.417,0.065],
    [-0.019,-0.253,0.948,-0.696,0.019,-0.019,-0.579,1.033,-0.453,0.019]]

# ref_act : Forwards50.motion のPose59 ~ Pose84
ref_act = [
    # LHipRoll,LHipPitch,LKneePitch,LAnklePitch,LAnkleRoll,RHipRoll,RHipPitch,RKneePitch,RAnklePitch,RAnkleRoll
    [0.007,-0.233,0.935,-0.702,-0.007,0.007,-0.566,1.034,-0.468,-0.007],
    [0.054,-0.181,0.893,-0.712,-0.054,0.054,-0.53,1.029,-0.499,-0.054],
    [0.095,-0.126,0.839,-0.714,-0.095,0.095,-0.49,1.014,-0.524,-0.095],
    [0.141,-0.11,0.869,-0.759,-0.141,0.121,-0.437,0.986,-0.55,-0.138],
    [0.166,-0.194,1.018,-0.824,-0.166,0.105,-0.417,0.982,-0.565,-0.157],
    [0.194,-0.422,1.275,-0.852,-0.194,0.122,-0.415,0.998,-0.583,-0.174],
    [0.202,-0.591,1.372,-0.781,-0.202,0.126,-0.426,1.02,-0.593,-0.179],
    [0.198,-0.716,1.369,-0.653,-0.198,0.125,-0.437,1.04,-0.603,-0.177],
    [0.184,-0.764,1.278,-0.515,-0.184,0.118,-0.436,1.047,-0.612,-0.17],
    [0.152,-0.712,1.087,-0.375,-0.152,0.109,-0.404,1.029,-0.625,-0.149],
    [0.128,-0.657,1.009,-0.352,-0.128,0.123,-0.37,1.006,-0.636,-0.128],
    [0.099,-0.629,1.006,-0.378,-0.099,0.099,-0.334,0.988,-0.654,-0.099],
    [0.04,-0.596,1.028,-0.432,-0.04,0.04,-0.279,0.964,-0.685,-0.04],
    [-0.007,-0.566,1.034,-0.468,0.007,-0.007,-0.233,0.935,-0.702,0.007],
    [-0.055,-0.53,1.029,-0.499,0.055,-0.055,-0.181,0.893,-0.712,0.055],
    [-0.111,-0.471,1.002,-0.531,0.111,-0.112,-0.104,0.821,-0.716,0.112],
    [-0.121,-0.437,0.986,-0.549,0.138,-0.141,-0.11,0.869,-0.759,0.141],
    [-0.112,-0.413,0.985,-0.571,0.164,-0.177,-0.26,1.108,-0.848,0.177],
    [-0.122,-0.415,0.998,-0.583,0.174,-0.194,-0.422,1.275,-0.852,0.194],
    [-0.126,-0.426,1.019,-0.593,0.179,-0.202,-0.591,1.372,-0.781,0.202],
    [-0.122,-0.438,1.046,-0.608,0.174,-0.192,-0.751,1.332,-0.582,0.192],
    [-0.112,-0.429,1.045,-0.616,0.164,-0.174,-0.759,1.214,-0.456,0.174],
    [-0.109,-0.404,1.029,-0.624,0.149,-0.153,-0.712,1.087,-0.375,0.153],
    [-0.114,-0.352,0.996,-0.644,0.115,-0.115,-0.64,1,-0.361,0.115],
    [-0.081,-0.317,0.981,-0.665,0.081,-0.081,-0.619,1.014,-0.395,0.081],
    [-0.04,-0.279,0.965,-0.685,0.04,-0.04,-0.596,1.028,-0.432,0.04]]

def Converting_initAct_to_CSV(filename='./ref-Data/initAct.csv'):
    global init_act
    motorNames = RobotFunc.getMotorNames(name_type='all')
    controlled_motorNames = ["LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch",
                             "LAnkleRoll", "RHipRoll", "RHipPitch", "RKneePitch",
                             "RAnklePitch", "RAnkleRoll"]

    act = []

    for ref_act_step in init_act:
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



def Converting_refAct_to_CSV(filename='./ref-Data/refAct.csv'):
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


def Converting_refObs_to_CSV(trajectory, filename='./ref-Data/trajectory.csv'):
    np.savetxt(filename, np.array(trajectory), delimiter=',')


if __name__ == '__main__':
    Converting_initAct_to_CSV()    # actionの模倣元データを作成する
    Converting_refAct_to_CSV()    # actionの模倣元データを作成する

    env = robotisImitationEnv(timestep=None)

    max_steps = len(ref_act) * 3      # 1周期が26周期

    init_obs = env.reset()

    trajectory = []
    for i in range(max_steps):
        action = [0] * 10
        obs, reward, done, _ = env.step(action)
        trajectory.append(list(obs))

    Converting_refObs_to_CSV(trajectory=trajectory, filename='./ref-Data/trajectory.csv')   # Stateの模倣元データを作成する
