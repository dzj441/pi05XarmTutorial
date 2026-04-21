from xarm.wrapper import XArmAPI

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial.transform import Rotation as R


XARM_HOME_VALUES = [
    366,
    23,
    280,
    180,
    0,
    0,
    1.0,
]


class Xarm:
    def __init__(self, ip):
        # is_radian: 设置用角度
        self._controller = XArmAPI(ip, is_radian=False)
        self._init_controller()

    def _init_controller(self):
        # error and warn call back
        self._controller.register_error_warn_changed_callback(
            self.hangle_err_warn_changed
        )
        self._controller.connect()
        # enable motion
        self._controller.motion_enable(enable=True)
        # set mode: position control mode
        self._controller.set_mode(0)
        # set state: sport state
        self._controller.set_state(state=0)

        # for gripper
        self._controller.set_gripper_mode(0)
        self._controller.set_gripper_enable(True)
        self._max_gripper_value = 850
        # defalt 0
        # self._min_gripper_value = 0

    @property
    def name(self):
        return "xarm"

    @property
    def recorder_functions(self):
        return {
            "joint_states": self.get_joint_state,
            "cartesian_position": self.get_cartesian_position,
        }

    def if_shutdown(self):
        code, state = self._controller.get_state()
        return code != 0

    def get_joint_state(self) -> np.ndarray:
        # (code, [position, velocity, effort])
        return np.array(self._controller.get_joint_states()[1], dtype=np.float32)

    def get_joint_position(self) -> np.ndarray:
        # 暂时用不到关节的位置
        pass
        # return np.array(self._controller.get_position()[1], dtype=np.float32)

    def get_cartesian_position(self) -> np.ndarray:
        return np.array(self._controller.get_position()[1], dtype=np.float32)
    
    def get_state(self) -> np.ndarray:
        return np.array(self._controller.get_position()[1], dtype=np.float32)

    # 齐次坐标系下的变换矩阵
    def get_rotation_matrix(self) -> np.ndarray:
        # [x,y,z,roll,pitch,yaw]
        cart = self.get_cartesian_position()

        # 外旋，绕固定坐标轴转动
        rotation = R.from_euler("xyz", cart[3:], degrees=True).as_matrix()
        translation = np.array(cart[:3])
        return np.block([[rotation, translation[:, np.newaxis]], [0, 0, 0, 1]])

    def get_joint_velocity(self) -> np.ndarray:
        return np.array(self._controller.get_joint_states()[1][1], dtype=np.float32)

    def get_joint_torque(self) -> np.ndarray:
        return np.array(self._controller.get_joints_torque()[1], dtype=np.float32)

    def get_gripper_state(self) -> int:
        return 0 if (self._controller.get_gripper_position()[1] < 800) else 1

    def home(self):
        self.move_coords(XARM_HOME_VALUES[0:6])
        self.move_gripper_percentage(XARM_HOME_VALUES[-1])

    def move(self, input_angles):
        self._controller.set_servo_angle(angle=input_angles)

    def move_coords(self, input_coords, speed=100, wait=True, wait_motion=True):
        # input_coords: [x,y,z,roll,pitch,yaw]
        x, y, z, roll, pitch, yaw = input_coords[:6]
        self._controller.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            wait=wait,
            speed=speed,
            wait_motion=wait_motion,
        )

    def move_matrix(self, input_matrix, speed=100, wait=True, wait_motion=True):
        t = input_matrix[:3, 3]
        R = Rotation.from_matrix(input_matrix[:3, :3]).as_euler("xyz", degrees=True)
        cart = np.concatenate([t, R], axis=0)

        self.move_coords(
            cart,
            speed=speed,
            wait=wait,
            wait_motion=wait_motion,
        )

    def move_gripper(self, position, wait=False, speed=5000, wait_motion=False):
        self._controller.set_gripper_position(
            position,
            wait=wait,
            speed=speed,
            wait_motion=wait_motion,
        )

    def move_gripper_percentage(
        self, percentage, wait=False, speed=5000, wait_motion=False
    ):
        self._controller.set_gripper_position(
            percentage * self._max_gripper_value,
            wait=wait,
            speed=speed,
            wait_motion=wait_motion,
        )

    def stop(self):
        self._controller.disconnect()

    def hangle_err_warn_changed(self, item):
        print(
            "Xarm ErrorCode: {}, WarnCode: {}".format(
                item["error_code"], item["warn_code"]
            )
        )
        # TODO：Do different processing according to the error code

    def clear_cache(self):
        self._controller.set_state(4)
        self._controller.set_state(0)