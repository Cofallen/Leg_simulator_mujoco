import numpy as np

class RobotSensor:
    def __init__(self, model, data):
        """
        model: mujoco.MjModel
        data: mujoco.MjData
        """
        self.model = model
        self.data = data

    # ---------------------------
    # IMU
    # ---------------------------
    def get_orientation_quat(self):
        """Return quaternion [w, x, y, z]"""
        return self.data.sensor("orientation").data.copy()

    def get_acc(self):
        """Return linear acceleration (m/s^2)"""
        return self.data.sensor("acc").data.copy()

    def get_gyro(self):
        """Return angular velocity (rad/s)"""
        return self.data.sensor("gyro").data.copy()

    # ---------------------------
    # Joint positions
    # ---------------------------
    def get_joint_positions(self):
        """Return all joint positions as dict"""
        return {
            "joint_l1_L": self.data.sensor("joint_l1_L").data[0],
            "joint_l6_L": self.data.sensor("joint_l6_L").data[0],
            "joint_wheel_L": self.data.sensor("joint_wheel_L").data[0],
            "joint_l1_R": self.data.sensor("joint_l1_R").data[0],
            "joint_l6_R": self.data.sensor("joint_l6_R").data[0],
            "joint_wheel_R": self.data.sensor("joint_wheel_R").data[0],
        }

    # ---------------------------
    # Combined state (recommended)
    # ---------------------------
    def get_state(self):
        """Return full robot state"""
        return {
            "quat": self.get_orientation_quat(),
            "acc": self.get_acc(),
            "gyro": self.get_gyro(),
            "joints": self.get_joint_positions(),
            "euler": self.quat_to_euler(self.get_orientation_quat()),
        }

    # ---------------------------
    # Optional: quaternion → euler
    # ---------------------------
    def quat_to_euler(self, quat):
        """Convert quaternion [w,x,y,z] → roll, pitch, yaw"""
        w, x, y, z = quat

        # roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])