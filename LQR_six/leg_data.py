import numpy as np
from mymath import Discreteness, PID_control


class Leg:
    def __init__(self, dt):
        # -------- discreteness --------
        self.disc_L = Discreteness(dt)
        self.disc_dL = Discreteness(dt)
        self.disc_theta = Discreteness(dt)
        self.disc_dtheta = Discreteness(dt)

        # -------- state --------
        self.state = {
            "theta": 0.0,
            "dtheta": 0.0,
            "ddtheta": 0.0,
            "s": 0.0,
            "dot_s": 0.0,
            "phi": 0.0,
            "dphi": 0.0,
            "delta": 0.0,
        }

        # -------- target --------
        self.target = {
            "theta": 0.0,
            "dtheta": 0.0,
            "s": 0.0,
            "dot_s": 0.0,
            "phi": 0.0,
            "dphi": 0.0,
            "l0": 0.25,
            "roll": 0.0,
            "yaw": 0.0,
            "d2theta": 0.0
        }

        # -------- VMC --------
        self.vmc = {
            "phi0": 0.0,
            "phi1": 0.0,
            "phi2": 0.0,
            "phi3": 0.0,
            "phi4": 0.0,
            "L0": 0.0,
            "L0_dot": 0.0,
            "L0_ddot": 0.0,
            "J": np.zeros((2, 2)),
        }

        # -------- LQR --------
        self.LQR = {
            "K": np.zeros(12),
            "T_w": 0.0,
            "T_p": 0.0,
            "F_0": 0.0,
            "torque_leg": np.zeros(2),
            "torque_wheel": 0.0,
            "Fn": 0.0,
        }

        # -------- PID --------
        self.pid_F = PID_control(4000, 1.0, 10000, 0.0)
        self.pid_roll = PID_control(1000, 0.0, 3000, 0.0)

        # -------- limit --------
        self.limit = {
            "T_max": 20.0,
            "W_max": 10.0,
        }