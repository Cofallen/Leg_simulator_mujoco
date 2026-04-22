import numpy as np
from mymath import PID_control

# -------- 常量（按你实际填）--------
MASS_BODY = 10.0
GRAVITY = 9.81

MAX_TORQUE_LEG_T = 20.0
MAX_TORQUE_LEG_W = 100.0

import vofa

class LQRController:
    def __init__(self):
        # -------- LQR增益（你需要填真实K）--------
        self.K = np.array([
    [-61.24193654, -4.18497996, -21.97446496, -20.76311853, 24.85036087, 4.03477100],
    [16.06953727, 0.71109186, 4.13798137, 3.18068175, 75.32333508, 4.01482828],
])
        # self.K = np.array([
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        # ])  # --- IGNORE ---


        # -------- PID --------
        self.pid_l0_p = PID_control(3000, 0, 1000, 0.15)
        self.pid_l0_s = PID_control(3000, 0, 1000, 0.15)

        self.pid_roll = PID_control(0, 0, 0, 0)
        self.pid_delta = PID_control(100, 0, 0, 0)
        self.pid_yaw = PID_control(10, 0, 0, 0)

    # -----------------------
    # 通用LQR计算
    # -----------------------
    def compute_lqr(self, leg):
        s = leg.state

        x = np.array([
            s["theta"],
            s["dtheta"],
            s["s"],
            s["dot_s"],
            s["phi"],
            s["dphi"]
        ])

        x_ref = np.array([
            leg.target.get("theta", 0),
            leg.target.get("dtheta", 0),
            leg.target.get("s", 0),
            leg.target.get("dot_s", 0),
            leg.target.get("phi", 0),
            leg.target.get("dphi", 0)
        ])

        err = x - x_ref

        u = self.K @ err   # (2,)
    
        T_w = (u[0])
        T_p = (u[1])

        return T_w, T_p

    # -----------------------
    # 左腿控制
    # -----------------------
    def control_left(self, leg, imu):
        # ---- LQR ----
        T_w, T_p = self.compute_lqr(leg)

        # ---- PID: 支撑力 ----
        F0_p = self.pid_l0_p.position_pid(leg.target["l0"], leg.vmc["L0"])
        # F0_s = self.pid_l0_s.position_pid(F0_p, leg.vmc["L0_dot"])

        dF_0 = F0_p

        # ---- Roll ----
        dF_roll = self.pid_roll.position_pid(leg.target["roll"], imu["euler"][0])

        # ---- Delta ----
        dF_delta = self.pid_delta.position_pid(leg.target["d2theta"], leg.state["delta"])

        # ---- Yaw ----
        dF_yaw = self.pid_yaw.position_pid(leg.target["yaw"], imu["euler"][2])

        # ---- F0 ----
        theta = leg.state["theta"]

        F_0 = (
            MASS_BODY / 2.0 * GRAVITY / np.cos(theta)
            + dF_0
            # - dF_roll
        )
        
        # ---- 修正 ----
        T_p = T_p + dF_delta
        T_w = T_w + dF_yaw

        # ---- 力 -> 关节 ----
        J = leg.vmc["J"]

        tau = J @ np.array([F_0, T_p])
        tau_w = T_w

        # ---- 限幅 ----
        tau[0] = np.clip(tau[0], -MAX_TORQUE_LEG_T, MAX_TORQUE_LEG_T)
        tau[1] = np.clip(tau[1], -MAX_TORQUE_LEG_T, MAX_TORQUE_LEG_T)
        tau_w = np.clip(tau_w, -MAX_TORQUE_LEG_W, MAX_TORQUE_LEG_W)

        return tau, tau_w

    # -----------------------
    # 右腿控制（符号不同！）
    # -----------------------
    def control_right(self, leg, imu):
        T_w, T_p = self.compute_lqr(leg)

        F0_p = self.pid_l0_p.position_pid(leg.target["l0"], leg.vmc["L0"])
        # F0_s = self.pid_l0_s.position_pid(F0_p, leg.vmc["L0_dot"])

        dF_0 = F0_p

        # 注意符号（和C一致）
        dF_roll = -self.pid_roll.position_pid(leg.target["roll"], imu["euler"][0])

        dF_delta = self.pid_delta.position_pid(leg.target["d2theta"], leg.state["delta"])
        dF_yaw = self.pid_yaw.position_pid(leg.target["yaw"], imu["euler"][2])

        theta = leg.state["theta"]

        F_0 = (
            MASS_BODY / 2.0 * GRAVITY / np.cos(theta)
            + dF_0
            # + dF_roll
        )

        # 注意符号（关键区别）
        T_p = T_p - dF_delta
        T_w = T_w - dF_yaw

        J = leg.vmc["J"]

        tau = J @ np.array([F_0, T_p])
        tau_w = T_w

        tau[0] = np.clip(tau[0], -MAX_TORQUE_LEG_T, MAX_TORQUE_LEG_T)
        tau[1] = np.clip(tau[1], -MAX_TORQUE_LEG_T, MAX_TORQUE_LEG_T)
        tau_w = np.clip(tau_w, -MAX_TORQUE_LEG_W, MAX_TORQUE_LEG_W)

        return tau, tau_w