import numpy as np
from mymath import PID_control
from mympc import MPCController

# -------- 常量（按你实际填）--------
MASS_BODY = 10.0
GRAVITY = 9.81

MAX_TORQUE_LEG_T = 40.0
MAX_TORQUE_LEG_W = 100.0

import vofa

class LQRController:
    def __init__(self):
        # -------- LQR增益（你需要填真实K）--------
        self.A = np.array([
            [0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [391.70211876, 0.00000000, 0.00000000, 0.00000000, 1.98661410, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
            [-46.93083005, 0.00000000, 0.00000000, 0.00000000, -0.03544551, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
            [11.68661222, 0.00000000, 0.00000000, 0.00000000, 9.95509900, 0.00000000],
        ])

        self.B = np.array([
            [0.00000000, 0.00000000],
            [-164.65826694, 49.92885416],
            [0.00000000, 0.00000000],
            [22.71291988, -5.83460390],
            [0.00000000, 0.00000000],
            [-1.96266516, 8.69500008],
        ])

        self.K = np.array([
            [-61.60299326, -4.18541300, -21.92104795, -20.67561006, 44.69923054, 4.54768597],
            [14.67801961, 0.75238450, 4.41221675, 3.82666266, 168.92846876, 6.02502416],
        ])

        self.B_i = np.array([
            [0.00000000, -0.00574222, 0.00000000, 0.00557167, 0.00000000, 0.03671203],
            [0.00000000, 0.00071543, 0.00000000, 0.01567879, 0.00000000, 0.12142138],
        ])
        # self.K = np.array([
        #     [0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0],
        # ])  # --- IGNORE ---

        self.mpc = MPCController(self.A, self.B, N=30, dt=0.01)
        # -------- PID --------
        self.pid_l0_p = PID_control(3000, 0, 1000, 0.15)
        self.pid_l0_s = PID_control(3000, 0, 1000, 0.15)

        self.pid_roll = PID_control(0, 0, 0, 0)
        self.pid_delta = PID_control(10, 0, 0, 0)
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
        
        u = self.K @ err   
        
        T_w = (u[0])
        T_p = (u[1])

        return T_w, T_p
    
    def compute_mpc(self, leg):
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

        u = -self.mpc.solve(x, x_ref) # 注意符号

        T_w = u[0]
        T_p = u[1]

        return T_w, T_p

    # -----------------------
    # 左腿控制
    # -----------------------
    def control_left(self, leg, imu):
        # ---- LQR ----
        T_w, T_p = self.compute_mpc(leg)

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
        T_w, T_p = self.compute_mpc(leg)

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