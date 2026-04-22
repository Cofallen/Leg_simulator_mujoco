import numpy as np
from mymath import Discreteness

RADIUS_WHEEL = 0.05  # 轮子半径，单位米

class StateEstimator:
    def __init__(self, dt):
        self.dt = dt
        self.disc_s = Discreteness(dt)
        self.disc_theta_wl = Discreteness(dt)
        self.disc_theta_wr = Discreteness(dt)

    def update_left(self, leg, imu, dt):
        pitch = imu["euler"][1]
        gyro_x = imu["gyro"][1]

        theta = np.pi/2 - leg.vmc["phi0"] + pitch

        dtheta = leg.disc_theta.Diff(theta)
        ddtheta = leg.disc_dtheta.Diff(dtheta)

        phi = -pitch
        dphi = -gyro_x

        leg.state["theta"] = theta
        leg.state["dtheta"] = dtheta
        leg.state["ddtheta"] = ddtheta
        leg.state["phi"] = phi
        leg.state["dphi"] = dphi
    
    def update_right(self, leg, imu, dt):
        pitch = imu["euler"][1]
        gyro_x = imu["gyro"][1]

        theta = -np.pi/2 + leg.vmc["phi0"] + pitch

        dtheta = leg.disc_theta.Diff(theta)
        ddtheta = leg.disc_dtheta.Diff(dtheta)

        phi = -pitch
        dphi = -gyro_x

        leg.state["theta"] = theta
        leg.state["dtheta"] = dtheta
        leg.state["ddtheta"] = ddtheta
        leg.state["phi"] = phi
        leg.state["dphi"] = dphi
    
    def update(self, leg_L, leg_R, motor, imu):
        # --- wheel angle ---
        theta_wl = motor["left_wheel"]
        theta_wr = motor["right_wheel"]

        dtheta_wl = self.disc_theta_wl.Diff(theta_wl)
        dtheta_wr = self.disc_theta_wr.Diff(theta_wr)

        # --- base velocity ---
        dot_s_b = RADIUS_WHEEL * (dtheta_wl + dtheta_wr) / 2.0

        # --- kinematic compensation ---
        dot_s = dot_s_b \
            + 0.5 * (
                leg_L.vmc["L0"] * leg_R.state["dtheta"] * np.cos(leg_L.state["theta"])
                + leg_R.vmc["L0"] * leg_L.state["dtheta"] * np.cos(leg_R.state["theta"])
            ) \
            + 0.5 * (
                leg_L.vmc["L0_dot"] * np.sin(leg_L.state["theta"])
                + leg_R.vmc["L0_dot"] * np.sin(leg_R.state["theta"])
            )

        # --- integrate position ---
        s = self.disc_s.Sum(dot_s)

        # --- delta ---
        delta = leg_R.state["theta"] - leg_L.state["theta"]

        # --- 写回 ---
        for leg in [leg_L, leg_R]:
            leg.state["s"] = s
            leg.state["dot_s"] = dot_s
            leg.state["delta"] = delta
        