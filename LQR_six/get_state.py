import numpy as np


class StateEstimator:
    def update(self, leg, imu, dt):
        pitch = imu["euler"][1]
        gyro_x = imu["gyro"][0]

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