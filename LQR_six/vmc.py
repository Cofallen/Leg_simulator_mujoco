import numpy as np
from vofa import VOFA

vofa = VOFA()

class VMC:
    def __init__(self):
        # ---- geometry ----
        self.l1 = 0.215
        self.l2 = 0.258
        self.l3 = 0.258
        self.l4 = 0.215
        self.l5 = 0.0

        # ---- lowpass state ----
        self.ddl_fL = 0.0
        self.acc_fL = 0.0
        self.ddtheta_fL = 0.0

        self.MASS_WHEEL = 1.0   # 按你的实际质量修改

    # ---------------------------
    # lowpass filter
    # y += alpha * (x - y)
    # ---------------------------
    def lowpass_filter(self, y_last, x, alpha):
        return y_last + alpha * (x - y_last)

    # ---------------------------
    # kinematics
    # ---------------------------
    def get_phi(self, phi1, phi4):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        x_B = -l5 / 2 + np.cos(phi1) * l1
        y_B = np.sin(phi1) * l1

        x_D = l5 / 2 + np.cos(phi4) * l4
        y_D = np.sin(phi4) * l4

        A0 = 2 * l2 * (x_D - x_B)
        B0 = 2 * l2 * (y_D - y_B)
        l_BD = np.sqrt((x_D - x_B) ** 2 + (y_D - y_B) ** 2)
        C0 = l2**2 + l_BD**2 - l3**2

        phi2 = 2 * np.arctan2(
            B0 + np.sqrt(A0**2 + B0**2 - C0**2),
            A0 + C0
        )

        x_C = -l5 / 2 + l1 * np.cos(phi1) + l2 * np.cos(phi2)
        y_C = l1 * np.sin(phi1) + l2 * np.sin(phi2)

        phi3 = np.arctan2(y_C - y_D, x_C - x_D)

        L0 = np.sqrt(x_C**2 + y_C**2)
        phi0 = np.arctan2(y_C, x_C)

        return phi0, phi2, phi3, L0

    # ---------------------------
    # Jacobian
    # ---------------------------
    def get_J(self, phi0, phi1, phi2, phi3, phi4, L0):
        l1, l4 = self.l1, self.l4

        J = np.zeros((2, 2))

        denom = np.sin(phi2 - phi3)

        if abs(denom) < 1e-6:
            return J

        J[0, 0] = -l1 * np.sin(phi0 - phi3) * np.sin(phi1 - phi2) / denom
        J[0, 1] = -l1 * np.sin(phi1 - phi2) * np.cos(phi0 - phi3) / (L0 * denom)

        J[1, 0] = -l4 * np.sin(phi0 - phi2) * np.sin(phi3 - phi4) / denom
        J[1, 1] = -l4 * np.sin(phi3 - phi4) * np.cos(phi0 - phi2) / (L0 * denom)

        return J

    # ---------------------------
    # Ground force estimation
    # 对应你的 Vmc_getFnL()
    # ---------------------------
    def get_Fn(self, leg, imu):
        """
        leg:
            leg.vmc["L0"]
            leg.vmc["L0_dot"]
            leg.vmc["L0_ddot"]

            leg.state["theta"]
            leg.state["dtheta"]
            leg.state["ddtheta"]

            leg.lqr["F"]
            leg.lqr["Tp"]

        imu:
            imu["accel"][2]
        """

        theta = leg.state["theta"]
        dtheta = leg.state["dtheta"]
        ddtheta = leg.state["ddtheta"]

        L0 = leg.vmc["L0"]
        L0_dot = leg.vmc["L0_dot"]
        L0_ddot = leg.vmc["L0_ddot"]

        # ---- lowpass ----
        self.ddl_fL = L0_ddot

        self.acc_fL = imu["acc"][2]

        self.ddtheta_fL = ddtheta

        # ---- P ----
        P = (
            leg.LQR["F_0"] * np.cos(theta)
            + leg.LQR["T_p"] * np.sin(theta) / L0
        )

        # ---- ddz_w ----
        ddz_w = (
            (self.acc_fL - 9.81)
            - self.ddl_fL * np.cos(theta)
            + 2.0 * L0_dot * dtheta * np.sin(theta)
            + L0 * np.sin(theta) * self.ddtheta_fL
            + L0 * dtheta * dtheta * np.cos(theta)
        )

        # ---- Fn ----
        Fn = (
            P
            + self.MASS_WHEEL * 9.81
            + self.MASS_WHEEL * ddz_w
        )

        leg.LQR["Fn"] = Fn
        
        vofa.send_command(Fn)

        return Fn
    
    # 主更新（左腿）
    # ---------------------------
    def update_left(self, leg, motor):

        phi1 = np.pi - motor["left_front"]
        phi4 = -motor["left_back"]

        phi0, phi2, phi3, L0 = self.get_phi(phi1, phi4)

        L0_dot = leg.disc_L.Diff(L0)
        L0_ddot = leg.disc_dL.Diff(L0_dot)

        J = self.get_J(phi0, phi1, phi2, phi3, phi4, L0)
        
        leg.vmc["phi0"] = phi0
        leg.vmc["phi1"] = phi1
        leg.vmc["phi2"] = phi2
        leg.vmc["phi3"] = phi3
        leg.vmc["phi4"] = phi4

        leg.vmc["L0"] = L0
        leg.vmc["L0_dot"] = L0_dot
        leg.vmc["L0_ddot"] = L0_ddot

        leg.vmc["J"] = J

    # ---------------------------
    # 主更新（右腿）
    # ---------------------------
    def update_right(self, leg, motor):

        phi1 = np.pi - motor["right_back"]
        phi4 = -motor["right_front"]

        phi0, phi2, phi3, L0 = self.get_phi(phi1, phi4)

        L0_dot = leg.disc_L.Diff(L0)
        L0_ddot = leg.disc_dL.Diff(L0_dot)

        J = self.get_J(phi0, phi1, phi2, phi3, phi4, L0)

        leg.vmc["phi0"] = phi0
        leg.vmc["phi1"] = phi1
        leg.vmc["phi2"] = phi2
        leg.vmc["phi3"] = phi3
        leg.vmc["phi4"] = phi4

        leg.vmc["L0"] = L0
        leg.vmc["L0_dot"] = L0_dot
        leg.vmc["L0_ddot"] = L0_ddot

        leg.vmc["J"] = J