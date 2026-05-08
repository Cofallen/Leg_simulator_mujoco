import numpy as np
from scipy.optimize import minimize


class TrajMPC:

    def __init__(self, dt=0.05, N=20):

        self.dt = dt
        self.N = N

        # 状态权重
        self.Q = np.diag([100.0, 100.0, 0.1])

        # 输入权重
        self.R = np.diag([0.1, 1.0])

        # 控制约束
        self.v_max = 0.2
        self.w_max = 2.0

    # ==========================================
    # 线性化离散模型
    # ==========================================
    def linearize(self, v, theta):

        A = np.eye(3) + np.array([
            [0, 0, -v * np.sin(theta)],
            [0, 0,  v * np.cos(theta)],
            [0, 0, 0]
        ]) * self.dt

        B = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ]) * self.dt

        return A, B

    # ==========================================
    # 构造预测矩阵
    # ==========================================
    def build_prediction(self, A, B):

        n = A.shape[0]
        m = B.shape[1]

        Phi = np.zeros((n * self.N, n))
        Gamma = np.zeros((n * self.N, m * self.N))

        for i in range(self.N):

            A_power = np.linalg.matrix_power(A, i + 1)

            Phi[i*n:(i+1)*n, :] = A_power

            for j in range(i + 1):

                A_tmp = np.linalg.matrix_power(A, i-j)

                Gamma[
                    i*n:(i+1)*n,
                    j*m:(j+1)*m
                ] = A_tmp @ B

        return Phi, Gamma

    # ==========================================
    # MPC求解
    # ==========================================
    def solve(self, x, x_ref, v_ref=0.5):

        theta = x[2]

        A, B = self.linearize(v_ref, theta)

        Phi, Gamma = self.build_prediction(A, B)

        n = 3
        m = 2

        # 展开Q R
        Q_bar = np.kron(np.eye(self.N), self.Q)
        R_bar = np.kron(np.eye(self.N), self.R)

        # 参考轨迹展开
        x_ref_bar = np.tile(x_ref, self.N)

        # 二次型
        H = Gamma.T @ Q_bar @ Gamma + R_bar

        f = Gamma.T @ Q_bar @ (Phi @ x - x_ref_bar)

        # ======================================
        # 目标函数
        # ======================================
        def cost(U):

            return 0.5 * U.T @ H @ U + f.T @ U

        # ======================================
        # 初值
        # ======================================
        U0 = np.zeros(m * self.N)

        # ======================================
        # 输入约束
        # ======================================
        bounds = []

        for i in range(self.N):

            bounds.append((-self.v_max, self.v_max))
            bounds.append((-self.w_max, self.w_max))

        # ======================================
        # 求解
        # ======================================
        res = minimize(
            cost,
            U0,
            bounds=bounds,
            method='SLSQP'
        )

        U_opt = res.x

        # 只执行第一步
        v_cmd = U_opt[0]
        w_cmd = U_opt[1]

        return v_cmd, w_cmd