import numpy as np
import scipy.linalg
from scipy.linalg import expm

def c2d(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n+m, n+m))
    M[:n, :n] = A
    M[:n, n:] = B

    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd = Md[:n, n:]

    return Ad, Bd


class MPCController:
    def __init__(self, A, B, N=30, dt=0.01):
        self.A, self.B = c2d(A, B, dt)   # ✅ 必须离散化
        self.N = N
        self.dt = dt
        
        self.nd = 2  # 扰动维度
        
        # self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.nx_orig = A.shape[0]
        self.nx = self.nx_orig + self.nd
        
        # -------- 稳定优先参数 --------
        self.Q = np.diag([1000, 10, 1000, 5, 1000, 5])
        self.R = np.diag([1, 1])
        self.Rd = np.diag([1, 1])

        self.u_min = np.array([-100, -100])
        self.u_max = np.array([100, 100])
        
        self.A_aug = np.block([
            [self.A, np.eye(self.nx_orig, self.nd)],
            [np.zeros((self.nd, self.nx_orig)), np.eye(self.nd)]
        ])
        
        self.B_aug = np.block([
            [self.B],
            [np.zeros((self.nd, self.nu))]
        ])
        
        self.E_aug = np.block([
            [np.eye(self.nx, self.nd)],
            [np.eye(self.nd)]
        ])
        
        self._build_prediction_matrices()

    # -----------------------------
    def _build_prediction_matrices(self):
        A, B, N = self.A_aug, self.B_aug, self.N
        nx, nu = self.nx_orig + self.nd, self.nu

        A_bar = np.zeros((nx * N, nx))
        B_bar = np.zeros((nx * N, nu * N))

        for i in range(N):
            A_power = np.linalg.matrix_power(A, i + 1)
            A_bar[i*nx:(i+1)*nx, :] = A_power

            for j in range(i + 1):
                A_ij = np.linalg.matrix_power(A, i - j)
                B_block = A_ij @ B
                B_bar[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = B_block

        self.A_bar = A_bar
        self.B_bar = B_bar

        # -------- Terminal cost --------
        Q_aug = np.block([
            [self.Q, np.zeros((self.nx_orig, self.nd))],
            [np.zeros((self.nd, self.nx_orig)), np.eye(self.nd) * 10]
        ])

        Q_list = [Q_aug] * N
        Q_bar = scipy.linalg.block_diag(*Q_list)
        R_bar = scipy.linalg.block_diag(*([self.R] * N))

        # -------- Δu惩罚 --------
        D = np.zeros((nu*(N-1), nu*N))
        for i in range(N-1):
            for j in range(nu):
                D[i*nu+j, i*nu+j] = -1
                D[i*nu+j, (i+1)*nu+j] = 1

        Rd_bar = scipy.linalg.block_diag(*([self.Rd] * (N-1)))

        # -------- Hessian --------
        self.H = B_bar.T @ Q_bar @ B_bar + R_bar + D.T @ Rd_bar @ D
        self.Q_bar = Q_bar

    # -----------------------------
    def solve(self, x0, d_hat, x_ref=None):
        if x_ref is None:
            x_ref = np.zeros(self.nx)

        x0_aug = np.concatenate([x0, d_hat])
        x_ref_aug = np.concatenate([x_ref, d_hat])

        x_ref_bar = np.tile(x_ref_aug, self.N)
        e = self.A_bar @ x0_aug - x_ref_bar
        
        f = self.B_bar.T @ self.Q_bar @ e

        try:
            U = -np.linalg.solve(self.H, f)
        except:
            U = np.zeros(self.nu * self.N)

        u0 = U[:self.nu]

        u0 = np.clip(u0, self.u_min, self.u_max)

        return u0