import numpy as np
import sympy as sp
import scipy.linalg
# import serial
import struct

# 获取步态周期内 各时刻 k 值
# TODO 写拟合/实时求解/其他mpc高级方法
def get_k_solve(L0_l, L0_r, theta, phi, R1, l1, mw1, mp1, M1, Iw1, Ip1, IM1):
    """
    get_k_gan 的 Docstring
    
    :param L0: 杆长 先只控左腿
    :param theta: 相对于垂直轴的顺时针夹角
    """
    L1 = L0_l / 2
    LM1 = L0_l / 2 
    pass

def get_k(leg_length):
    # 1. 定义符号变量
    t = sp.symbols('t')
    theta = sp.Function('theta')(t)
    x = sp.Function('x')(t)
    phi = sp.Function('phi')(t)
    
    theta_dot, theta_ddot = sp.symbols('theta_dot theta_ddot')
    x_dot, x_ddot = sp.symbols('x_dot x_ddot')
    phi_dot, phi_ddot = sp.symbols('phi_dot phi_ddot')

    # 中间变量和输入
    N, P, N_M, P_M = sp.symbols('N P N_M P_M')
    T, T_p = sp.symbols('T T_p')

    # 2. 物理参数 (直接定义为数值，不再用符号)
    R_val = 0.05
    L_val = leg_length / 2.0
    L_M_val = leg_length / 2.0
    Body_val = 0.574
    l_val = 0.028
    m_w_val = 0.572
    m_p_val = 0.9810
    M_val = 10.0 / 2.0
    I_w_val = 0.5 * m_w_val * R_val**2
    # I_p_val = m_p_val * ((leg_length)**2 + 0.12**2) / 12.0
    # I_M_val = M_val * (0.24**2 + 0.22**2) / 12.0
    I_p_val = m_p_val * leg_length**2 / 12.0 + m_p_val * 0.0084306**2
    I_M_val = M_val * Body_val**2 / 12.0 + Body_val * l_val**2 
    g_val = 9.81

    # 3. 动力学方程推导 (直接使用数值参数)
    x_b = x + (L_val + L_M_val) * sp.sin(theta)
    
    replacements = [
        (sp.diff(theta, t, 2), theta_ddot), 
        (sp.diff(x, t, 2), x_ddot), 
        (sp.diff(phi, t, 2), phi_ddot),
        (sp.diff(theta, t), theta_dot), 
        (sp.diff(x, t), x_dot), 
        (sp.diff(phi, t), phi_dot)
    ]

    def diff_and_sub(expr, order=1):
        d_expr = sp.diff(expr, t, order)
        return d_expr.subs(replacements).subs(replacements)

    # 方程 f4, f5, f7, f8 (带入数值)
    eq1 = N - N_M - m_p_val * (x_ddot + diff_and_sub(L_val * sp.sin(theta), 2))
    eq2 = P - P_M - m_p_val * g_val - m_p_val * diff_and_sub(L_val * sp.cos(theta), 2)
    eq3 = N_M - M_val * diff_and_sub(x_b - l_val * sp.sin(phi), 2)
    eq4 = P_M - M_val * g_val - M_val * diff_and_sub((L_val + L_M_val) * sp.cos(theta) + l_val * sp.cos(phi), 2)

    # 4. 求解中间变量
    # 此时方程中只有 theta, phi 及其导数是符号，参数都是数字，求解会非常快
    sol_intermediate = sp.solve([eq1, eq2, eq3, eq4], [N, P, N_M, P_M], dict=True)
    if not sol_intermediate: return None
    
    sol_dict = sol_intermediate[0]
    N_sol = sol_dict[N]
    P_sol = sol_dict[P]
    N_M_sol = sol_dict[N_M]
    P_M_sol = sol_dict[P_M]

    # 5. 求解状态变量方程
    eq_f3 = (T - N_sol * R_val) / (I_w_val / R_val + m_w_val * R_val) - x_ddot
    
    eq_f6 = (P_sol * L_val + P_M_sol * L_M_val) * sp.sin(theta) - \
            (N_sol * L_val + N_M_sol * L_M_val) * sp.cos(theta) - T + T_p - I_p_val * theta_ddot
            
    eq_f9 = T_p + N_M_sol * l_val * sp.cos(phi) + P_M_sol * l_val * sp.sin(phi) - I_M_val * phi_ddot

    sols_acc = sp.solve([eq_f3, eq_f6, eq_f9], [theta_ddot, x_ddot, phi_ddot], dict=True)
    if not sols_acc: return None
        
    acc_dict = sols_acc[0]
    g1 = acc_dict[theta_ddot]
    g2 = acc_dict[x_ddot]
    g3 = acc_dict[phi_ddot]

    # 6. 线性化
    theta0, x0, phi0 = sp.symbols('theta0 x0 phi0')
    subs_linear = [(theta, theta0), (x, x0), (phi, phi0)]
    
    g1 = g1.subs(subs_linear)
    g2 = g2.subs(subs_linear)
    g3 = g3.subs(subs_linear)

    f_vec = sp.Matrix([theta_dot, g1, x_dot, g2, phi_dot, g3])
    state_vars = sp.Matrix([theta0, theta_dot, x0, x_dot, phi0, phi_dot])
    input_vars = sp.Matrix([T, T_p])

    # 求雅可比 (此时表达式已经包含数值，求导会很简单)
    A_sym = f_vec.jacobian(state_vars)
    B_sym = f_vec.jacobian(input_vars)

    # 7. 代入平衡点
    eq_point = {theta0: 0, theta_dot: 0, x0: 0, x_dot: 0, phi0: 0, phi_dot: 0, T: 0, T_p: 0}
    
    A_num = np.array(A_sym.subs(eq_point)).astype(np.float64)
    B_num = np.array(B_sym.subs(eq_point)).astype(np.float64)

    # 8. LQR 求解
    Q = np.diag([3000, 1, 500, 1, 30000, 1])
    R_mat = np.diag([1, 1]) 
    try:
        P_sol = scipy.linalg.solve_continuous_are(A_num, B_num, Q, R_mat)
        K = np.linalg.inv(R_mat) @ B_num.T @ P_sol
        return A_num, B_num, np.array(K)
    except Exception as e:
        print(f"LQR Solver Error: {e}")
        return None, None, None

def polyfit_matrix(x, Y, order):
    return [np.polyfit(x, Y[:, i], order) for i in range(Y.shape[1])]

def fit_ABK():
    leg_lengths = np.arange(0.12, 0.36, 0.005)

    A_all, B_all, K_all = [], [], []

    for ll in leg_lengths:
        print("solve:", float(ll))
        A, B, K = get_k(float(ll))

        if A is None:
            continue

        A_all.append(A.flatten())
        B_all.append(B.flatten())
        K_all.append(K.flatten())

    A_all = np.array(A_all)   # (N, 36)
    B_all = np.array(B_all)   # (N, 12)
    K_all = np.array(K_all)   # (N, 12)

    order = 3

    A_coeffs = polyfit_matrix(leg_lengths, A_all, order)
    B_coeffs = polyfit_matrix(leg_lengths, B_all, order)
    K_coeffs = polyfit_matrix(leg_lengths, K_all, order)

    return A_coeffs, B_coeffs, K_coeffs

def export_matrix(name, coeffs):
    print(f"float {name}[{len(coeffs)}][4] = {{")
    for c in coeffs:
        print("    { " + ", ".join(f"{x:.8f}" for x in c) + " },")
    print("};\n")


def export_all(Ac, Bc, Kc):
    export_matrix("A_coeffs", Ac)
    export_matrix("B_coeffs", Bc)
    export_matrix("K_coeffs", Kc)
    
def print_matrix(name, mat):
    print(f"self.{name} = np.array([")
    for row in mat:
        row_str = ", ".join([f"{float(x):.8f}" for x in row])
        print(f"    [{row_str}],")
    print("])\n")
    
    
if __name__ == "__main__":
    leg_len = 0.15   # 你想要的腿长

    A, B, K = get_k(leg_len)

    if A is not None:
        print_matrix("A", A)
        print_matrix("B", B)
        print_matrix("K", K)