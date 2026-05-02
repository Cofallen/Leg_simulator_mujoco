# MPC 控制器原理与代码对应（从 LQR 到实现）

## 1. 问题背景

我们考虑一个线性系统（你代码里的模型）：

```
x_dot = A x + B u
```

离散化后：

```
x[k+1] = A x[k] + B u[k]
```

其中：

* `x ∈ R^6`：状态（theta, dtheta, s, dot_s, phi, dphi）
* `u ∈ R^2`：控制（T_w, T_p）

---

## 2. LQR 的本质（你现在已有）

你现在使用的是误差系统：

```
e = x - x_ref
e_dot = A e + B u
```

目标：

```
min ∫ (e^T Q e + u^T R u) dt
```

解出来是：

```
u = -K e
```

👉 特点：

* 无限时域（∞ horizon）
* 一次求解，得到固定增益 K
* 本质：**最优线性反馈**

---

## 3. MPC 是什么？

MPC（Model Predictive Control）做的事情：

👉 每一帧都在解一个“有限时域最优控制问题”

---

### 3.1 优化目标

```
min Σ_{k=0}^{N-1} (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k
    + (x_N - x_ref)^T P (x_N - x_ref)
```

---

### 3.2 约束

```
x[k+1] = A x[k] + B u[k]
```

---

👉 核心思想：

* 预测未来 N 步
* 优化整段控制序列
* 只执行第一步
* 下一帧重新算

---

## 4. 从单步到“堆叠系统”（关键推导）

我们把未来状态展开：

```
x1 = A x0 + B u0
x2 = A x1 + B u1 = A^2 x0 + A B u0 + B u1
x3 = ...
```

---

### 4.1 写成矩阵形式

定义：

```
X = [x1
     x2
     ...
     xN]

U = [u0
     u1
     ...
     u_{N-1}]
```

则：

```
X = A_bar x0 + B_bar U
```

---

### 4.2 A_bar（预测矩阵）

```
A_bar =
[A
 A^2
 ...
 A^N]
```

👉 对应代码：

```python
A_power = np.linalg.matrix_power(A, i + 1)
A_bar[i*nx:(i+1)*nx, :] = A_power
```

---

### 4.3 B_bar（控制矩阵）

```
B_bar =
[B      0      0
 A B    B      0
 A^2 B  A B    B
 ...
]
```

👉 对应代码：

```python
A_ij = np.linalg.matrix_power(A, i - j)
B_block = A_ij @ B
```

---

## 5. 代价函数矩阵化

目标函数：

```
J = Σ (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k
```

---

### 5.1 写成二次型

```
J = (X - X_ref)^T Q_bar (X - X_ref) + U^T R_bar U
```

其中：

```
Q_bar = diag(Q, Q, ..., P)
R_bar = diag(R, R, ..., R)
```

---

👉 对应代码：

```python
Q_list = [Q]*(N-1) + [P]
Q_bar = block_diag(...)
R_bar = block_diag(...)
```

---

## 6. 代入系统关系

```
X = A_bar x0 + B_bar U
```

代入：

```
J = (A_bar x0 + B_bar U - X_ref)^T Q_bar (...) + U^T R_bar U
```

展开得到：

```
J = U^T H U + 2 f^T U + const
```

---

### 6.1 Hessian（核心）

```
H = B_bar^T Q_bar B_bar + R_bar
```

---

### 6.2 一次项

```
f = B_bar^T Q_bar (A_bar x0 - X_ref)
```

---

👉 对应代码：

```python
self.H = B_bar.T @ Q_bar @ B_bar + R_bar
f = B_bar.T @ Q_bar @ e
```

---

## 7. 最优解

二次优化问题：

```
min 1/2 U^T H U + f^T U
```

解：

```
U* = - H^{-1} f
```

---

👉 对应代码：

```python
U = np.linalg.solve(self.H , f)
u0 = -U[:nu]
```

⚠️ 注意符号（你实际系统是反号）

---

## 8. 为什么要加 Δu 惩罚？

原始问题：

```
只惩罚 u
```

会导致：

* 控制剧烈跳变
* 电机抖动

---

### 8.1 定义：

```
Δu_k = u_k - u_{k-1}
```

---

### 8.2 新代价：

```
Σ Δu^T Rd Δu
```

---

### 8.3 矩阵形式：

```
D U = [u1-u0
       u2-u1
       ...]
```

```
J += U^T D^T Rd_bar D U
```

---

👉 对应代码：

```python
D[i, i] = -1
D[i, i+1] = 1

self.H += D.T @ Rd_bar @ D
```

---

## 9. 为什么要 Terminal Cost（P）

MPC是有限时域：

```
只看N步
```

👉 会“短视”

---

LQR解：

```
P = solve_discrete_are(A, B, Q, R)
```

---

👉 加入：

```
最后一步用 P
```

👉 等价于：

👉 **逼近无限时域 LQR**

---

## 10. 和 LQR 的关系

| 方法  | 本质       |
| --- | -------- |
| LQR | 无限时域 MPC |
| MPC | 有限时域 LQR |

---

当：

```
N → ∞
```

👉 MPC → LQR

---

## 11. 你的代码流程总结

每一帧：

```
1. 读取状态 x
2. 计算参考 x_ref
3. solve MPC:
    - 构建误差
    - 解 U
4. 取 u0
5. 输出控制
```

---

## 12. 最关键的工程点（你已经踩过）

### ❗ 1. 必须离散化

```
A_d = exp(A dt)
```

---

### ❗ 2. 控制符号必须一致

```
u = -Kx   （标准）
```

---

### ❗ 3. Δu 必须有

否则：

👉 抖动 / 炸

---

### ❗ 4. Terminal cost 必须有

否则：

👉 不稳定

---

## 13. 一句话总结

> MPC = “每一帧重新做一次有限时域 LQR，并只执行第一步控制”

---

## 14. 你现在这套控制器的本质

```
离散线性系统 + 有限时域最优控制 + 平滑约束（Δu） + 终端稳定（P）
```

👉 已经是：

👉 **标准机器人 MPC 框架（入门完整体）**

---

## 15. 下一步可以做什么？

* 加约束（QP）
* 做 Δu-based MPC（更稳）
* 加 gait / 速度参考
* 做非线性 MPC（NMPC）

---

# ✅ 总结

你现在已经完成：

✔ 从 LQR → MPC
✔ 从静态反馈 → 预测控制
✔ 从理论 → 实际机器人运行

👉 这一步非常关键

---

如果你后面要做“能走”，下一步就是：

👉 MPC + 轨迹 / 落脚点规划
