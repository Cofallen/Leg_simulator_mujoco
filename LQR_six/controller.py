import mujoco
import mujoco.viewer
import time

from lqr_controller import LQRController
from leg_data import Leg
from get_sensor import RobotSensor
from write_sensor import RobotController
from vmc import VMC
from vofa import VOFA
from get_state import StateEstimator
from keyboardInput import KeyboardInput
from mpcTraj import TrajMPC
import numpy as np

model = mujoco.MjModel.from_xml_path("chuan.xml")
data = mujoco.MjData(model)

sensor = RobotSensor(model, data)
controller = RobotController(model, data)


dt = model.opt.timestep

vmc = VMC()
leg_L = Leg(dt)
leg_R = Leg(dt)
vofa = VOFA()
state_estimator = StateEstimator(dt)
kb = KeyboardInput(dt)

lqr = LQRController()
mpc = TrajMPC()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()

        # --- control ---
        # controller.set_actuator("left_wheel", -0.1)
        # controller.set_actuator("left_front", 1.5)
        # controller.set_actuator("left_back",  1.5)
        # --- step simulation ---
        mujoco.mj_step(model, data)

        state = sensor.get_state()

        imu = {
            "acc": state["acc"],
            "gyro": state["gyro"],
            "euler": state["euler"]
        }

        # 左腿
        motor = {
            "left_front": state["joints"]["joint_l1_L"],
            "left_back": state["joints"]["joint_l6_L"],
            "right_front": state["joints"]["joint_l1_R"],
            "right_back": state["joints"]["joint_l6_R"],
            "left_wheel": state["joints"]["joint_wheel_L"],
            "right_wheel": state["joints"]["joint_wheel_R"],
        }

        vmc.update_left(leg_L, motor)
        vmc.update_right(leg_R, motor)
        vmc.get_Fn(leg_L, imu)
        vmc.get_Fn(leg_R, imu)
        state_estimator.update_left(leg_L, imu, dt)
        state_estimator.update_right(leg_R, imu, dt)
        state_estimator.update(leg_L, leg_R, motor, imu)

        # vofa.send_command(leg_L.vmc["L0"],leg_R.vmc["L0"])
        # vofa.send_command(leg_L.state["theta"], leg_L.state["dtheta"] ,leg_L.state["s"], leg_L.state["phi"], leg_L.state["dphi"],)
        # vofa.send_command(leg_L.state["phi"], leg_L.state["dphi"], leg_L.state["s"], leg_L.state["dot_s"], leg_L.state["theta"], leg_L.state["dtheta"])
        # vofa.send_command(leg_L.LQR["F_0"])
        # 左腿
        tau_L, tau_w_L = lqr.control_left(leg_L, imu)

        # 右腿
        tau_R, tau_w_R = lqr.control_right(leg_R, imu)

        # ---- 输出到MuJoCo ----
        controller.set_actuator("left_front", -tau_L[0])
        controller.set_actuator("left_back",  -tau_L[1])
        controller.set_actuator("left_wheel", -tau_w_L)

        controller.set_actuator("right_front", tau_R[0])
        controller.set_actuator("right_back",  tau_R[1])
        controller.set_actuator("right_wheel", tau_w_R)

        target = kb.get_target()
        kb.update()
        
        x = state["pos"]
        print(x)
        x = x[0:3]
        x_ref = np.array([
            1.0,
            1.0,
            0.0
        ])
        v, w = mpc.solve(x, x_ref)
        
        leg_L.target["dot_s"] = v
        leg_R.target["dot_s"] = v
        leg_L.target["s"] += v * 0.002
        leg_R.target["s"] += v * 0.002
    
        leg_L.target["yaw"] = w
        leg_R.target["yaw"] = w
        leg_L.target["l0"] = target["l0"]
        leg_R.target["l0"] = target["l0"]

        # --- sync viewer ---
        viewer.sync()

        elapsed = time.time() - start
        if elapsed < dt:
            time.sleep(dt - elapsed)
            