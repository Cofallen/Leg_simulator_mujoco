import mujoco
import mujoco.viewer
import time

from get_sensor import RobotSensor
from write_sensor import RobotController

model = mujoco.MjModel.from_xml_path("chuan.xml")
data = mujoco.MjData(model)

sensor = RobotSensor(model, data)
controller = RobotController(model, data)


dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()

        # --- control ---
        controller.set_actuator("left_wheel", -0.1)

        # --- step simulation ---
        mujoco.mj_step(model, data)

        # --- read sensors ---
        state = sensor.get_state()
        print("Wheel L:", state["joints"]["joint_wheel_L"])
        print("Wheel R:", state["joints"]["joint_wheel_R"])

        # --- sync viewer ---
        viewer.sync()

        elapsed = time.time() - start
        if elapsed < dt:
            time.sleep(dt - elapsed)