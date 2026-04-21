class RobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # cache actuator ids (faster than string lookup every loop)
        self.actuator_ids = {
            "left_front": model.actuator("left_front").id,
            "left_back": model.actuator("left_back").id,
            "left_wheel": model.actuator("left_wheel").id,
            "right_front": model.actuator("right_front").id,
            "right_back": model.actuator("right_back").id,
            "right_wheel": model.actuator("right_wheel").id,
        }

    # ---------------------------
    # Low-level actuator control
    # ---------------------------
    def set_actuator(self, name, value):
        self.data.ctrl[self.actuator_ids[name]] = value

    # ---------------------------
    # Wheel control
    # ---------------------------
    def set_wheel_velocity(self, left, right):
        """
        NOTE:
        You are using position actuators in XML.
        So this actually sets TARGET POSITION, not velocity.
        """
        self.set_actuator("left_wheel", left)
        self.set_actuator("right_wheel", right)

    # ---------------------------
    # Leg control (optional)
    # ---------------------------
    def set_leg_pose(self, lf, lb, rf, rb):
        self.set_actuator("left_front", lf)
        self.set_actuator("left_back", lb)
        self.set_actuator("right_front", rf)
        self.set_actuator("right_back", rb)

    # ---------------------------
    # Stop all motors
    # ---------------------------
    def stop_all(self):
        for key in self.actuator_ids:
            self.set_actuator(key, 0.0)