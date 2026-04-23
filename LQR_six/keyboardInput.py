from pynput import keyboard
import threading
from mymath import Discreteness

class KeyboardInput:
    def __init__(self, dt):
        self.lock = threading.Lock()
        self.disc_s = Discreteness(dt)

        self.target = {
            "theta": 0.0,
            "s": 0.0,
            "dot_s": 0.0,
            "phi": 0.0,
            "yaw": 0.0,
        }

        self.step = {
            "dot_s": 1.0,
            "yaw": 0.05,
        }

        self.pressed = set()

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            self.pressed.add(key)

    def on_release(self, key):
        with self.lock:
            if key in self.pressed:
                self.pressed.remove(key)

    def update(self):
        with self.lock:
            if keyboard.Key.up in self.pressed:
                self.target["dot_s"] = self.step["dot_s"]
            elif keyboard.Key.down in self.pressed:
                self.target["dot_s"] = -self.step["dot_s"]
            else:
                self.target["dot_s"] = 0.0

            self.target["s"] = self.disc_s.Sum(self.target["dot_s"])

            if keyboard.Key.left in self.pressed:
                self.target["yaw"] += self.step["yaw"]
            if keyboard.Key.right in self.pressed:
                self.target["yaw"] -= self.step["yaw"]

    def get_target(self):
        with self.lock:
            return self.target.copy()