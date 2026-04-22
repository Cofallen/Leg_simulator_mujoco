from pynput import keyboard
import threading

class KeyboardInput:
    def __init__(self):
        self.lock = threading.Lock()

        # 你要控制的 target
        self.target = {
            "theta": 0.0,
            "dot_s": 0.5,
            "phi": 0.0,
            "yaw": 0.0,
        }

        self.step = {
            "theta": 0,
            "dot_s": 0.05,
            "phi": 0,
            "yaw": 0.05,
        }

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            with self.lock:
                # ---- 腿长 ----
                if key == keyboard.Key.up:
                    self.target["dot_s"] += self.step["dot_s"]
                elif key == keyboard.Key.down:
                    self.target["dot_s"] -= self.step["dot_s"]

                elif key == keyboard.Key.left:
                    self.target["yaw"] += self.step["yaw"]
                elif key == keyboard.Key.right:
                    self.target["yaw"] -= self.step["yaw"]

        except AttributeError:
            pass

    def get_target(self):
        with self.lock:
            return self.target.copy()