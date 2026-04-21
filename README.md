# Read sensors and send motor commands

Run `read_sensors.py` to read the sensors defined in `chuan.xml` and apply example motor commands.

Quick start:

1. Install dependencies (choose the proper MuJoCo binding for your setup):

```bash
pip install -r requirements.txt
# Or install the upstream mujoco package following its install docs
```

2. Run the script (headless):

```bash
python3 read_sensors.py --xml chuan.xml --headless
```

The script prints sensor values periodically and sets a small sinusoidal command on actuators whose name contains `wheel`.
