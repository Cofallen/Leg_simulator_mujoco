#!/usr/bin/env python3
"""Read sensors from the MuJoCo model and write motor commands.

This script supports both the modern `mujoco` Python bindings and
`mujoco_py` as a fallback. It parses the XML to enumerate sensors
in order so sensordata can be split into named entries.
"""
import time
import argparse
import xml.etree.ElementTree as ET
import math
import sys


def parse_sensors(xml_path):
    root = ET.parse(xml_path).getroot()
    sensor_elems = root.findall('.//sensor/*')
    info = []
    # approximate dims for common sensors (used to slice sensordata)
    dims = {
        'framequat': 4,
        'accelerometer': 3,
        'acc': 3,
        'gyro': 3,
        'jointpos': 1,
        'jointvel': 1,
        'force': 1,
    }
    for elem in sensor_elems:
        tag = elem.tag.lower()
        name = elem.get('name') or elem.get('site') or elem.get('joint') or tag
        dim = dims.get(tag, 1)
        info.append((tag, name, dim))
    return info


def split_sensordata(sensordata, sensor_info):
    out = {}
    idx = 0
    for tag, name, dim in sensor_info:
        out[name] = sensordata[idx: idx + dim].copy()
        idx += dim
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', '-x', default='chuan.xml', help='MuJoCo XML file')
    parser.add_argument('--steps', '-n', type=int, default=1000, help='Simulation steps')
    parser.add_argument('--headless', action='store_true', help='Run headless (no viewer)')
    args = parser.parse_args()

    sensor_info = parse_sensors(args.xml)
    print('Detected sensors:')
    for tag, name, dim in sensor_info:
        print(f'  - {name} ({tag}) dim={dim}')

    # Try modern `mujoco` first, fallback to `mujoco_py`
    sim = None
    backend = None
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(args.xml)
        sim = mujoco.MjSim(model)
        backend = 'mujoco'
    except Exception:
        try:
            import mujoco_py
            model = mujoco_py.load_model_from_path(args.xml)
            sim = mujoco_py.MjSim(model)
            backend = 'mujoco_py'
        except Exception as e:
            print('Failed to import mujoco or mujoco_py:', e)
            sys.exit(1)

    print('Using backend:', backend)

    # optional viewer
    viewer = None
    if not args.headless and backend == 'mujoco':
        try:
            from mujoco import viewer as muj_viewer
            viewer = muj_viewer.launch_passive(sim.model, sim.data)
        except Exception:
            viewer = None
    elif not args.headless and backend == 'mujoco_py':
        try:
            from mujoco_py import MjViewer
            viewer = MjViewer(sim)
        except Exception:
            viewer = None

    # main loop
    dt = getattr(sim.model.opt, 'timestep', 0.002) if hasattr(sim.model, 'opt') else 0.002
    try:
        for i in range(args.steps):
            # read sensordata
            sd = sim.data.sensordata
            sensors = split_sensordata(sd, sensor_info)

            # print a short summary
            if i % 50 == 0:
                print(f'step={i}')
                for name, val in sensors.items():
                    print(f'  {name}: {val}')

            # Example control: simple sinusoidal command for wheel actuators
            # we try to set all actuator controls; if sizes mismatch we clip
            try:
                nctrl = sim.data.ctrl.size
            except Exception:
                nctrl = len(sim.data.ctrl)

            cmds = [0.0] * nctrl
            # put a small oscillation on any actuator whose name contains 'wheel'
            for ai in range(nctrl):
                # best-effort: check actuator name when available
                aname = None
                try:
                    aname = sim.model.actuator_name[ai]
                except Exception:
                    try:
                        aname = sim.model.actuator_names[ai]
                    except Exception:
                        aname = ''
                if aname and 'wheel' in aname:
                    cmds[ai] = 0.5 * math.sin(i * 0.1)

            # write controls
            for k in range(min(len(sim.data.ctrl), len(cmds))):
                sim.data.ctrl[k] = cmds[k]

            # step simulation
            sim.step()

            # render if viewer present
            if viewer is not None:
                try:
                    if backend == 'mujoco':
                        viewer_sync = viewer.sync(sim.model, sim.data)
                    else:
                        viewer.render()
                except Exception:
                    pass

            time.sleep(dt)
    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    main()
