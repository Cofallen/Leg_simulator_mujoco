#!/usr/bin/env python3
"""
Convert MJCF box geometries with 'fromto' to standard 'pos', 'size', and 'quat'.
Usage: python convert_fromto.py input.xml output.xml
"""

import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

def fromto_to_quat(p1, p2):
    """Return quaternion (x,y,z,w) to rotate from X axis to direction v = p2-p1."""
    v = np.array(p2) - np.array(p1)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return [1, 0, 0, 0]  # identity
    u = v / norm
    # Default orientation: box extends along X axis (1,0,0)
    x_axis = np.array([1.0, 0.0, 0.0])
    # Axis of rotation = cross(x_axis, u)
    axis = np.cross(x_axis, u)
    angle = np.arccos(np.clip(np.dot(x_axis, u), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-12:
        # Parallel or antiparallel
        if angle < 1e-12:
            return [1, 0, 0, 0]  # identity
        else:
            # 180 degree rotation: any perpendicular axis, e.g. Y axis
            return [0, 0, 1, 0]  # 180 deg around Y
    axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(angle * axis)
    quat = rot.as_quat()  # [x, y, z, w]
    return quat.tolist()

def convert_geom(geom):
    if geom.get('type') != 'box':
        return
    fromto_str = geom.get('fromto')
    if fromto_str is None:
        return

    # Parse fromto
    try:
        coords = list(map(float, fromto_str.split()))
    except ValueError:
        print(f"Warning: invalid fromto '{fromto_str}', skipping.")
        return
    if len(coords) != 6:
        print(f"Warning: fromto requires 6 numbers, got {len(coords)}, skipping.")
        return
    p1 = coords[:3]
    p2 = coords[3:]

    # Get existing size (radius perpendicular to axis)
    size_str = geom.get('size')
    if size_str is None:
        print(f"Warning: geom without size attribute, using default 0.005")
        radius = 0.005
    else:
        # In MuJoCo, for fromto boxes, size is a single value
        try:
            radius = float(size_str)
        except ValueError:
            print(f"Warning: invalid size '{size_str}', using default")
            radius = 0.005

    # Compute center and length
    center = [(p1[i] + p2[i]) / 2.0 for i in range(3)]
    length = np.linalg.norm(np.array(p2) - np.array(p1))
    half_len = length / 2.0

    # New size: [half_len, radius, radius]
    new_size = [half_len, radius, radius]

    # Compute quaternion for rotation
    quat = fromto_to_quat(p1, p2)

    # Remove fromto and set new attributes
    del geom.attrib['fromto']
    geom.set('pos', f"{center[0]:.8f} {center[1]:.8f} {center[2]:.8f}")
    geom.set('size', f"{new_size[0]:.8f} {new_size[1]:.8f} {new_size[2]:.8f}")
    geom.set('quat', f"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}")

    # If there was an existing pos attribute, it's replaced; if not, that's fine.

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_fromto.py input.xml output.xml")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        tree = ET.parse(input_file)
    except Exception as e:
        print(f"Error parsing {input_file}: {e}")
        sys.exit(1)

    root = tree.getroot()

    # Find all geom elements in the whole tree
    for geom in root.iter('geom'):
        convert_geom(geom)

    # Write output with proper declaration
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Converted {input_file} -> {output_file}")

if __name__ == '__main__':
    main()