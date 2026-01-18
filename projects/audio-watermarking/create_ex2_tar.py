#!/usr/bin/env python3
"""
Create a tar archive `ex2.tar` containing:
 - task1_watermark.py
 - task2_extract_functions.py
 - task3_speedup.py
 - the `outputs/` directory (all files under it)

Writes `ex2.tar` into the same directory as this script.
"""
import os
import tarfile

BASE = os.path.dirname(__file__)
files_to_add = [
    'task1_watermark.py',
    'task2_extract_functions.py',
    'task3_speedup.py',
]
outputs_dir = os.path.join(BASE, 'outputs')
out_tar = os.path.join(BASE, 'ex2.tar')

with tarfile.open(out_tar, 'w') as tf:
    # add listed files if present
    for fn in files_to_add:
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            tf.add(p, arcname=fn)
        else:
            print(f"Warning: {fn} not found, skipping")
    # add outputs directory recursively if present
    if os.path.exists(outputs_dir):
        for root, dirs, files in os.walk(outputs_dir):
            for f in files:
                full = os.path.join(root, f)
                # preserve relative path under outputs/
                rel = os.path.relpath(full, BASE)
                tf.add(full, arcname=rel)
    else:
        print('Warning: outputs/ directory not found, skipping')

print('Wrote', out_tar)
