#!/usr/bin/env bash
# Run the minimal detector using tiny_venv if available, otherwise use system python.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <video_path> [win_size]"
  exit 1
fi
VIDEO="$1"
WIN=${2:-5}
if [ -x "./tiny_venv/bin/python" ]; then
  ./tiny_venv/bin/python ex1_minimal.py "$VIDEO" "$WIN"
else
  python3 ex1_minimal.py "$VIDEO" "$WIN"
fi

