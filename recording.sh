#!/usr/bin/env bash
set -e

# Allow X11 access for root in containers (needed for pynput / GUI)
xhost +local:root >/dev/null

# Start containers in background
docker compose up -d

# Run publisher (openvla)
docker exec -i openvla bash -lc "
  python project/franka_pipeline/my_main.py --simulated-robot
" &


# Small delay so PUB binds before SUB connects
sleep 1

# Run subscriber (metavision)
docker exec -it metavision bash -lc "
  python3 project/zmq_pose_receiver.py
"

# Cleanup on exit
trap 'docker compose down; xhost -local:root >/dev/null' EXIT
