#!/usr/bin/env bash
set -e  # stop if any script fails

python3 multi_recording.py
python3 raw_to_frames.py
python3 display_videos.py
