"""
ZMQ pose and event receiver for event camera 3D reconstruction.

Subscribes to the ZMQ PUB socket from my_main.py and receives:
  - topic b"pose": robot end-effector pose, joint positions, gripper state
  - topic b"event": episode control events

Prints received data periodically to verify connection.
"""

import os
import time
from pathlib import Path

import zmq
import msgpack
import numpy as np


def main(
    zmq_connect: str = None,
    print_interval: float = 1.0,
) -> None:
    """
    Receive and print ZMQ pose/event data.
    
    Args:
        zmq_connect: ZMQ address to connect to (default from POSE_PUB_BIND env var or Docker service name)
        print_interval: How often to print received data (seconds)
    """
    if zmq_connect is None:
        zmq_connect = os.getenv("POSE_PUB_CONNECT") or os.getenv("POSE_PUB_BIND") or "tcp://openvla:5556"

    
    print(f"Connecting to ZMQ: {zmq_connect}")
    
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    
    # Subscribe to both topics
    sub.setsockopt(zmq.SUBSCRIBE, b"pose")
    sub.setsockopt(zmq.SUBSCRIBE, b"event")
    sub.setsockopt(zmq.RCVHWM, 100)
    
    sub.connect(zmq_connect)
    print("Connected. Waiting for data...")
    
    last_print_t = 0.0
    pose_count = 0
    event_count = 0
    
    try:
        while True:
            topic, payload = sub.recv_multipart()
            
            try:
                msg = msgpack.unpackb(payload, raw=False)
                
                if topic == b"pose":
                    pose_count += 1
                    now = time.time()
                    
                    # Print occasionally
                    if now - last_print_t >= print_interval:
                        print(f"\n--- POSE (#{pose_count}) ---")
                        print(f"  Timestamp: {msg.get('t_ns', 'N/A')} ns")
                        print(f"  Episode: {msg.get('ep', 'N/A')}, Step: {msg.get('step', 'N/A')}")
                        
                        if "ee_T" in msg:
                            ee_T = np.array(msg["ee_T"]).reshape(4, 4)
                            print(f"  EE Pose (4x4):\n{ee_T}")
                        
                        if "q" in msg:
                            q = np.array(msg["q"])
                            print(f"  Joint positions: {q}")
                        
                        if "gripper_q" in msg:
                            print(f"  Gripper q: {msg['gripper_q']}")
                        
                        last_print_t = now
                
                elif topic == b"event":
                    event_count += 1
                    print(f"\n--- EVENT (#{event_count}) ---")
                    print(f"  Type: {msg.get('type', 'N/A')}")
                    print(f"  Episode: {msg.get('ep', 'N/A')}")
                    print(f"  Timestamp: {msg.get('t_ns', 'N/A')} ns")
                    if "step" in msg:
                        print(f"  Step: {msg['step']}")
            
            except Exception as e:
                print(f"Error decoding message: {e}")
    
    except zmq.error.Again:
        # No data available yet, sleep a bit
        time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sub.close()
        ctx.term()
        print(f"Received {pose_count} pose messages and {event_count} event messages")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Receive ZMQ pose/event data from Franka pipeline"
    )
    parser.add_argument(
        "--zmq-connect",
        default=None,
        help='ZMQ SUB connect address (default: env POSE_PUB_BIND or "tcp://localhost:5556")',
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="How often to print pose data (seconds, default: 1.0)",
    )
    
    args = parser.parse_args()
    main(zmq_connect=args.zmq_connect, print_interval=args.print_interval)
