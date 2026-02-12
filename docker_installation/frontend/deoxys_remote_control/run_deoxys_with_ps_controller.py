import argparse
import time
import threading
import numpy as np

from pyPS4Controller.controller import Controller

from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


class PS4Controller(Controller):
    """PlayStation 4 controller interface for robot control"""
    
    def __init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, **kwargs):
        Controller.__init__(self, interface=interface, connecting_using_ds4drv=connecting_using_ds4drv, **kwargs)
        
        # Control state variables
        self.control_state = {
            'dpos': np.array([0.0, 0.0, 0.0]),      # x, y, z translation
            'raw_drotation': np.array([0.0, 0.0, 0.0]),  # roll, pitch, yaw rotation
            'grasp': False,
            'reset': False
        }
        
        # Joystick values (normalized to -1 to 1)
        self.left_joystick_x = 0.0   # Left-right rotation (roll)
        self.left_joystick_y = 0.0   # Front-back rotation (pitch)
        self.right_joystick_x = 0.0  # Left-right translation
        self.right_joystick_y = 0.0  # Front-back translation
        
        # Trigger values for z translation and yaw rotation
        self.l2_value = 0.0  # Down movement
        self.r2_value = 0.0  # Up movement
        self.l1_pressed = False  # Negative yaw rotation
        self.r1_pressed = False  # Positive yaw rotation
        
        # Gripper control
        self.x_pressed = False  # Open gripper
        self.circle_pressed = False  # Close gripper
        
        # Scaling factors
        self.translation_scale = 0.005  # Scale factor for translation
        self.rotation_scale = 0.005     # Scale factor for rotation
        
        # Thread lock for state updates
        self._lock = threading.Lock()
        
        print("PS4 Controller initialized")
        print("Control scheme:")
        print("  Right stick: Move robot left-right and front-back")
        print("  L2/R2: Move robot down/up")
        print("  Left stick: Rotate robot around x and y axes")
        print("  L1/R1: Rotate robot around z axis")
        print("  X button: Open gripper")
        print("  Circle button: Close gripper")
        print("  Triangle: Reset (stop all motion)")
    
    def get_controller_state(self):
        """Return the current controller state in the format expected by input2action"""
        with self._lock:

            print("DEBUG: Joystick Values before sending command")
            print("self.right_joystick_x:", self.right_joystick_x)
            print("self.right_joystick_y:", self.right_joystick_y)
            print("self.r2_value:", self.r2_value)
            print("self.l2_value:", self.l2_value)

            print("self.left_joystick_x:", self.left_joystick_x)
            print("self.left_joystick_y:", self.left_joystick_y)
            print("self.l1_pressed:", self.l1_pressed)
            print("self.r1_pressed:", self.r1_pressed)


            # Calculate z translation from triggers
            z_translation = (self.l2_value - self.r2_value) * self.translation_scale
            
            # Calculate yaw rotation from shoulder buttons
            yaw_rotation = 0.0
            if self.r1_pressed:
                yaw_rotation = -1 * self.rotation_scale
            if self.l1_pressed:
                yaw_rotation = 1 * self.rotation_scale


            # Update control state
            self.control_state['dpos'] = np.array([
                self.right_joystick_y * self.translation_scale,    # y (front-back, inverted)
                self.right_joystick_x * self.translation_scale,    # x (left-right)
                z_translation                                      # z (up-down)
            ])
            
            self.control_state['raw_drotation'] = np.array([
                self.left_joystick_y * self.rotation_scale,    # pitch (around y-axis, inverted)
                self.left_joystick_x * self.rotation_scale,    # roll (around x-axis)
                yaw_rotation                                   # yaw (around z-axis)
            ])
            
            # Gripper control - close if circle pressed, open if x pressed
            if self.circle_pressed:
                self.control_state['grasp'] = True
            elif self.x_pressed:
                self.control_state['grasp'] = False
            
            return {
                'dpos': self.control_state['dpos'].copy(),
                'rotation': np.eye(3),  # Not used in current implementation
                'raw_drotation': self.control_state['raw_drotation'].copy(),
                'grasp': self.control_state['grasp'],
                'reset': self.control_state['reset']
            }
    
    # Left joystick callbacks (rotation control)
    def on_L3_up(self, value):
        with self._lock:
            self.left_joystick_y = value / 32767.0
    
    def on_L3_down(self, value):
        with self._lock:
            self.left_joystick_y = value / 32767.0
    
    def on_L3_left(self, value):
        with self._lock:
            self.left_joystick_x = value / 32767.0
    
    def on_L3_right(self, value):
        with self._lock:
            self.left_joystick_x = value / 32767.0
    
    def on_L3_y_at_rest(self):
        with self._lock:
            self.left_joystick_y = 0.0
    
    def on_L3_x_at_rest(self):
        with self._lock:
            self.left_joystick_x = 0.0
    
    # Right joystick callbacks (translation control)
    def on_R3_up(self, value):
        with self._lock:
            self.right_joystick_y = value / 32767.0
    
    def on_R3_down(self, value):
        with self._lock:
            self.right_joystick_y = value / 32767.0
    
    def on_R3_left(self, value):
        with self._lock:
            self.right_joystick_x = value / 32767.0
    
    def on_R3_right(self, value):
        with self._lock:
            self.right_joystick_x = value / 32767.0
    
    def on_R3_y_at_rest(self):
        with self._lock:
            self.right_joystick_y = 0.0
    
    def on_R3_x_at_rest(self):
        with self._lock:
            self.right_joystick_x = 0.0
    
    # Trigger callbacks (z translation)
    # L2/R2 go from -32767.0 to 32767.0, so we need to normalize differently
    def on_L2_press(self, value):
        with self._lock:
            self.l2_value = (value + 32767.0 )/ (32767.0 * 2)
    
    def on_L2_release(self):
        with self._lock:
            self.l2_value = 0.0
    
    def on_R2_press(self, value):
        with self._lock:
            self.r2_value = (value + 32767.0 )/ (32767.0 * 2)
    
    def on_R2_release(self):
        with self._lock:
            self.r2_value = 0.0
    
    # Shoulder button callbacks (yaw rotation)
    def on_L1_press(self):
        with self._lock:
            self.l1_pressed = True
    
    def on_L1_release(self):
        with self._lock:
            self.l1_pressed = False
    
    def on_R1_press(self):
        with self._lock:
            self.r1_pressed = True
    
    def on_R1_release(self):
        with self._lock:
            self.r1_pressed = False
    
    # Gripper control callbacks
    def on_x_press(self):
        with self._lock:
            self.x_pressed = True
        print("Gripper: Open")
    
    def on_x_release(self):
        with self._lock:
            self.x_pressed = False
    
    def on_circle_press(self):
        with self._lock:
            self.circle_pressed = True
        print("Gripper: Close")
    
    def on_circle_release(self):
        with self._lock:
            self.circle_pressed = False
    
    # Reset callback
    def on_triangle_press(self):
        with self._lock:
            self.control_state['reset'] = True
            # Reset all control values
            self.left_joystick_x = 0.0
            self.left_joystick_y = 0.0
            self.right_joystick_x = 0.0
            self.right_joystick_y = 0.0
            self.l2_value = 0.0
            self.r2_value = 0.0
            self.l1_pressed = False
            self.r1_pressed = False
        print("Reset: All motion stopped")
    
    def on_triangle_release(self):
        with self._lock:
            self.control_state['reset'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="config/charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--controller-interface", type=str, default="/dev/input/js0", 
                        help="Controller device interface")
    parser.add_argument("--use-ds4drv", action="store_true", 
                        help="Connect using ds4drv")
    
    args = parser.parse_args()
    
    # Initialize PS4 controller
    try:
        device = PS4Controller(
            interface=args.controller_interface, 
            connecting_using_ds4drv=args.use_ds4drv
        )
        print(f"PS4 Controller connected at {args.controller_interface}")
    except Exception as e:
        print(f"Failed to connect PS4 controller: {e}")
        print("Make sure the controller is connected and available at the specified interface")
        print("You can check available devices with: ls -la /dev/input/js*")
        return
    
    # Start controller in background thread
    controller_thread = threading.Thread(target=device.listen, daemon=True)
    controller_thread.start()
    
    # Allow some time for controller to initialize
    time.sleep(1)
    
    # Initialize robot interface
    try:
        robot_interface = FrankaInterface(args.interface_cfg, use_visualizer=False)
        print("Robot interface initialized")
    except Exception as e:
        print(f"Failed to initialize robot interface: {e}")
        return
    
    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    
    robot_interface._state_buffer = []
    
    print(f"Starting robot control with {controller_type} controller")
    print("Press Triangle to reset/stop all motion")
    print("Use Ctrl+C to exit")
    
    try:
        for i in range(10000):  # Run for a long time
            start_time = time.time_ns()
            
            # Get action from PS4 controller
            try:
                action, grasp = input2action(
                    device=device,
                    controller_type=controller_type,
                )
                
                if action is None:
                    # Controller reset was triggered
                    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    print("Reset triggered - stopping all motion")
                
            except Exception as e:
                logger.warning(f"Error getting controller input: {e}")
                # Default to safe action (no movement)
                action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                grasp = 1.0
            

            # TODO remove
            # action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

            # Send action to robot
            try:
                robot_interface.control(
                    controller_type=controller_type,
                    action=action,
                    controller_cfg=controller_cfg,
                )
                
                # Control gripper
                robot_interface.gripper_control(
                    action=grasp,
                )
            except Exception as e:
                logger.error(f"Error sending commands to robot: {e}")
                # Try to send safe stop command
                try:
                    robot_interface.control(
                        controller_type=controller_type,
                        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        controller_cfg=controller_cfg,
                    )
                except:
                    pass  # Final attempt failed, will exit
                break
            
            end_time = time.time_ns()
            logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Send termination command
        robot_interface.control(
            controller_type=controller_type,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
            controller_cfg=controller_cfg,
            termination=True,
        )
        
        robot_interface.close()
        print("Robot interface closed")
        
        # Check if there are any missing state frames
        for (state, next_state) in zip(
            robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
        ):
            if (next_state.frame - state.frame) > 1:
                print(f"Missing frames: {state.frame} -> {next_state.frame}")


if __name__ == "__main__":
    main()
