# PlayStation 4 Controller for Franka Emika Panda Robot

This directory contains scripts to control the Franka Emika Panda robot arm using a PlayStation 4 controller through the deoxys_control framework.

## Setup

### 1. Install Dependencies

```bash
pip install pyPS4Controller
```

### 2. Connect PS4 Controller

#### USB Connection (Recommended)
1. Connect your PS4 controller via USB cable
2. The controller should appear as `/dev/input/js0` (or js1, js2, etc.)
3. Check available controllers: `ls -la /dev/input/js*`

#### Bluetooth Connection (Alternative)
1. Install ds4drv: `sudo pip install ds4drv`
2. Run ds4drv: `sudo ds4drv`
3. Put controller in pairing mode (hold Share + PS button)
4. Use the `--use-ds4drv` flag when running scripts

### 3. Test Controller Connection

Before controlling the robot, test your controller connection:

```bash
python test_ps4_controller.py
```

If your controller is at a different device (e.g., js1):
```bash
python test_ps4_controller.py --interface /dev/input/js1
```

## Control Scheme

The control scheme is designed to be intuitive for robot manipulation:

### Translation (Robot Movement)
- **Right Joystick Left/Right**: Move robot left/right (X-axis)
- **Right Joystick Forward/Back**: Move robot forward/back (Y-axis)  
- **L2 Trigger**: Move robot down (Z-axis)
- **R2 Trigger**: Move robot up (Z-axis)

### Rotation (Robot Orientation)
- **Left Joystick Left/Right**: Rotate around X-axis (roll)
- **Left Joystick Forward/Back**: Rotate around Y-axis (pitch)
- **L1 Button**: Rotate counterclockwise around Z-axis (yaw)
- **R1 Button**: Rotate clockwise around Z-axis (yaw)

### Gripper Control
- **X Button**: Open gripper
- **Circle Button**: Close gripper

### Safety
- **Triangle Button**: Emergency stop / Reset (stops all motion)

## Usage

### Basic Robot Control

```bash
python run_deoxys_with_ps_controller.py
```

### Debug Mode

To see all controller inputs and robot commands in real-time:

```bash
python run_deoxys_with_ps_controller.py --debug
```

This will print:
- All button presses and releases
- Joystick movements with raw and normalized values
- Trigger values
- Final robot actions (translation, rotation, gripper commands)
- Controller state when there's movement

### Test Controller First

```bash
# Basic test
python test_ps4_controller.py

# Test with debug output
python test_ps4_controller.py --debug
```

### Controller Calibration & Range Finding

**✅ IMPORTANT: Your controller ranges have been auto-detected and calibrated!**

The robot control script now accounts for:
- **Joystick dead zones**: ±259 (was ±32767)
- **Trigger range**: -32252 to +32767 (was 0 to +32767)
- **Proper normalization**: Dead zone compensation and range mapping

To verify or recalibrate:

```bash
# Auto-calibrate (recommended - updates script automatically)
python auto_calibrate_ps4.py

# Find min/max values for ALL controller inputs
python find_ps4_controller_ranges.py

# Calibrate specific robot control inputs with real-time feedback
python calibrate_ps4_robot_controls.py
```

The calibration tools show:
- Real-time mapping from controller input to robot commands
- Range coverage analysis  
- Dead zone compensation
- Scaling factor verification
- Recommendations for optimal control

### Advanced Options

```bash
# Use different controller interface
python run_deoxys_with_ps_controller.py --controller-interface /dev/input/js1

# Use different robot configuration
python run_deoxys_with_ps_controller.py --interface-cfg config/my_robot.yml

# Use different controller type
python run_deoxys_with_ps_controller.py --controller-type OSC_POSITION

# Use with ds4drv
python run_deoxys_with_ps_controller.py --use-ds4drv

# Enable debug mode to see all inputs
python run_deoxys_with_ps_controller.py --debug

# Combine options
python run_deoxys_with_ps_controller.py --debug --controller-interface /dev/input/js1
```

## Available Controller Types

- `OSC_POSE`: Full 6-DOF position and orientation control (default)
- `OSC_POSITION`: Position-only control (3-DOF translation)
- `OSC_YAW`: Position + yaw rotation control
- `JOINT_IMPEDANCE`: Joint-level impedance control

## Troubleshooting

### Controller Not Found
```
❌ Controller not found at /dev/input/js0
```
**Solutions:**
1. Check connected controllers: `ls -la /dev/input/js*`
2. Try different interface: `--controller-interface /dev/input/js1`
3. Make sure controller is properly connected
4. Try with ds4drv: `--use-ds4drv`

### Permission Denied
```
❌ Permission denied accessing /dev/input/js0
```
**Solutions:**
1. Add user to input group: `sudo usermod -a -G input $USER`
2. Run with sudo (not recommended): `sudo python run_deoxys_with_ps_controller.py`
3. Set permissions: `sudo chmod 666 /dev/input/js0`

### Robot Connection Issues
```
❌ Failed to initialize robot interface
```
**Solutions:**
1. Check robot is powered on and connected
2. Verify interface configuration file exists
3. Ensure proper network connection to robot
4. Check deoxys_control installation

### Import Errors
```
❌ Import "pyPS4Controller.controller" could not be resolved
```
**Solutions:**
1. Install pyPS4Controller: `pip install pyPS4Controller`
2. Check Python environment is correct
3. Verify all deoxys dependencies are installed

## Safety Notes

⚠️ **Important Safety Guidelines:**

1. **Always keep the Triangle button accessible** - it's your emergency stop
2. **Start with small movements** to get familiar with the control scheme
3. **Ensure clear workspace** around the robot before operation
4. **Have physical emergency stop accessible** on the robot system
5. **Test controller response** with `test_ps4_controller.py` before robot operation
6. **Monitor robot movement** - be ready to stop if unexpected behavior occurs

## File Structure

- `run_deoxys_with_ps_controller.py` - Main robot control script (✅ calibrated)
- `test_ps4_controller.py` - Controller connection test script
- `auto_calibrate_ps4.py` - **NEW** Auto-calibration tool (updates script automatically)
- `find_ps4_controller_ranges.py` - Find min/max values for all controller inputs
- `calibrate_ps4_robot_controls.py` - Real-time calibration tool for robot controls
- `ps_controller_test.py` - Basic PS4 controller example (reference)

## Technical Details

The PS4 controller interface implements the same `get_controller_state()` method as the SpaceMouse, allowing it to work seamlessly with the existing `input2action()` function in deoxys_control. 

The controller values are now properly calibrated for your specific hardware:

**Original (generic) scaling:**
- Translation: `0.005` scale factor
- Rotation: `0.02` scale factor  
- Joystick values: Raw ÷ 32767 (assumed full range)
- Trigger values: Raw ÷ 32767 (assumed 0 to +32767)

**Updated (calibrated) scaling:**
- Translation: `0.005` scale factor (same)
- Rotation: `0.02` scale factor (same)
- Joystick values: **Dead zone compensated** (±259 dead zone removed)
- Trigger values: **Full range mapped** (-32252 to +32767 → 0.0 to 1.0)
- Thread-safe state management with proper normalization

**Key improvements:**
- ✅ **Dead zone elimination**: Small movements near center now register as 0.0
- ✅ **Full range utilization**: Uses actual controller limits, not theoretical ones
- ✅ **Proper trigger mapping**: Handles negative trigger values correctly  
- ✅ **Consistent scaling**: Same sensitivity across the full range of motion

---

# SpaceMouse Installation (Original Documentation)

## Installing Spacenvad drivers on Host

### Installing spacenavd without spnavcfg -> no GUI settings

sudo apt install libx11-dev libxi-dev libxtst-dev libgl1-mesa-dev

mkdir spacenav_install
cd spacenav_install

git clone https://github.com/FreeSpacenav/spacenavd
cd spacenavd
git checkout tags/v1.3.1
./configure
make
sudo make install
sudo ./setup_init

cd ..
git clone https://github.com/FreeSpacenav/libspnav
cd libspnav
git checkout tags/v1.2
./configure
make
sudo make install

### Running Space mouse in Deoxys (does not even need the spacenavd drivers in the container, but they are required in the host)

docker run -it --rm -v /cshome:/cshome -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --privileged deoxys

make sure spacenavd is running on the host before starting the container:

sudo spacenavd

In container:

export PYTHONPATH="/root/deoxys_control/deoxys"
python examples/run_deoxys_with_space_mouse.py --vendor-id 9583 --product-id 50746

To enable grasping with the left SpaceMouse Button:

Edit in examples/run_deoxys_with_space_mouse.py after the following add the last 3 lines

```
robot_interface.control(
    controller_type=controller_type,
    action=action,
    controller_cfg=controller_cfg,
)

robot_interface.gripper_control(
    action=grasp,
)
```
