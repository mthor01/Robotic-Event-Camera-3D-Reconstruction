#!/usr/bin/env python3
"""
PS4 Controller Range Finder

This script helps you find the minimum and maximum values for all joysticks,
triggers, and buttons on your PS4 controller. This is useful for:
- Understanding the full range of your controller
- Calibrating scaling factors
- Debugging controller issues
- Ensuring proper normalization

Usage:
    python find_ps4_controller_ranges.py
    python find_ps4_controller_ranges.py --interface /dev/input/js1
    python find_ps4_controller_ranges.py --use-ds4drv

Instructions:
1. Run the script
2. Move all joysticks to their extreme positions
3. Press all triggers fully
4. Press all buttons
5. Press Ctrl+C when done to see the results
"""

import time
import signal
import sys
from collections import defaultdict
from pyPS4Controller.controller import Controller


class PS4RangeFinder(Controller):
    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)
        
        # Track min/max values for all inputs
        self.ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'count': 0})
        
        # Track button states
        self.button_states = {}
        
        print("PS4 Controller Range Finder")
        print("=" * 50)
        print("Instructions:")
        print("1. Move BOTH joysticks to ALL extreme positions (up, down, left, right)")
        print("2. Press ALL triggers FULLY")
        print("3. Press ALL buttons at least once")
        print("4. Press Ctrl+C when done to see the results")
        print()
        print("Starting data collection...")
        print("(Move joysticks and press buttons now)")
        print()
        
        # Set up signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C to show results and exit"""
        print("\n" + "=" * 50)
        print("CONTROLLER RANGE ANALYSIS RESULTS")
        print("=" * 50)
        self.print_results()
        sys.exit(0)
    
    def update_range(self, name, value):
        """Update min/max range for a given input"""
        self.ranges[name]['min'] = min(self.ranges[name]['min'], value)
        self.ranges[name]['max'] = max(self.ranges[name]['max'], value)
        self.ranges[name]['count'] += 1
    
    def print_results(self):
        """Print comprehensive results"""
        print("\nJOYSTICK RANGES:")
        print("-" * 30)
        joystick_inputs = [name for name in self.ranges.keys() if 'stick' in name.lower()]
        for name in sorted(joystick_inputs):
            data = self.ranges[name]
            range_val = data['max'] - data['min']
            midpoint = (data['max'] + data['min']) / 2
            print(f"{name:20}: Min={data['min']:6}, Max={data['max']:6}, "
                  f"Range={range_val:6}, Mid={midpoint:7.1f}, Samples={data['count']}")
        
        print("\nTRIGGER RANGES:")
        print("-" * 30)
        trigger_inputs = [name for name in self.ranges.keys() if 'trigger' in name.lower()]
        for name in sorted(trigger_inputs):
            data = self.ranges[name]
            range_val = data['max'] - data['min']
            print(f"{name:20}: Min={data['min']:6}, Max={data['max']:6}, "
                  f"Range={range_val:6}, Samples={data['count']}")
        
        print("\nBUTTON STATES DETECTED:")
        print("-" * 30)
        button_inputs = [name for name in self.ranges.keys() if 'button' in name.lower()]
        for name in sorted(button_inputs):
            data = self.ranges[name]
            print(f"{name:20}: Presses={data['count']}")
        
        print("\nNORMALIZATION ANALYSIS:")
        print("-" * 30)
        print("Expected joystick range: -32767 to +32767 (65534 total)")
        print("Expected trigger range: 0 to +32767")
        print()
        
        # Analyze joystick normalization
        for name in sorted(joystick_inputs):
            data = self.ranges[name]
            if data['count'] > 0:
                range_val = data['max'] - data['min']
                expected_range = 65534  # -32767 to +32767
                coverage = (range_val / expected_range) * 100 if expected_range > 0 else 0
                print(f"{name:20}: Coverage={coverage:5.1f}% of expected range")
        
        # Analyze trigger normalization  
        for name in sorted(trigger_inputs):
            data = self.ranges[name]
            if data['count'] > 0:
                range_val = data['max'] - data['min']
                expected_range = 32767  # 0 to +32767
                coverage = (range_val / expected_range) * 100 if expected_range > 0 else 0
                print(f"{name:20}: Coverage={coverage:5.1f}% of expected range")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 30)
        
        # Check joystick coverage
        joystick_issues = []
        for name in sorted(joystick_inputs):
            data = self.ranges[name]
            if data['count'] == 0:
                joystick_issues.append(f"❌ {name}: No input detected")
            else:
                range_val = data['max'] - data['min']
                coverage = (range_val / 65534) * 100
                if coverage < 80:
                    joystick_issues.append(f"⚠️  {name}: Low coverage ({coverage:.1f}%) - move to extremes")
                elif abs(data['min'] + 32767) > 1000 or abs(data['max'] - 32767) > 1000:
                    joystick_issues.append(f"⚠️  {name}: Range offset detected")
                else:
                    print(f"✅ {name}: Good range coverage")
        
        # Check trigger coverage
        trigger_issues = []
        for name in sorted(trigger_inputs):
            data = self.ranges[name]
            if data['count'] == 0:
                trigger_issues.append(f"❌ {name}: No input detected")
            else:
                if data['max'] < 30000:
                    trigger_issues.append(f"⚠️  {name}: May not be fully pressed (max={data['max']})")
                else:
                    print(f"✅ {name}: Good range coverage")
        
        # Print issues
        for issue in joystick_issues + trigger_issues:
            print(issue)
        
        if not joystick_issues and not trigger_issues:
            print("✅ All inputs look good!")
        
        print(f"\nTotal unique inputs detected: {len(self.ranges)}")
        print(f"Total samples collected: {sum(data['count'] for data in self.ranges.values())}")
    
    # Left joystick callbacks
    def on_L3_up(self, value):
        self.update_range("left_stick_up", value)
        print(f"L3 UP: {value:6} (range: {self.ranges['left_stick_up']['min']:6} to {self.ranges['left_stick_up']['max']:6})")
    
    def on_L3_down(self, value):
        self.update_range("left_stick_down", value)
        print(f"L3 DOWN: {value:6} (range: {self.ranges['left_stick_down']['min']:6} to {self.ranges['left_stick_down']['max']:6})")
    
    def on_L3_left(self, value):
        self.update_range("left_stick_left", value)
        print(f"L3 LEFT: {value:6} (range: {self.ranges['left_stick_left']['min']:6} to {self.ranges['left_stick_left']['max']:6})")
    
    def on_L3_right(self, value):
        self.update_range("left_stick_right", value)
        print(f"L3 RIGHT: {value:6} (range: {self.ranges['left_stick_right']['min']:6} to {self.ranges['left_stick_right']['max']:6})")
    
    # Right joystick callbacks
    def on_R3_up(self, value):
        self.update_range("right_stick_up", value)
        print(f"R3 UP: {value:6} (range: {self.ranges['right_stick_up']['min']:6} to {self.ranges['right_stick_up']['max']:6})")
    
    def on_R3_down(self, value):
        self.update_range("right_stick_down", value)
        print(f"R3 DOWN: {value:6} (range: {self.ranges['right_stick_down']['min']:6} to {self.ranges['right_stick_down']['max']:6})")
    
    def on_R3_left(self, value):
        self.update_range("right_stick_left", value)
        print(f"R3 LEFT: {value:6} (range: {self.ranges['right_stick_left']['min']:6} to {self.ranges['right_stick_left']['max']:6})")
    
    def on_R3_right(self, value):
        self.update_range("right_stick_right", value)
        print(f"R3 RIGHT: {value:6} (range: {self.ranges['right_stick_right']['min']:6} to {self.ranges['right_stick_right']['max']:6})")
    
    # Trigger callbacks
    def on_L2_press(self, value):
        self.update_range("L2_trigger", value)
        print(f"L2 TRIGGER: {value:6} (range: {self.ranges['L2_trigger']['min']:6} to {self.ranges['L2_trigger']['max']:6})")
    
    def on_R2_press(self, value):
        self.update_range("R2_trigger", value)
        print(f"R2 TRIGGER: {value:6} (range: {self.ranges['R2_trigger']['min']:6} to {self.ranges['R2_trigger']['max']:6})")
    
    # Button callbacks
    def on_x_press(self):
        self.update_range("X_button", 1)
        print("X button pressed ✓")
    
    def on_circle_press(self):
        self.update_range("Circle_button", 1)
        print("Circle button pressed ✓")
    
    def on_triangle_press(self):
        self.update_range("Triangle_button", 1)
        print("Triangle button pressed ✓")
    
    def on_square_press(self):
        self.update_range("Square_button", 1)
        print("Square button pressed ✓")
    
    def on_L1_press(self):
        self.update_range("L1_button", 1)
        print("L1 button pressed ✓")
    
    def on_R1_press(self):
        self.update_range("R1_button", 1)
        print("R1 button pressed ✓")
    
    def on_up_arrow_press(self):
        self.update_range("Up_arrow_button", 1)
        print("Up arrow pressed ✓")
    
    def on_down_arrow_press(self):
        self.update_range("Down_arrow_button", 1)
        print("Down arrow pressed ✓")
    
    def on_left_arrow_press(self):
        self.update_range("Left_arrow_button", 1)
        print("Left arrow pressed ✓")
    
    def on_right_arrow_press(self):
        self.update_range("Right_arrow_button", 1)
        print("Right arrow pressed ✓")
    
    def on_L3_press(self):
        self.update_range("L3_click_button", 1)
        print("L3 click pressed ✓")
    
    def on_R3_press(self):
        self.update_range("R3_click_button", 1)
        print("R3 click pressed ✓")
    
    def on_options_press(self):
        self.update_range("Options_button", 1)
        print("Options button pressed ✓")
    
    def on_share_press(self):
        self.update_range("Share_button", 1)
        print("Share button pressed ✓")
    
    def on_playstation_button_press(self):
        self.update_range("PlayStation_button", 1)
        print("PlayStation button pressed ✓")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find PS4 controller input ranges")
    parser.add_argument("--interface", type=str, default="/dev/input/js0", 
                        help="Controller device interface")
    parser.add_argument("--use-ds4drv", action="store_true", 
                        help="Connect using ds4drv")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    try:
        controller = PS4RangeFinder(
            interface=args.interface,
            connecting_using_ds4drv=args.use_ds4drv
        )
        print(f"Connected to controller at {args.interface}")
        print(f"Collection will timeout after {args.timeout} seconds")
        print()
        
        # Listen for input
        controller.listen(timeout=args.timeout)
        
    except KeyboardInterrupt:
        # This is handled by the signal handler
        pass
    except FileNotFoundError:
        print(f"❌ Controller not found at {args.interface}")
        print("Available devices:")
        import os
        try:
            devices = [f for f in os.listdir("/dev/input/") if f.startswith("js")]
            for device in devices:
                print(f"  /dev/input/{device}")
        except:
            print("  Could not list devices")
        print("\nTry:")
        print(f"  python {sys.argv[0]} --interface /dev/input/js1")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure pyPS4Controller is installed: pip install pyPS4Controller")
        print("2. Check controller connection: ls -la /dev/input/js*")
        print("3. Try with ds4drv: --use-ds4drv")


if __name__ == "__main__":
    main()
