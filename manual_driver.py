import msgParser
import carState
import carControl
import time
import os
import csv
from datetime import datetime
import threading

class TelemetryCollector:
    """High-performance telemetry logging with minimal overhead - includes control inputs for ML training"""
    def __init__(self):
        self.telemetry_dir = 'telemetry_data'
        os.makedirs(self.telemetry_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.telemetry_file = os.path.join(self.telemetry_dir, f'telemetry_{timestamp}.csv')
        
        # Headers for car state data
        self.headers = [
            'timestamp', 'distance_raced', 'current_lap_time', 'last_lap_time',
            'race_position', 'speed_x', 'speed_y', 'speed_z', 'angle',
            'track_position', 'rpm', 'gear', 'fuel', 'damage', 'distance_from_start'
        ]
        
        # Add track sensor headers (19 sensors from -90 to +90 degrees)
        self.headers += [f'track_sensor_{angle}' for angle in range(-90, 91, 10)]
        
        # Add opponent sensor headers (36 opponent distances)
        self.headers += [f'opponent_{i*10}' for i in range(36)]
        
        # Add wheel speed headers
        self.headers += [f'wheel_speed_{wheel}' for wheel in ['front_left', 'front_right', 'rear_left', 'rear_right']]
        
        # *** CRUCIAL FOR ML TRAINING: Add control input headers ***
        self.headers += [
            'control_steer',      # Steering input (-1 to +1)
            'control_accel',      # Acceleration input (0 to 1)
            'control_brake',      # Brake input (0 to 1) 
            'control_gear',       # Gear selection (-1, 1-6)
            'reverse_mode',       # Whether reverse was requested (boolean)
            'auto_transmission',  # Whether auto transmission is enabled (boolean)
            'key_left',          # Left arrow pressed (boolean)
            'key_right',         # Right arrow pressed (boolean)
            'key_up',            # Up arrow pressed (boolean)
            'key_down',          # Down arrow pressed (boolean)
            'key_space'          # Spacebar pressed (boolean)
        ]
        
        # Keep file open for performance
        self.file = open(self.telemetry_file, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.headers)
        
        print(f"Telemetry initialized: {self.telemetry_file}")
        print(f"Total data columns: {len(self.headers)} (includes control inputs for ML training)")
        
    def log_telemetry(self, car_state, control_inputs):
        """Log both car state AND human control inputs for ML training"""
        try:
            # Car state data
            data = [
                datetime.now().strftime('%H:%M:%S.%f')[:-3],
                car_state.getDistRaced() or 0,
                car_state.getCurLapTime() or 0,
                car_state.getLastLapTime() or 0,
                car_state.getRacePos() or 0,
                car_state.getSpeedX() or 0,
                car_state.getSpeedY() or 0,
                car_state.getSpeedZ() or 0,
                car_state.getAngle() or 0,
                car_state.getTrackPos() or 0,
                car_state.getRpm() or 0,
                car_state.getGear() or 1,
                car_state.getFuel() or 0,
                car_state.getDamage() or 0,
                car_state.getDistFromStart() or 0,
            ]
            
            # Add track sensors (19 values)
            track_data = car_state.getTrack() or [0.0] * 19
            data.extend(track_data[:19])
            
            # Add opponent sensors (36 values)
            opponent_data = car_state.getOpponents() or [200.0] * 36  # Default to far distance
            data.extend(opponent_data[:36])
            
            # Add wheel speeds (4 values)
            wheel_data = car_state.getWheelSpinVel() or [0.0] * 4
            data.extend(wheel_data[:4])
            
            # *** ADD CONTROL INPUTS - This is what makes it ML training data! ***
            data.extend([
                control_inputs['steer'],           # Current steering command
                control_inputs['accel'],           # Current acceleration command  
                control_inputs['brake'],           # Current brake command
                control_inputs['gear'],            # Current gear selection
                control_inputs['reverse_mode'],    # Reverse mode flag
                control_inputs['auto_transmission'], # Auto transmission flag
                control_inputs['keys']['left'],    # Raw key states
                control_inputs['keys']['right'],
                control_inputs['keys']['up'],
                control_inputs['keys']['down'],
                control_inputs['keys']['space']
            ])
            
            self.writer.writerow(data)
            self.file.flush()  # Immediate write for safety
            
        except Exception as e:
            print(f"Telemetry error: {e}")

    def __del__(self):
        """Ensure proper file cleanup"""
        if hasattr(self, 'file'):
            self.file.close()

class Driver:
    """Ultra-responsive manual driver with automatic transmission for optimal AI training data"""
    def __init__(self, stage, auto_transmission=True):
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Initialize telemetry (low overhead)
        self.telemetry = TelemetryCollector()
        
        # Control state
        self.current_steer = 0.0
        self.current_accel = 0.0
        self.current_brake = 0.0
        self.current_gear = 1
        self.tick_counter = 0
        self.reverse_requested = False
        self.auto_transmission = auto_transmission
        
        # Automatic transmission parameters (from C++ SimpleDriver)
        self.gear_up_rpm = [5000, 6000, 6000, 6500, 7000, 0]    # RPM thresholds for upshifting
        self.gear_down_rpm = [0, 2500, 3000, 3000, 3500, 3500]  # RPM thresholds for downshifting
        
        # Key states for proper handling
        self.keys_pressed = {
            'left': False,
            'right': False,
            'up': False,
            'down': False,
            'space': False
        }
        
        # Input handling
        self._init_input_system()
        transmission_mode = "AUTOMATIC" if auto_transmission else "MANUAL"
        print(f"Manual driver ready (Stage: {stage}) - {transmission_mode} transmission")
        print("Controls: Arrow keys for steering/throttle/brake-reverse, Space for emergency brake")
        if auto_transmission:
            print("Transmission: AUTOMATIC - Focus on steering, throttle, and braking")
            print("Down Arrow: Brake when moving, Reverse when stopped")
        else:
            print("Gears: 1-6 for forward gears")
            print("Down Arrow: Brake when moving, Reverse when stopped")
        print("*** RECORDING CONTROL INPUTS FOR ML TRAINING ***")

    def _init_input_system(self):
        """Set up event-based keyboard controls"""
        try:
            import keyboard
            self.keyboard = keyboard
            
            # Set up keyboard event handlers
            self._setup_keyboard_handlers()
            
            print("Keyboard controls initialized successfully")
            
        except ImportError:
            print("ERROR: 'keyboard' module not available. Install with: pip install keyboard")
            print("Note: You may need to run as administrator/root for keyboard access")
            self.keyboard = None

    def _setup_keyboard_handlers(self):
        """Setup all keyboard event handlers"""
        if not self.keyboard:
            return
            
        # Steering controls
        self.keyboard.on_press_key('left', lambda _: self._on_key_press('left'))
        self.keyboard.on_release_key('left', lambda _: self._on_key_release('left'))
        self.keyboard.on_press_key('right', lambda _: self._on_key_press('right'))
        self.keyboard.on_release_key('right', lambda _: self._on_key_release('right'))
        
        # Throttle/Brake controls
        self.keyboard.on_press_key('up', lambda _: self._on_key_press('up'))
        self.keyboard.on_release_key('up', lambda _: self._on_key_release('up'))
        self.keyboard.on_press_key('down', lambda _: self._on_key_press('down'))
        self.keyboard.on_release_key('down', lambda _: self._on_key_release('down'))
        self.keyboard.on_press_key('space', lambda _: self._on_key_press('space'))
        self.keyboard.on_release_key('space', lambda _: self._on_key_release('space'))
        
        # Gear controls - only if manual transmission
        if not self.auto_transmission:
            for gear in range(1, 7):
                self.keyboard.on_press_key(str(gear), lambda e, g=gear: self._set_gear(g))

    def _on_key_press(self, key):
        """Handle key press events"""
        self.keys_pressed[key] = True
        
        if key == 'left':
            self.current_steer = 0.6   # Turn left (fixed steering direction)
        elif key == 'right':
            self.current_steer = -0.6  # Turn right (fixed steering direction)
        elif key == 'up':
            # Forward acceleration - exit reverse mode
            if self.reverse_requested:
                self.reverse_requested = False
                self.current_gear = 1
                print("Forward gear engaged")
            self.current_accel = 1.0
            self.current_brake = 0.0
        elif key == 'down':
            # Smart down arrow: brake when moving forward, reverse when stopped/slow
            current_speed = abs(self.state.getSpeedX() or 0)
            
            if current_speed > 5.0:  # Moving fast - apply brake
                self.current_brake = 0.8
                self.current_accel = 0.0
                print(f"Braking (speed: {current_speed:.1f})")
            else:  # Stopped or very slow - engage reverse
                if not self.reverse_requested:
                    self.reverse_requested = True
                    self.current_gear = -1
                    print("Reverse gear engaged")
                # Apply reverse throttle
                self.current_accel = 0.6  # Reverse acceleration
                self.current_brake = 0.0
        elif key == 'space':
            self.current_brake = 1.0   # Emergency brake
            self.current_accel = 0.0

    def _on_key_release(self, key):
        """Handle key release events"""
        self.keys_pressed[key] = False
        
        if key in ['left', 'right']:
            # Only center steering if no steering keys are pressed
            if not (self.keys_pressed['left'] or self.keys_pressed['right']):
                self.current_steer = 0.0
        elif key == 'up':
            self.current_accel = 0.0
        elif key == 'down':
            # Reset the down key mode when released
            if hasattr(self, '_down_mode_set'):
                delattr(self, '_down_mode_set')
            
            # Stop all acceleration/braking
            self.current_brake = 0.0
            self.current_accel = 0.0
            
            # Note: Don't automatically exit reverse mode - let it stay until up arrow is pressed
            print("Down arrow released")
            
        elif key == 'space':
            # Only release brake if down arrow isn't pressed
            if not self.keys_pressed['down']:
                self.current_brake = 0.0

    def _set_gear(self, gear):
        """Set forward gear (1-6) - only for manual transmission"""
        if not self.auto_transmission and 1 <= gear <= 6:
            self.current_gear = gear
            print(f"Gear {gear} selected")

    def _get_optimal_gear(self):
        """Calculate optimal gear based on RPM (like C++ SimpleDriver)"""
        if not self.auto_transmission:
            return self.current_gear
            
        # Don't change gear if we're in reverse mode
        if self.reverse_requested:
            return -1
            
        current_gear = self.current_gear
        rpm = self.state.getRpm() or 0
        
        # Handle special cases
        if current_gear < 1:
            return 1
            
        # Check for upshift (if not in highest gear and RPM is high enough)
        if current_gear < 6 and rpm >= self.gear_up_rpm[current_gear - 1]:
            return current_gear + 1
            
        # Check for downshift (if not in lowest gear and RPM is low enough)
        if current_gear > 1 and rpm <= self.gear_down_rpm[current_gear - 1]:
            return current_gear - 1
            
        # Keep current gear
        return current_gear

    def _get_control_inputs_for_telemetry(self):
        """Package current control inputs for telemetry logging"""
        return {
            'steer': self.current_steer,
            'accel': self.current_accel,
            'brake': self.current_brake,
            'gear': self.current_gear,
            'reverse_mode': self.reverse_requested,
            'auto_transmission': self.auto_transmission,
            'keys': self.keys_pressed.copy()  # Copy to avoid reference issues
        }

    def init(self):
        """Initialize sensors - must return the init string format expected by TORCS"""
        # Set up sensor angles like the C++ version
        angles = [-90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 75, 90]
        
        # Build the init message in the expected format
        init_dict = {'init': [str(angle) for angle in angles]}
        return self.parser.stringify(init_dict)

    def drive(self, msg):
        """Main control loop - optimized for minimal latency"""
        try:
            # 1. Update car state
            self.state.setFromMsg(msg)
            
            # 2. Handle gear selection
            if self.reverse_requested:
                # Force reverse gear when requested
                self.current_gear = -1
            elif self.auto_transmission and not self.reverse_requested:
                # Automatic gear selection for forward gears only
                optimal_gear = self._get_optimal_gear()
                if optimal_gear != self.current_gear and optimal_gear > 0:
                    self.current_gear = optimal_gear
            
            # 3. *** ENHANCED TELEMETRY: Log both state AND control inputs ***
            if self.tick_counter % 5 == 0:  # Reduced frequency for performance
                control_inputs = self._get_control_inputs_for_telemetry()
                self.telemetry.log_telemetry(self.state, control_inputs)
            self.tick_counter += 1
            
            # 4. Apply controls (steering is now correctly mapped)
            self.control.setSteer(self.current_steer)
            self.control.setAccel(self.current_accel)
            self.control.setBrake(self.current_brake)
            self.control.setGear(self.current_gear)
            
            # 5. Auto-center steering with gradual decay
            if self.keyboard and not (self.keys_pressed.get('left', False) or self.keys_pressed.get('right', False)):
                self.current_steer *= 0.8  # Gradual centering
                if abs(self.current_steer) < 0.02:
                    self.current_steer = 0.0
            
            # 6. Display status every 30 ticks
            if self.tick_counter % 30 == 0:
                gear_name = "REVERSE" if self.current_gear == -1 else str(self.current_gear)
                speed = self.state.getSpeedX() or 0
                rpm = self.state.getRpm() or 0
                transmission_indicator = " (AUTO)" if self.auto_transmission else ""
                reverse_indicator = " [REV MODE]" if self.reverse_requested else ""
                print(f"\rSpeed: {speed:.1f} km/h | Gear: {gear_name}{transmission_indicator}{reverse_indicator} | RPM: {rpm} | Steer: {self.current_steer:.2f}    ", end="", flush=True)
            
            return self.control.toMsg()
            
        except Exception as e:
            print(f"Drive error: {e}")
            return self.control.toMsg()

    def onShutDown(self):
        """Cleanup resources"""
        print("\nShutting down manual driver...")
        print("Telemetry data saved for ML training!")
        if hasattr(self, 'telemetry'):
            del self.telemetry
        if self.keyboard:
            # Unhook all keyboard events
            self.keyboard.unhook_all()

    def onRestart(self):
        """Reset for new race"""
        print("\nRestarting race...")
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.current_steer = 0.0
        self.current_accel = 0.0
        self.current_brake = 0.0
        self.current_gear = 1
        self.reverse_requested = False
        self.tick_counter = 0
        
        # Reset key states
        for key in self.keys_pressed:
            self.keys_pressed[key] = False