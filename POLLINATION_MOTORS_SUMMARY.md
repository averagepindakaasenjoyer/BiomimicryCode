# Pollination Motors Implementation

## Overview
Added support for two new continuous motors used in the pollination process:
1. **Vibration Motor** - For mechanical stimulation
2. **Van de Graaf Generator Motor** - For electrostatic charge

Both motors are controlled via millisecond-based timing for precise control during pollination sequences.

## Hardware Setup

### Pi Server (pi_server.py)

**Motor Definitions:**
```python
continuous_motor_dict = {
    "vibrate": kit2.motor3,      # Vibration motor
    "van_de_graaf": kit2.motor4,  # Van de Graaf generator motor
}
```

**Throttle Settings:**
- `VIBRATE_THROTTLE = 1.0` - Full speed (adjustable if needed)
- `VAN_DE_GRAAF_THROTTLE = 1.0` - Full speed (adjustable if needed)

### Motor Control Functions

#### `continuous_motor_worker(motor_obj, motor_name, duration_ms, throttle=1.0)`
Worker thread function that:
- Runs motor at specified throttle for exact duration in milliseconds
- Automatically stops motor after duration expires
- Handles exceptions gracefully

#### `execute_motor_command(command_dict)`
Updated to handle both stepper and continuous motors:
- **Stepper motors** (rails, main, arm): Step counts (integer)
- **Continuous motors** (vibrate, van_de_graaf): Duration in milliseconds

Example commands:
```python
{"vibrate": 500}              # Run vibration motor for 500ms
{"van_de_graaf": 300}         # Run van de Graaf for 300ms
{"vibrate": 500, "van_de_graaf": 300}  # Run both simultaneously
```

## Laptop Client (laptop_client_manual.py)

### Pollination Functions

#### `vibrate_motor(client_socket, duration_ms)`
Control vibration motor independently
```python
vibrate_motor(client_socket, 500)  # Vibrate for 500ms
```

#### `van_de_graaf_motor(client_socket, duration_ms)`
Control van de Graaf motor independently
```python
van_de_graaf_motor(client_socket, 300)  # Run van de Graaf for 300ms
```

#### `pollinate(client_socket, vibrate_duration_ms=500, van_de_graaf_duration_ms=500, repeat=1)`
Complete pollination sequence:
- Runs vibration first
- Then runs van de Graaf
- Repeats cycle N times
- Includes delays for motors to complete

```python
pollinate(client_socket, 500, 500, 2)  # Pollinate 2 cycles
```

### Interactive Commands

In manual control mode, new commands available:

| Command | Format | Description |
|---------|--------|-------------|
| `vibrate` | `vibrate <ms>` | Run vibration motor for N milliseconds |
| `van_de_graaf` | `van_de_graaf <ms>` | Run van de Graaf motor for N milliseconds |
| `pollinate` | `pollinate [vibrate_ms] [vdg_ms] [repeat]` | Run pollination sequence |

### Command Examples

```
> vibrate 500
[POLLINATE] Vibrating for 500ms
[COMMAND] Vibrate command sent!

> van_de_graaf 300
[POLLINATE] Running van de Graaf for 300ms
[COMMAND] Van de Graaf command sent!

> pollinate 500 500 2
[POLLINATE] Starting pollination sequence (x2)
[POLLINATE] Cycle 1/2
[POLLINATE] Vibrating for 500ms
[POLLINATE] Running van de Graaf for 500ms
[POLLINATE] Cycle 2/2
...
[POLLINATE] Pollination sequence complete
```

## Usage in Demo Mode

Can be integrated into demo mode to automatically pollinate when flower is found:

```python
# In demo_mode_worker, when flower found:
if flower_reached_target_depth:
    pollinate(client_socket, 500, 500, 1)
```

## Timing Precision

- **Resolution**: Milliseconds (1ms minimum practical unit)
- **Method**: `time.sleep()` based timing
- **Accuracy**: ±10-50ms depending on system load
- **Concurrent**: Multiple motors can run simultaneously if desired

## Future Enhancements

- Adjust `VIBRATE_THROTTLE` and `VAN_DE_GRAAF_THROTTLE` for variable power
- Add frequency control for vibration (if hardware supports PWM)
- Integrate pollination into demo mode automatically
- Add sensor feedback (e.g., pollen collection sensors)
- Implement adaptive pollination based on flower type/size
