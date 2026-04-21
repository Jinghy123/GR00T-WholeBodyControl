"""Subscribes to the TWIST2 teleop Redis key and drives the 2-DOF neck.

Replaces what ~/g1-onboard/docker_neck.sh does on the onboard computer.
Run this on whichever machine is connected to the U2D2 over USB.

Data path:
  xrobot_teleop_to_robot_w_hand.py
    -> Redis SET "action_neck_unitree_g1_with_hands" = "[yaw_rad, pitch_rad]"
    -> (this script) periodically GETs the key
    -> tick = ZERO_TICK + int(rad * 4096 / (2*pi))
    -> Dynamixel write

Ctrl+C for clean shutdown (torque off, port closed).
"""
import json
import math
import signal
import sys
import time

import redis
from dynamixel_sdk import PacketHandler, PortHandler

# ---- your calibration ----
YAW_ZERO_TICK = 1897
PITCH_ZERO_TICK = 2900
# --------------------------

# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_KEY = "action_neck_unitree_g1_with_hands"

# Serial / motor
PORT = "/dev/ttyUSB0"
BAUD = 2_000_000
YAW_ID = 0
PITCH_ID = 1

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_HW_ERROR_STATUS = 70

TICKS_PER_REV = 4096
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

# Software clamps (radians) - conservative to avoid re-triggering overload
YAW_LIMIT_RAD = math.radians(60.0)
PITCH_LIMIT_RAD = math.radians(45.0)

# Smooth the commanded radians with a first-order low-pass to avoid
# Dynamixel "Electrical Shock" / overload trips from jumpy input.
SMOOTH_ALPHA = 0.3  # 0 = frozen, 1 = no smoothing

_running = True


def on_sigint(sig, frame):
    global _running
    _running = False


def rad_to_tick(rad, zero_tick):
    return zero_tick + int(rad * TICKS_PER_REV / (2 * math.pi))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def check_and_reboot(packet, port, motor_id):
    hw, _, err = packet.read1ByteTxRx(port, motor_id, ADDR_HW_ERROR_STATUS)
    if err == 128 or hw != 0:
        print(f"[WARN] ID {motor_id} hw_error=0b{hw:08b} -> rebooting")
        packet.reboot(port, motor_id)
        time.sleep(0.5)
        packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, 1)
        return True
    return False


def main():
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    # Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError as e:
        sys.exit(f"ERROR: cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT} ({e})")

    # Dynamixel
    port = PortHandler(PORT)
    packet = PacketHandler(2.0)
    if not port.openPort() or not port.setBaudRate(BAUD):
        sys.exit(f"ERROR: could not open {PORT} @ {BAUD}")

    for i in (YAW_ID, PITCH_ID):
        _, res, err = packet.ping(port, i)
        if res != 0:
            port.closePort()
            sys.exit(f"ERROR: ping ID {i} failed (res={res})")
        check_and_reboot(packet, port, i)
        packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)

    # Start at zero pose
    yaw_cmd = 0.0
    pitch_cmd = 0.0
    packet.write4ByteTxRx(port, YAW_ID, ADDR_GOAL_POSITION,
                          rad_to_tick(yaw_cmd, YAW_ZERO_TICK))
    packet.write4ByteTxRx(port, PITCH_ID, ADDR_GOAL_POSITION,
                          rad_to_tick(pitch_cmd, PITCH_ZERO_TICK))
    time.sleep(0.3)

    print("=" * 62)
    print(" TWIST2 neck driver (Redis -> Dynamixel)")
    print(f"   Redis   : {REDIS_HOST}:{REDIS_PORT}  key={REDIS_KEY}")
    print(f"   Serial  : {PORT} @ {BAUD} baud (IDs {YAW_ID}/{PITCH_ID})")
    print(f"   Zero    : yaw={YAW_ZERO_TICK}  pitch={PITCH_ZERO_TICK}")
    print(f"   Limits  : yaw +/-{math.degrees(YAW_LIMIT_RAD):.0f}deg, "
          f"pitch +/-{math.degrees(PITCH_LIMIT_RAD):.0f}deg")
    print(f"   Loop    : {CONTROL_HZ} Hz (smoothing alpha={SMOOTH_ALPHA})")
    print(" Ctrl+C to stop.")
    print("=" * 62)

    last_hw_check = 0.0
    last_status_print = 0.0
    last_value = None

    try:
        next_tick = time.time()
        while _running:
            next_tick += DT

            # --- Read latest target from Redis ---
            raw = None
            try:
                raw = r.get(REDIS_KEY)
            except redis.exceptions.ConnectionError:
                raw = None

            yaw_target = yaw_cmd
            pitch_target = pitch_cmd
            if raw is not None:
                try:
                    data = json.loads(raw)
                    if isinstance(data, list) and len(data) >= 2:
                        yaw_target = float(data[0])
                        pitch_target = float(data[1])
                        last_value = (yaw_target, pitch_target)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            # Clamp
            yaw_target = clamp(yaw_target, -YAW_LIMIT_RAD, YAW_LIMIT_RAD)
            pitch_target = clamp(pitch_target, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD)

            # Low-pass smoothing
            yaw_cmd += SMOOTH_ALPHA * (yaw_target - yaw_cmd)
            pitch_cmd += SMOOTH_ALPHA * (pitch_target - pitch_cmd)

            # --- Write to motors ---
            yaw_tick = rad_to_tick(yaw_cmd, YAW_ZERO_TICK)
            pitch_tick = rad_to_tick(pitch_cmd, PITCH_ZERO_TICK)
            packet.write4ByteTxRx(port, YAW_ID, ADDR_GOAL_POSITION, yaw_tick)
            packet.write4ByteTxRx(port, PITCH_ID, ADDR_GOAL_POSITION, pitch_tick)

            now = time.time()

            # Periodic hw-error check every 2s
            if now - last_hw_check > 2.0:
                for i in (YAW_ID, PITCH_ID):
                    check_and_reboot(packet, port, i)
                last_hw_check = now

            # Status line every 0.2s
            if now - last_status_print > 0.2:
                src = "redis" if last_value is not None else "idle"
                sys.stdout.write(
                    f"\r [{src}] yaw {math.degrees(yaw_cmd):+6.1f}deg "
                    f"(tick {yaw_tick:4d})  |  "
                    f"pitch {math.degrees(pitch_cmd):+6.1f}deg "
                    f"(tick {pitch_tick:4d})   "
                )
                sys.stdout.flush()
                last_status_print = now

            # Sleep to maintain loop rate
            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()

    finally:
        print("\nShutting down: returning to zero, torque off")
        packet.write4ByteTxRx(port, YAW_ID, ADDR_GOAL_POSITION,
                              rad_to_tick(0.0, YAW_ZERO_TICK))
        packet.write4ByteTxRx(port, PITCH_ID, ADDR_GOAL_POSITION,
                              rad_to_tick(0.0, PITCH_ZERO_TICK))
        time.sleep(0.5)
        for i in (YAW_ID, PITCH_ID):
            packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
        port.closePort()
        print("Done.")


if __name__ == "__main__":
    main()
