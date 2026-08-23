"""Send the WBC a start command and hold the token socket open, nothing else.

The deploy idles until it receives a 'command' start, and replay_sonic.py only
sends that at the instant it begins streaming. Lowering the elastic band before
then means dropping a limp robot, which falls and resets. This binds the same
PUB socket, sends start (planner mode, so the robot just stands), holds the
socket while the band is lowered, then closes it so replay_sonic.py can bind and
take over.

    python sim_replay/start_control.py [hold_seconds]
"""

import os
import sys
import time

import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

hold = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
port = int(os.environ.get("ZMQ_PORT", 5556))

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind(f"tcp://*:{port}")
time.sleep(1.5)  # let the deploy's SUB reconnect to the new bind
sock.send(build_command_message(start=True, stop=False, planner=True))
print(f"[start_control] start sent, holding :{port} for {hold:.1f}s", flush=True)
time.sleep(hold)
sock.close()
ctx.term()
print("[start_control] socket released", flush=True)
