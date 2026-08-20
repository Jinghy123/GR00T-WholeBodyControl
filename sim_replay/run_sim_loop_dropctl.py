"""run_sim_loop.py wrapper that can release the elastic band without GUI focus.

gear_sonic/scripts/run_sim_loop.py holds the robot on a virtual elastic band
(ENABLE_ELASTIC_BAND in the WBC yaml) until you press '9' in the MuJoCo viewer.
That needs a focused window, which scripted runs do not have, so this wrapper
drives the band from a flag file instead:

    touch $DROP_FLAG   ->  lower the band, then release  (robot stands on its own)
    rm    $DROP_FLAG   ->  raise and re-engage the band  (robot hangs again)

The release is a ramp, not a switch: the band anchors the pelvis at z=1.0, about
0.2 m above the policy's standing height, so flipping enable=False outright drops
the robot and the transient throws it several metres forward. Lowering first
(what the viewer's '8' key does) puts the feet on the ground under load.

It also traces pelvis height/tilt at 1 Hz, which is what you look at to tell a
real fall from a controller that was never started.

Takes the same tyro CLI as run_sim_loop.py.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro

from gear_sonic.scripts.run_sim_loop import ArgsConfig, main
from gear_sonic.utils.mujoco_sim import base_sim as _base_sim
from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as _bridge

FLAG = os.environ.get("DROP_FLAG", "/tmp/g1_sim_drop")

_orig_band_init = _bridge.ElasticBand.__init__


def _band_init(self):
    _orig_band_init(self)

    def watch():
        armed = None
        while True:
            want_drop = os.path.exists(FLAG)
            if want_drop != armed:
                armed = want_drop
                if want_drop:
                    for i in range(1, 26):
                        self.length = 0.01 * i  # 0 -> 0.25 m over ~2.5 s
                        time.sleep(0.1)
                    self.enable = False
                    print("[dropctl] band lowered and released", flush=True)
                else:
                    self.length = 0.0
                    self.enable = True
                    print("[dropctl] band re-engaged", flush=True)
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()


_bridge.ElasticBand.__init__ = _band_init

_orig_check_fall = _base_sim.DefaultEnv.check_fall
_last_trace = [0.0]


def _check_fall(self):
    now = time.time()
    if now - _last_trace[0] > 1.0:
        _last_trace[0] = now
        q = self.mj_data.qpos
        print(
            f"[trace] t={now:.1f} pelvis_z={q[2]:.3f} xy=({q[0]:.2f},{q[1]:.2f}) "
            f"quat={q[3]:.3f},{q[4]:.3f},{q[5]:.3f},{q[6]:.3f}",
            flush=True,
        )
    return _orig_check_fall(self)


_base_sim.DefaultEnv.check_fall = _check_fall


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
