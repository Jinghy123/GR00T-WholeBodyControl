"""Headless MuJoCo sim runner.

Wraps gear_sonic.scripts.run_sim_loop's logic:
  * forces ENABLE_ONSCREEN=False (no viewer; works on a headless server)
  * starts the sim as a thread
  * SIGUSR1: equivalent of pressing `9` in the viewer — calls
    sim.elastic_band.handle_keyboard_button("9"), which toggles enable.

Prints `SIM_READY` once the sim is up so the orchestrator can sync on it.

This file does NOT modify any existing source — it composes the public APIs
that run_sim_loop.py already uses.
"""

from __future__ import annotations

import signal
import sys
import time
import threading
from dataclasses import dataclass

import tyro

from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model


@dataclass
class HeadlessArgs:
    sim: SimLoopConfig
    viewer: bool = False
    """Show the mujoco viewer (default: off for headless server)."""


def _flush_print(msg: str) -> None:
    print(msg, flush=True)


def main(args: HeadlessArgs) -> None:
    cfg = args.sim
    cfg.enable_onscreen = args.viewer  # headless by default; --viewer enables viewer

    wbc_config = cfg.load_wbc_yaml()
    wbc_config["ENV_NAME"] = cfg.env_name

    if cfg.enable_image_publish:
        assert cfg.enable_offscreen, "enable_offscreen must be True when enable_image_publish is True"

    _ = instantiate_g1_robot_model()  # parity with run_sim_loop

    init_channel(config=wbc_config)
    sim = SimulatorFactory.create_simulator(
        config=wbc_config,
        env_name=cfg.env_name,
        onscreen=args.viewer,
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
    )

    # SIGUSR1 = press `9` in the viewer (toggle elastic band).
    # elastic_band lives on sim.sim_env (DefaultEnv), not on the BaseSimulator wrapper.
    # Install BEFORE start_simulator so it's already wired up when sim begins.
    def _sigusr1(_signum, _frame):
        try:
            env = getattr(sim, "sim_env", None)
            eb = getattr(env, "elastic_band", None) if env is not None else None
            if eb is not None:
                eb.handle_keyboard_button("9")
                _flush_print(f"SIM_TOGGLED_BAND enable={eb.enable}")
            else:
                _flush_print("SIM_DROP_SKIP no elastic_band")
        except Exception as e:  # noqa: BLE001
            _flush_print(f"SIM_DROP_ERROR {e!r}")
    signal.signal(signal.SIGUSR1, _sigusr1)

    if args.viewer:
        # Viewer needs to run on the MAIN thread to keep its event loop happy.
        # SIGUSR1 will still preempt sim_step and run the handler because Python
        # delivers signals to the main thread regardless.
        _flush_print("SIM_READY")
        SimulatorFactory.start_simulator(
            sim,
            as_thread=False,  # blocks until viewer closes / Ctrl+C
            enable_image_publish=cfg.enable_image_publish,
            mp_start_method=cfg.mp_start_method,
            camera_port=cfg.camera_port,
        )
    else:
        SimulatorFactory.start_simulator(
            sim,
            as_thread=True,
            enable_image_publish=cfg.enable_image_publish,
            mp_start_method=cfg.mp_start_method,
            camera_port=cfg.camera_port,
        )
        _flush_print("SIM_READY")
        try:
            while sim.sim_thread.is_alive():
                sim.sim_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main(tyro.cli(HeadlessArgs))
