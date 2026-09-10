"""Online instruction client: manual/HLP instructions with RTC or HTTP actions."""

import argparse
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import requests

from psix import robot, wm as world_model, vla, recording
from psix.hlp import HlpClient, HlpConfig, MACRO_PROMPTS, resolve_macro
from psix.instructions import InstructionManager
from psix.keyboard import KeyboardInput
from psix.recording import InferenceRecorder
from rollout_recorder import maybe_rollout_recorder


TASK_INSTRUCTION = "Pick up the object and place it in the container."
DEFAULT_ROLLOUT_ROOT = str(Path.home() / "Desktop/psi/.logs/psix_rollouts")
DEFAULT_WM_DUMP_DIR = None


def build_base_parser(description="PSIX robot client"):
    """The full client CLI; the launchers forward their flags unchanged."""
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--host", type=str, default="localhost",
                        help="VLA policy server host")
    parser.add_argument("--port", type=int, default=8014,
                        help="VLA policy server port")
    parser.add_argument(
        "--embodiment-tag", default=os.environ.get("EMBODIMENT_TAG"),
        help="Expected VLA embodiment tag; defaults to EMBODIMENT_TAG. The "
             "server /info remains authoritative for the wire dimensions."
    )
    parser.add_argument("--zmq-host", type=str, default="localhost",
                        help="ZMQ host for robot state subscriber")
    parser.add_argument("--zmq-pub-port", type=int, default=5556,
                        help="ZMQ PUB port for sending pose to WBC")
    parser.add_argument("--zmq-sub-port", type=int, default=5557,
                        help="ZMQ SUB port for receiving robot state")
    parser.add_argument("--zmq-topic", type=str, default="pose",
                        help="ZMQ topic for pose messages")
    parser.add_argument("--zmq-sub-topic", type=str, default="g1_debug",
                        help="ZMQ topic for robot state subscription")
    parser.add_argument("--camera-address", type=str, default="tcp://192.168.123.164:5558",
                        help="Camera ZMQ address")
    parser.add_argument("--camera-timeout-ms", type=int, default=robot.DEFAULT_CAMERA_TIMEOUT_MS,
                        help="Camera ZMQ send/receive timeout; timed-out REQ sockets recover")
    parser.add_argument("--prompt", "--instruction", dest="instruction", type=str, default=None,
                        help="Current instruction or HLP macro prompt")
    parser.add_argument("--method-name", type=str, default=None,
                        help="Label for the VLA checkpoint/ablation arm under test (e.g. "
                             "statehist_80k, goaldrop_80k, generalist_40k). Embedded in "
                             "run_manifest.json and the init-frame sidecar for later "
                             "cross-method comparison; purely a label, does not affect "
                             "serving or control.")
    parser.add_argument(
        "--goal-source", choices=("wm", "none"), default="wm",
        help="WM goal images (default), or instruction only without WM",
    )
    parser.add_argument("--wm-host", type=str, default="192.168.123.240",
                        help="WM server address on the direct G1-wired subnet")
    parser.add_argument("--wm-port", type=int, default=8016,
                        help="WM HTTP port")
    parser.add_argument("--wm-period", type=float, default=1.6,
                        help="Serialized WM refresh period in seconds (default: 1.6, matched to "
                             "--wm-seconds so a goal is replaced about when it comes due rather "
                             "than being held past its horizon)")
    parser.add_argument("--wm-timeout", type=float, default=15.0,
                        help="Timeout for one POST /wm request; a hung WM blocks the "
                             "serialized refresh worker this long, so keep it short")
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG quality for ego/subgoal wired transport (1-100)")
    parser.add_argument("--wm-mode", type=str, default="future", choices=("future", "subgoal"),
                        help="Match the WM server's prediction mode: future uses the task; "
                             "subgoal uses the stage caption (default: future)")
    parser.add_argument("--wm-seconds", type=float, default=1.6,
                        help="Future prediction horizon in seconds; the server may clamp it "
                             "to its trained range (default: 1.6; 0 uses the server default)")
    parser.add_argument(
        "--wm-goal-hard-age", type=float, default=world_model.DEFAULT_WM_GOAL_HARD_AGE,
        help="Stop/hold only when the current prompt's last-good WM goal exceeds "
             "this age; 0 disables (default: 30s)"
    )
    parser.add_argument("--wm-collapse-gate", dest="wm_collapse_gate",
                        action="store_true", default=False,
                        help="Reject suspected collapsed goals using fixed image "
                             "thresholds; disabled by default. Stay-put retries are separate.")
    parser.add_argument("--no-wm-collapse-gate", dest="wm_collapse_gate",
                        action="store_false",
                        help="Log collapse scores without collapse rejection (default); "
                             "stay-put retries remain enabled")
    parser.add_argument("--output-dir", "--wm-dump-dir", dest="wm_dump_dir", type=str, default=DEFAULT_WM_DUMP_DIR,
                        help="Directory for WM observation/goal pairs and metadata; "
                             "default includes a run timestamp")
    parser.add_argument("--no-rollout-record", action="store_true",
                        help="Disable continuous states/actions/ego-video recording "
                             "into <wm-dump-dir>/rollout (on by default)")
    parser.add_argument("--rollout-video-fps", type=float, default=10.0,
                        help="Ego video frame rate for the rollout recording")
    parser.add_argument("--rollout-video-scale", type=float, default=1.0,
                        help="Resize factor for the recorded ego video")
    parser.add_argument("--observation-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if state, camera, or last VLA observation is older")
    parser.add_argument("--action-stale-timeout", type=float, default=0.5,
                        help="Stop/hold WBC if no fresh monotonic VLA action arrives")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run camera/state + WM + VLA validation without binding or publishing WBC")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--include-neck", dest="include_neck", action="store_true",
        help="Require the 45-D/80-D neck layout (normally inferred from VLA /info)"
    )
    layout.add_argument(
        "--no-include-neck", dest="include_neck", action="store_false",
        help="Require the legacy 43-D/78-D layout (normally inferred from VLA /info)"
    )
    parser.set_defaults(include_neck=None)
    parser.add_argument("--neck-pub-host", type=str, default=robot.DEFAULT_NECK_PUB_HOST,
                        help=f"Neck command PUB bind host (default: {robot.DEFAULT_NECK_PUB_HOST})")
    parser.add_argument("--neck-pub-port", type=int, default=robot.DEFAULT_NECK_PUB_PORT,
                        help=f"Neck command PUB port (default: {robot.DEFAULT_NECK_PUB_PORT})")
    parser.add_argument("--neck-state-zmq", type=str, default=robot.DEFAULT_NECK_STATE_ZMQ,
                        help=f"Neck-state SUB address (default: {robot.DEFAULT_NECK_STATE_ZMQ})")

    parser.add_argument("--show-goal", action="store_true",
                        help="Open a local cv2 window showing the goal image as sent to the VLA (needs DISPLAY)")
    parser.add_argument("--no-show-goal", dest="show_goal", action="store_false", help="Disable the goal window")
    # Default OFF: the served instruction is "Task: X" alone. Stage advance and the
    # WM request keep using the per-stage subtask either way -- this switch only
    # decides whether the clause reaches the VLA.
    parser.add_argument("--subtask-prompt", dest="subtask_prompt", action="store_true",
                        default=False,
                        help="Append '. Subtask: <stage>' to the served instruction. "
                             "Off by default, so the VLA gets 'Task: <task>' alone.")
    parser.add_argument("--no-subtask-prompt", dest="subtask_prompt", action="store_false",
                        help="Explicitly keep the subtask clause off (already the default).")
    parser.add_argument("--neck-reset", dest="neck_reset", action="store_true", default=True,
                        help="Drive the neck to home before connecting (default: on).")
    parser.add_argument("--no-neck-reset", dest="neck_reset", action="store_false",
                        help="Start from wherever the neck currently is.")
    parser.add_argument("--neck-reset-yaw", type=float, default=0.0,
                        help="Home yaw in RADIANS (default: 0.0)")
    parser.add_argument("--neck-reset-pitch", type=float, default=0.0,
                        help="Home pitch in RADIANS (default: 0.0)")
    parser.add_argument("--neck-reset-hold", type=float, default=3.0,
                        help="Max seconds to publish the home target (default: 3.0)")
    parser.add_argument("--neck-reset-tol", type=float, default=0.02,
                        help="Convergence band in rad on both axes (default: 0.02)")
    parser.add_argument("--neck-reset-on-fail", choices=("warn", "abort"), default="warn",
                        help="If the neck never reaches home: warn and continue (default), "
                             "or abort.")
    parser.add_argument("--frozen-action", dest="frozen_action", action="store_true",
                        default=True,
                        help="On a starved tick (server flags rtc_repeat_last), publish a "
                             "body token re-encoded from the CURRENT measured pose instead "
                             "of replaying the stale one; hand/neck keep their last values. "
                             "Default on. Matters most under rtc_mode=off, which has no "
                             "continuity mechanism and starves at every chunk boundary.")
    parser.add_argument("--no-frozen-action", dest="frozen_action", action="store_false",
                        help="Replay the last action verbatim on starved ticks (legacy).")
    parser.add_argument("--encoder-version", dest="encoder_version",
                        choices=("v1", "v1_1"), default="v1",
                        help="Input format of the encoder that builds the first-frame RTC "
                             "prefix and frozen-action hold token: v1 (release model_encoder.onnx, 1762-dim obs, "
                             "the default) or v1_1 (policy/sonic_v1_1/model_encoder.onnx, "
                             "1751-dim obs, heading-normalized anchor). Must match the "
                             "VLA body-token version and the decoder the WBC runs.")
    parser.add_argument("--v1_1", "--v1.1", dest="encoder_version", action="store_const",
                        const="v1_1",
                        help="Shorthand for --encoder-version v1_1.")
    parser.add_argument("--neck-reset-settle", type=float, default=5.0,
                        help="Seconds to wait after homing before the rollout starts "
                             "(default: 5.0). Convergence is read off the encoder, but "
                             "NeckMotor's on-board EMA is still smoothing the last command "
                             "out, so without this the first observations come from a head "
                             "that is still drifting. 0 disables the wait.")
    return parser



def validate_args(parser, args):
    """Range/consistency checks; every failure is a parser.error."""
    if args.camera_timeout_ms <= 0:
        parser.error("--camera-timeout-ms must be positive")
    if args.wm_period <= 0 or args.wm_timeout <= 0:
        parser.error("--wm-period and --wm-timeout must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.wm_goal_hard_age < 0:
        parser.error("--wm-goal-hard-age must be non-negative")
    if args.observation_stale_timeout <= 0:
        parser.error("--observation-stale-timeout must be positive")
    if args.action_stale_timeout <= 0:
        parser.error("--action-stale-timeout must be positive")



def build_parser(*, full_help=False):
    parser = build_base_parser("Online HLP/WM instruction client")
    parser.set_defaults(dry_run=True, embodiment_tag=None, wm_dump_dir=None,
                        host="127.0.0.1", zmq_host="127.0.0.1")
    parser.add_argument("--hlp-mode", choices=("manual", "hlp", "shadow"), default="manual",
                        help="Instruction source; shadow records HLP without applying it")
    parser.add_argument("--rtc-mode", choices=("train", "test_time", "off"), default="test_time",
                        help="Must match the VLA server; off selects HTTP chunk + frozen action")
    parser.add_argument("--next-prompt", action="append", default=[], metavar="TEXT",
                        help="Next manual instruction on Enter; repeat for a sequence")
    parser.add_argument("--subgoal", help="Explicit target text for --wm-mode subgoal")
    parser.add_argument("--list-prompts", action="store_true", help="List HLP macro presets and exit")
    parser.add_argument("--real", dest="dry_run", action="store_false",
                        help="Publish robot commands (default: dry-run)")
    parser.add_argument("--check-only", action="store_true", help="Check service contracts and exit")
    parser.add_argument("--print-config", action="store_true", help="Print resolved settings without network I/O")
    parser.add_argument("--no-record", action="store_true", help="Disable request and continuous rollout recording")
    parser.add_argument("--verbose", action="store_true", help="Show full console output as well as saving client.log")
    parser.add_argument("--client-lock-file", default="/tmp/psix_rtc_robot_client.lock")
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--hlp-host", default="127.0.0.1")
    parser.add_argument("--hlp-port", type=int, default=8015)
    parser.add_argument("--hlp-period", type=float, default=1.0, help="HLP request interval in seconds")
    parser.add_argument("--hlp-timeout", type=float, default=20.0)
    parser.add_argument("--hlp-atomic", action="store_true",
                        help="Let the HLP break the registered cleaning/drink tasks into atomic steps")
    parser.add_argument("--help-all", action="help", help="Show all connection and tuning options")
    primary = {"help", "help_all", "instruction", "hlp_mode", "hlp_atomic", "rtc_mode", "encoder_version",
               "next_prompt", "wm_seconds", "wm_period", "hlp_period", "wm_dump_dir", "show_goal",
               "no_record", "dry_run", "check_only", "print_config", "list_prompts"}
    for action in parser._actions:
        if action.dest == "instruction":
            action.help = "Current instruction, or HLP macro prompt/key"
        elif action.dest == "show_goal":
            action.help = "Enable/disable the local goal-image window (cv2 popup)"
        if not full_help and action.dest not in primary:
            action.help = argparse.SUPPRESS
    if not full_help:
        parser.usage = ("%(prog)s [--prompt TEXT] [--hlp-mode {manual,hlp,shadow}]\n"
                        "                      [--rtc-mode {train,test_time,off}] "
                        "[--encoder-version {v1,v1_1}] [options]")
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(full_help="--help-all" in argv)
    args = parser.parse_args(argv)
    if args.list_prompts:
        for name, prompt in MACRO_PROMPTS.items():
            print(f"{name}: {prompt}")
        parser.exit()
    if args.instruction is None:
        args.instruction = TASK_INSTRUCTION if args.hlp_mode == "manual" else resolve_macro("")[0]
    args.instruction = args.instruction.strip()
    if not args.instruction or any(not text.strip() for text in args.next_prompt):
        parser.error("--prompt and --next-prompt must be non-empty")
    if args.hlp_mode != "manual":
        args.instruction, _ = resolve_macro(args.instruction)
        if args.goal_source != "wm":
            parser.error("HLP instructions currently require --goal-source wm")
    if args.hlp_atomic and args.hlp_mode == "manual":
        parser.error("--hlp-atomic requires --hlp-mode hlp or shadow")
    if args.wm_mode == "subgoal" and not (args.subgoal or "").strip():
        parser.error("--wm-mode subgoal requires --subgoal TEXT")
    if args.wm_mode == "future" and args.subgoal:
        parser.error("--subgoal applies to --wm-mode subgoal")
    if min(args.http_timeout, args.hlp_timeout, args.hlp_period) <= 0:
        parser.error("HTTP/HLP timing settings must be positive")
    for port in (args.port, args.wm_port, args.hlp_port):
        if not 1 <= port <= 65535:
            parser.error("service ports must be in [1, 65535]")
    validate_args(parser, args)
    if args.wm_dump_dir is None:
        args.wm_dump_dir = str(Path(DEFAULT_ROLLOUT_ROOT) / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.wm_dump_dir = str(Path(args.wm_dump_dir).expanduser().resolve())
    return args


def preflight(args):
    """Read the selected endpoints; never acquire HLP or send inference requests."""
    with requests.Session() as session:
        session.trust_env = False

        def get(url):
            response = session.get(url, timeout=5.0)
            response.raise_for_status()
            return response.json()

        info = get(f"http://{args.host}:{args.port}/info")
        if not isinstance(info, dict):
            raise ValueError("VLA /info must return a JSON object")
        if info.get("rtc_mode") != args.rtc_mode:
            raise ValueError(f"VLA rtc_mode={info.get('rtc_mode')!r}; requested {args.rtc_mode!r}")
        if args.goal_source == "wm":
            ready = get(f"http://{args.wm_host}:{args.wm_port}/ready")
            if not isinstance(ready, dict) or ready.get("ready") is not True:
                raise ValueError("WM /ready did not report ready=true")
        if args.hlp_mode != "manual":
            args.hlp_server_info = get(f"http://{args.hlp_host}:{args.hlp_port}/health")
            if args.hlp_atomic and not args.hlp_server_info.get("hierarchical"):
                raise ValueError("this HLP checkpoint cannot plan atomic steps")
    args.goal_hw = robot.goal_hw_from_info(info)
    layout = robot.resolve_vla_embodiment(
        args.host, args.port, requested_tag=args.embodiment_tag,
        include_neck_override=args.include_neck, info=info, require_goal=args.goal_source == "wm",
    )
    print(f"[client] VLA={layout[0]} dims={layout[1]}/{layout[2]} "
          f"rtc={args.rtc_mode} HLP={args.hlp_mode} token={args.encoder_version}")
    return layout[3]


@contextmanager
def client_lock(path):
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another client owns {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


# High-rate status lines the terminal never shows without --verbose. The terminal
# is for instruction/subtask switches; everything else is in client.log.
PERIODIC_LINES = (
    "[client] condition prompt_stage=",
    "[client] observation send",
    "[client] action published:",
    "[client] gated:",
)


class ConsoleLog:
    """Keep the full console in the run directory; filter only the terminal."""

    def __init__(self, file, terminal, verbose=False):
        self.file, self.terminal, self.verbose = file, terminal, verbose
        self._lock = threading.Lock()
        self._local = threading.local()

    def write(self, text):
        with self._lock:
            self.file.write(text)
            self.file.flush()
            if text.startswith("\r[prompt]"):
                self.terminal.write(text)
                self.terminal.flush()
                return len(text)
            if self.verbose:
                self.terminal.write(text)
            else:
                pending = getattr(self._local, "pending", "") + text
                lines = pending.split("\n")
                self._local.pending = lines.pop()
                for line in lines:
                    plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
                    if plain.startswith(PERIODIC_LINES):
                        continue  # per-tick telemetry stays in client.log only
                    if (plain.startswith(("[client]", "[MAIN]", "[hlp]", "[prompt]", "[goal-window]", "[safety]",
                                          "[wm] prompt ->", "[wm] restart"))
                            or any(word in line.lower() for word in ("error", "warning", "failed", "traceback"))):
                        self.terminal.write(line + "\n")
            self.terminal.flush()
        return len(text)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def _close(component, method="stop"):
    try:
        getattr(component, method)()
    except Exception as exc:
        print(f"[client] cleanup failed for {type(component).__name__}: {exc}", flush=True)


def _home_neck(args, publisher, reader, include_neck):
    if not args.neck_reset or not include_neck or publisher is None:
        return
    ok, before, after, elapsed = robot.reset_neck_to_home(
        publisher, reader, yaw=args.neck_reset_yaw, pitch=args.neck_reset_pitch,
        hold_s=args.neck_reset_hold, tol=args.neck_reset_tol)
    recording.log_event("neck_reset", converged=ok, start=before, end=after, elapsed_s=elapsed)
    print(f"[client] neck home: {'converged' if ok else 'WARNING: not converged'}", flush=True)
    if not ok and args.neck_reset_on_fail == "abort":
        raise RuntimeError("neck did not reach home")
    time.sleep(args.neck_reset_settle)


def run(args, include_neck):
    if args.rtc_mode == "off":
        transport = vla.HttpChunkClient
        transport_options = {"http_timeout": args.http_timeout}
    else:
        transport = vla.RTCWebSocketClient
        transport_options = {}
    hlp_config = None if args.hlp_mode == "manual" else HlpConfig(
        task=args.instruction, host=args.hlp_host, port=args.hlp_port, period=args.hlp_period,
        timeout=args.hlp_timeout, jpeg_quality=args.jpeg_quality, atomic=args.hlp_atomic)
    server_url = (f"http://{args.host}:{args.port}/act" if args.rtc_mode == "off"
                  else f"ws://{args.host}:{args.port}/ws")
    wm_base_url = f"http://{args.wm_host}:{args.wm_port}"
    config = vars(args)
    record_requests = not args.no_record
    rollout_record = not (args.no_record or args.no_rollout_record)
    running = threading.Event()
    running.set()
    run_dir = os.path.abspath(os.path.expanduser(args.wm_dump_dir))
    os.makedirs(run_dir, exist_ok=True)
    print(f"[client] starting {transport.__name__}; {'dry-run' if args.dry_run else 'REAL robot commands'}", flush=True)
    with ExitStack() as cleanup:
        cleanup.callback(running.clear)
        event_log = recording.EventLog(os.path.join(run_dir, "events.jsonl"))
        recording.set_event_log(event_log)
        cleanup.callback(recording.set_event_log, None)
        cleanup.callback(_close, event_log)
        recorder = InferenceRecorder(run_dir, enabled=record_requests)
        rollout = maybe_rollout_recorder(os.path.join(run_dir, "rollout"), enabled=rollout_record,
                                         video_fps=args.rollout_video_fps, video_scale=args.rollout_video_scale)
        if rollout is not None:
            cleanup.callback(_close, rollout, "close")

        def component(obj):
            cleanup.callback(_close, obj)
            return obj

        publisher = None if args.dry_run else component(robot.TokenPublisher(host="*", port=args.zmq_pub_port, topic=args.zmq_topic))
        state = component(robot.RobotStateSubscriber(host=args.zmq_host, port=args.zmq_sub_port, topic=args.zmq_sub_topic))
        camera_type = robot.ZedNeckCamera if include_neck else robot.RSCamera
        camera = component(camera_type(address=args.camera_address, timeout_ms=args.camera_timeout_ms))
        neck_state = component(robot.NeckStateReader(args.neck_state_zmq)) if include_neck else None
        neck_pub = (component(robot.NeckPublisher(host=args.neck_pub_host, port=args.neck_pub_port))
                    if include_neck and not args.dry_run else None)
        _home_neck(args, neck_pub, neck_state, include_neck)

        if args.goal_source == "none":
            wm = world_model.NoGoalProvider(task=args.instruction)
        else:
            wm = world_model.WmClient(
                base_url=wm_base_url, subgoal=args.subgoal, task=args.instruction, period=args.wm_period,
                timeout=args.wm_timeout, jpeg_quality=args.jpeg_quality, goal_hard_age=args.wm_goal_hard_age,
                dump_dir=run_dir, mode=args.wm_mode, seconds=args.wm_seconds, request_recorder=recorder,
                collapse_gate=args.wm_collapse_gate)
        component(wm)
        recording.write_run_manifest(
            run_dir, config={"launch": config, "server_url": server_url, "goal_source": args.goal_source,
                             "task_instruction": args.instruction, "wm_period": args.wm_period,
                             "wm_seconds": args.wm_seconds, "wm_mode": args.wm_mode, "dry_run": args.dry_run,
                             "include_neck": include_neck, "method_name": args.method_name,
                             "gate_version": world_model.WM_GATE_VERSION if args.goal_source == "wm" else "bypass"},
            vla_info=recording._fetch_json(server_url.rsplit("/", 1)[0].replace("ws://", "http://") + "/info"),
            wm_state=recording._fetch_json(wm_base_url + "/state") if args.goal_source == "wm" else {"source": "no_goal"},
            episode_session_id=wm._episode_session_id)
        recording.save_init_frame(run_dir, camera, args.instruction, args.method_name,
                             args.camera_address, include_neck, wm._episode_session_id)
        # Camera capture has one owner after this initial frame: the transport's observation loop.
        for _ in range(30):
            if state.get_state() is not None:
                break
            time.sleep(0.5)
        else:
            print("[client] WARNING: no robot state after 15s", flush=True)
        client = transport(
            server_url=server_url, state_subscriber=state, camera=camera, token_publisher=publisher,
            wm_provider=wm, task_instruction=args.instruction, dry_run=args.dry_run,
            observation_stale_timeout=args.observation_stale_timeout, action_stale_timeout=args.action_stale_timeout,
            include_neck=include_neck,
            neck_publisher=neck_pub, neck_state_reader=neck_state, rollout_recorder=rollout,
            encoder_version=args.encoder_version, frozen_action=args.frozen_action,
            subtask_prompt=args.subtask_prompt, goal_hw=getattr(args, "goal_hw", (270, 480)),
            run_event=running, show_goal=args.show_goal, **transport_options)
        component(client)
        hlp = HlpClient(hlp_config, recorder) if hlp_config is not None else None
        instructions = InstructionManager(client, wm, prompt=args.instruction,
                                           next_prompts=args.next_prompt or (), mode=args.hlp_mode,
                                           hlp=hlp, log_event=recording.log_event)
        component(instructions)
        keyboard = component(KeyboardInput(instructions, wm, client, running))
        wm.start()
        instructions.start()
        keyboard.start()
        thread = threading.Thread(target=client.run, name="vla-transport", daemon=True)
        thread.start()
        hold_thread = threading.Thread(target=client.hold_loop, name="frozen-pose", daemon=True)
        hold_thread.start()
        # Register last: stop publication before waiting for HLP or closing robot I/O.
        def stop_transport():
            running.clear()
            instructions.cancel()
            client.stop()
            thread.join(timeout=3)
            hold_thread.join(timeout=2)
            if thread.is_alive():
                print("[client] WARNING: VLA transport is still shutting down", flush=True)
        cleanup.callback(stop_transport)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, lambda *_: running.clear())
            cleanup.callback(signal.signal, signum, previous)
        recording.log_event("run_start", dry_run=args.dry_run, transport=transport.__name__)
        print("[client] running; Enter: move on | p: back | i: type prompt | q: quit", flush=True)
        try:
            while running.is_set() and thread.is_alive():
                time.sleep(0.2)
        finally:
            recording.log_event("shutdown")
    print("[client] shutdown complete; recorders flushed", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        return
    include_neck = preflight(args)
    if args.check_only:
        print("[client] service checks passed")
        return
    with client_lock(args.client_lock_file):
        Path(args.wm_dump_dir).mkdir(parents=True, exist_ok=True)
        print(f"[client] output: {args.wm_dump_dir}")
        with open(Path(args.wm_dump_dir) / "client.log", "a", buffering=1) as file:
            console = ConsoleLog(file, sys.stdout, args.verbose)
            with redirect_stdout(console), redirect_stderr(console):
                print(f"[client] prompt: {args.instruction}")
                run(args, include_neck)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        raise SystemExit(f"[client] {exc}")
