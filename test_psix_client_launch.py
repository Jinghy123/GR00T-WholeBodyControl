"""Offline launcher parsing, transport selection, and service-contract checks."""

import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import create_autospec, patch

import psix_client as launch
from psix import vla, robot
from psix.instructions import InstructionManager
from contextlib import ExitStack
import tempfile


class ClientLaunchTest(unittest.TestCase):
    def test_shell_defaults_and_overrides_need_no_exports_or_task_files(self):
        root = Path(__file__).resolve().parent
        env = dict(os.environ, TASK_KEY="does-not-exist", EMBODIMENT_TAG="stale",
                   HLP_MODE="off", VLA_HOST="stale", HLP_DRIVER_MODE="off")
        for script, expected_hlp, expected_atomic in (
                ("run_psix_hlpwm_client.sh", "manual", False),
                ("run_psix_hlp_instruction_client.sh", "hlp", False),
                ("run_psix_hlp_atomic_client.sh", "hlp", True)):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", str(root / script), "--print-config", "--prompt", "Custom task",
                     "--rtc-mode", "off", "--v1_1", "--real", "--wm-period", "2.0",
                     "--wm-seconds", "2.4", "--hlp-period", "0.7", "--no-show-goal"],
                    cwd="/tmp", env=env, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(result.stdout)
                self.assertEqual(config["instruction"], "Custom task")
                self.assertEqual(config["hlp_mode"], expected_hlp)
                self.assertEqual(config["hlp_atomic"], expected_atomic)
                self.assertEqual(config["rtc_mode"], "off")
                self.assertEqual(config["encoder_version"], "v1_1")
                self.assertEqual(config["host"], "127.0.0.1")
                self.assertIsNone(config["embodiment_tag"])
                self.assertFalse(config["dry_run"])
                self.assertNotIn("task_key", config)
                self.assertEqual((config["wm_period"], config["wm_seconds"], config["hlp_period"]), (2.0, 2.4, 0.7))
                self.assertFalse(config["show_goal"])

    def test_token_hlp_rtc_axes_select_the_same_runtime_features(self):
        for mode in ("train", "test_time", "off"):
            for source in ("manual", "hlp", "shadow"):
                for version in ("v1", "v1_1"):
                    with self.subTest(mode=mode, source=source, version=version), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                        args = launch.parse_args([
                            "--rtc-mode", mode, "--hlp-mode", source,
                            "--encoder-version", version, "--prompt", "Task A",
                            "--next-prompt", "Task B", "--no-record", "--output-dir", directory,
                        ])
                        factories = {}
                        for name in ("RTCWebSocketClient", "HttpChunkClient"):
                            factories[name] = stack.enter_context(patch.object(vla, name, autospec=True))
                            factories[name].__name__ = name
                        for name in ("TokenPublisher", "RobotStateSubscriber", "ZedNeckCamera", "NeckPublisher", "NeckStateReader"):
                            stack.enter_context(patch.object(robot, name))
                        for name in ("write_run_manifest", "save_init_frame", "_fetch_json"):
                            stack.enter_context(patch.object(launch.recording, name))
                        instructions = stack.enter_context(patch.object(launch, "InstructionManager"))
                        hlp = stack.enter_context(patch.object(launch, "HlpClient"))
                        stack.enter_context(patch.object(launch, "KeyboardInput"))
                        stack.enter_context(patch("sys.stdout", io.StringIO()))
                        launch.run(args, include_neck=True)
                        selected = factories["HttpChunkClient" if mode == "off" else "RTCWebSocketClient"]
                        selected.assert_called_once()
                        options = selected.call_args.kwargs
                        self.assertEqual(options["task_instruction"], "Task A")
                        self.assertEqual(options["encoder_version"], version)
                        self.assertTrue(options["frozen_action"])
                        self.assertTrue(options["include_neck"])
                        self.assertTrue(options["dry_run"])
                        self.assertEqual(instructions.call_args.kwargs["next_prompts"], ["Task B"])
                        self.assertEqual(instructions.call_args.kwargs["mode"], source)
                        self.assertEqual(hlp.call_count, int(source != "manual"))
                        robot.TokenPublisher.assert_not_called()

    def test_preflight_rejects_rtc_mismatch_before_inference(self):
        args = launch.parse_args(["--rtc-mode", "off"])
        with patch.object(launch.requests, "Session") as factory:
            session = factory.return_value.__enter__.return_value
            session.get.return_value.json.return_value = {"rtc_mode": "train"}
            with self.assertRaisesRegex(ValueError, "rtc_mode"):
                launch.preflight(args)
            session.get.assert_called_once_with("http://127.0.0.1:8014/info", timeout=5.0)
            session.post.assert_not_called()

    def test_help_is_short_and_advanced_defaults_remain_available(self):
        for option, has_ports in (("--help", False), ("--help-all", True)):
            output = io.StringIO()
            with patch("sys.stdout", output), self.assertRaises(SystemExit) as raised:
                launch.parse_args([option])
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual("--wm-port" in output.getvalue(), has_ports)
            self.assertIn("--prompt", output.getvalue())
            self.assertNotIn("--task-key", output.getvalue())

    def test_manual_sequence_changes_wm_and_vla_without_dataset_stages(self):
        from types import SimpleNamespace
        client = SimpleNamespace(task="A")
        provider = SimpleNamespace(task="A")
        def change(text, *, reason):
            client.task = provider.task = text
        client.set_instruction = change
        manager = InstructionManager(client, provider, prompt="A", next_prompts=["B"])
        self.assertTrue(manager.command("next"))
        self.assertEqual((client.task, provider.task), ("B", "B"))
        self.assertFalse(manager.command("next"))
        self.assertTrue(manager.command("prev"))
        self.assertEqual((client.task, provider.task), ("A", "A"))
        self.assertTrue(manager.command("next"))
        self.assertEqual(client.task, "B")

    def test_quiet_terminal_keeps_colored_hlp_switches_and_full_log(self):
        log, terminal = io.StringIO(), io.StringIO()
        console = launch.ConsoleLog(log, terminal)
        console.write("[hlp-poll] diagnostic detail\n")
        console.write("\x1b[1;31m[hlp] >>> SWITCH >>> Instruction B\x1b[0m")
        console.write("\n")
        self.assertIn("SWITCH", terminal.getvalue())
        self.assertNotIn("diagnostic detail", terminal.getvalue())
        self.assertIn("diagnostic detail", log.getvalue())

    def test_quiet_terminal_reports_wbc_stop(self):
        log, terminal = io.StringIO(), io.StringIO()
        console = launch.ConsoleLog(log, terminal)
        stopped = "[safety] WBC stopped/held: VLA action stream stale (0.600s > 0.500s)\n"
        console.write(stopped)
        self.assertIn(stopped, terminal.getvalue())
        self.assertIn(stopped, log.getvalue())


if __name__ == "__main__":
    unittest.main()
