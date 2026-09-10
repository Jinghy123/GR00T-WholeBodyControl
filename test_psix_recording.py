"""Exercise inference recording through the real WM provider and HLP driver."""

from base64 import b64decode
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import requests

from psix.recording import InferenceRecorder
from psix.hlp import HlpConfig, HlpClient
from psix.instructions import InstructionManager
from psix.wm import WmClient
from rollout_recorder import RolloutRecorder


def reply(payload, status=200):
    return SimpleNamespace(status_code=status, json=lambda: payload)


def hlp_reply(**changes):
    return {"revision": 2, "instruction": "Instruction B", "level": "instruction",
            "parent": None, "done": False, "reason": None, "can_undo": True,
            "memory": {"instruction": ["Instruction A", "Instruction B"], "atomic": []},
            "model": {"decision": "switch", "text": "Instruction B", "raw": "x" * 500},
            **changes}


class InferenceRecordingTest(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.root = Path(self.output.name)
        self.wm = WmClient("http://unused", "", task="Macro", dump_dir=self.root)
        self.addCleanup(self.wm._session.close)
        self.wm.update_latest_ego(np.zeros((48, 64, 3), np.uint8))
        self.client = SimpleNamespace(
            _task="Macro", hold_instructions=lambda reason: None,
            condition_snapshot=lambda: {"executing": False, "execution_started_at": None,
                                        "wm": self.wm.snapshot()})
        def apply(text, *, reason, task=None):
            self.client._task = task or text
            self.wm.set_task(text)
        self.client.set_instruction = apply
        self.hlp = HlpClient(HlpConfig(task="Macro"), self.wm.request_recorder)
        self.manager = InstructionManager(self.client, self.wm, prompt="Macro", mode="hlp", hlp=self.hlp)
        self.addCleanup(self.manager.stop)

    def record(self, service):
        files = sorted((self.root / "requests" / service).glob("*.json"))
        self.assertTrue(files)
        return files[-1], json.loads(files[-1].read_text())

    def test_each_service_records_its_own_exact_image_and_full_reply(self):
        sent = {}

        def wm_post(url, *, json, timeout):
            sent["wm"] = b64decode(json["ego_jpeg"])
            path, pending = self.record("wm")
            self.assertEqual(pending["outcome"], "pending")
            self.assertEqual(path.with_suffix(".obs.jpg").read_bytes(), sent["wm"])
            return reply({"req_id": json["req_id"], "subgoal_jpeg": json["ego_jpeg"]})

        with patch.object(self.wm._session, "post", side_effect=wm_post):
            self.wm._poll_once()
        wm_path, wm_record = self.record("wm")
        self.assertEqual(wm_record["outcome"], "accept_first_monitor")
        self.assertEqual(wm_path.with_suffix(".goal.jpg").read_bytes(), sent["wm"])

        self.wm.update_latest_ego(np.full((48, 64, 3), 200, np.uint8))

        def hlp_post(url, *, json, timeout):
            sent["hlp"] = b64decode(json["ego_image"]["jpeg_b64"])
            path, pending = self.record("hlp")
            self.assertEqual(pending["outcome"], "pending")
            self.assertEqual(path.with_suffix(".obs.jpg").read_bytes(), sent["hlp"])
            return reply(hlp_reply())

        with patch.object(self.hlp._http, "post", side_effect=hlp_post):
            self.manager.poll_once()
        hlp_path, hlp_record = self.record("hlp")
        self.assertNotEqual(sent["wm"], sent["hlp"])
        self.assertEqual(hlp_record["outcome"], "applied")
        self.assertEqual(hlp_record["response"]["model"]["raw"], "x" * 500)
        self.assertEqual(self.client._task, "Instruction B")
        self.assertIsNotNone(cv2.imread(str(hlp_path.with_suffix(".obs.jpg"))))

    def test_timeouts_keep_both_request_images(self):
        for service, session, poll in (("wm", self.wm._session, self.wm._poll_once),
                                       ("hlp", self.hlp._http, self.manager.poll_once)):
            with self.subTest(service=service), patch.object(session, "post", side_effect=requests.Timeout("test timeout")):
                poll()
                path, record = self.record(service)
                self.assertEqual(record["outcome"], "transport_error")
                self.assertIn("test timeout", record["error"])
                self.assertTrue(path.with_suffix(".obs.jpg").exists())

    def test_http_errors_and_invalid_replies_are_recorded(self):
        for service, session, poll in (("wm", self.wm._session, self.wm._poll_once),
                                       ("hlp", self.hlp._http, self.manager.poll_once)):
            for status in (503, 200):
                with self.subTest(service=service, status=status), patch.object(session, "post", return_value=reply({"error": "unavailable"}, status)):
                    poll()
                    _, record = self.record(service)
                    self.assertEqual(record["http_status"], status)
                    self.assertIn(record["outcome"], ("http_error", "invalid_response", "server_error"))
                    self.assertEqual(record["response"], {"error": "unavailable"})

    def test_hlp_control_endpoints_keep_request_reply_pairs(self):
        with patch.object(self.hlp._http, "post", side_effect=lambda url, *, json, timeout: reply(hlp_reply())):
            self.hlp.finish(self.hlp.reset(), "episode_started")
            for command in ("undo", "resume", "override"):
                self.hlp.finish(self.hlp.control(command, subtask="Instruction B"), "applied")
        records = [json.loads(p.read_text()) for p in (self.root / "requests/hlp").glob("*.json")]
        self.assertEqual({r["endpoint"].rsplit("/", 1)[-1] for r in records},
                         {"reset", "undo", "resume", "override"})
        self.assertTrue(all(r["outcome"] not in ("pending", "received") for r in records))

    def test_rejected_and_stayput_goals_keep_the_returned_image(self):
        black = self.wm._encode_jpeg(np.zeros((48, 64, 3), np.uint8), 90)
        white = self.wm._encode_jpeg(np.full((48, 64, 3), 200, np.uint8), 90)
        image = black

        def post(url, *, json, timeout):
            return reply({"req_id": json["req_id"], "subgoal_jpeg": image})

        with patch.object(self.wm._session, "post", side_effect=post):
            self.wm._poll_once()
            image = white
            with patch.object(self.wm, "_collapse_gate", True):
                self.wm._poll_once()
            path, record = self.record("wm")
            self.assertEqual(record["outcome"], "reject")
            self.assertEqual(path.with_suffix(".goal.jpg").read_bytes(), b64decode(white))
            self.assertEqual(self.wm.snapshot()["goal_generation"], 1)
            image = black
            for _ in range(3):
                self.wm._poll_once()
            path, record = self.record("wm")
            self.assertEqual(record["outcome"], "stayput_hold")
            self.assertEqual(path.with_suffix(".goal.jpg").read_bytes(), b64decode(black))

    def test_disabled_recorder_writes_no_images_or_request_records(self):
        recorder = InferenceRecorder(self.root, enabled=False)
        self.wm.request_recorder = recorder
        self.hlp.recorder = recorder
        with patch.object(self.wm._session, "post", side_effect=lambda url, *, json, timeout: reply({"req_id": json["req_id"], "subgoal_jpeg": json["ego_jpeg"]})):
            self.wm._poll_once()
        with patch.object(self.hlp._http, "post", side_effect=requests.Timeout("test timeout")):
            self.manager.poll_once()
        self.assertEqual(list(self.root.iterdir()), [])


class RolloutRecordingTest(unittest.TestCase):
    def test_close_flushes_states_actions_and_readable_avi(self):
        with tempfile.TemporaryDirectory() as directory, patch("rollout_recorder.signal.signal"):
            recorder = RolloutRecorder(directory)
            recorder.record_state(1.0, np.arange(45))
            recorder.record_action(1.0, 1, 2, 3, False, np.arange(80))
            for tick in range(3):
                recorder.record_video_frame(1.0 + tick, np.full((48, 64, 3), tick * 60, np.uint8))
            recorder.close()
            self.assertFalse(recorder._thread.is_alive())
            root = Path(directory)
            with np.load(root / "states_0000.npz") as rows:
                self.assertEqual(rows["states"].shape, (1, 45))
            with np.load(root / "actions_0000.npz") as rows:
                self.assertEqual(rows["action"].shape, (1, 80))
            meta = json.loads((root / "rollout_meta.json").read_text())
            self.assertFalse(meta["partial"])
            self.assertEqual(meta["video"]["file"], "ego.avi")
            self.assertEqual(meta["video"]["frames"], 3)
            video = cv2.VideoCapture(str(root / "ego.avi"))
            try:
                self.assertTrue(video.read()[0])
            finally:
                video.release()

    def test_close_drains_a_full_queue_without_a_stop_slot(self):
        release = threading.Event()
        original = RolloutRecorder._writer

        def delayed_writer(recorder):
            release.wait(3.0)
            original(recorder)

        with tempfile.TemporaryDirectory() as directory, patch("rollout_recorder.signal.signal"), patch("rollout_recorder._QUEUE", 1), patch.object(RolloutRecorder, "_writer", delayed_writer):
            recorder = RolloutRecorder(directory)
            recorder.record_state(1.0, np.arange(43))
            closer = threading.Thread(target=recorder.close, kwargs={"timeout": 4.0})
            closer.start()
            self.assertTrue(recorder._closed.wait(1.0))
            release.set()
            closer.join(5.0)
            self.assertFalse(recorder._thread.is_alive())
            with np.load(Path(directory) / "states_0000.npz") as rows:
                self.assertEqual(rows["states"].shape, (1, 43))


if __name__ == "__main__":
    unittest.main()
