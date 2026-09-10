"""WM request/recording consistency across instruction changes; no live services."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from psix.wm import WmClient


class WmRequestTest(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)

    def provider(self, *, subtask="", mode="future"):
        provider = WmClient(
            "http://unused", subtask, task="Instruction A", mode=mode,
            seconds=1.6, dump_dir=self.output.name,
        )
        self.addCleanup(provider._session.close)
        provider.update_latest_ego(np.zeros((12, 16, 3), dtype=np.uint8))
        return provider

    def test_future_envelope_without_semantic_subtask_or_hlp(self):
        provider = self.provider()
        body = provider._build_request_body(provider._request_snapshot())
        self.assertEqual(body["subtask"], "Instruction A")
        self.assertEqual(body["seconds"], 1.6)
        self.assertEqual(provider.snapshot()["subtask"], "")

    def test_subgoal_keeps_target_caption_and_omits_future_horizon(self):
        provider = self.provider(subtask="Object on the table", mode="subgoal")
        body = provider._build_request_body(provider._request_snapshot())
        self.assertEqual(body["subtask"], "Object on the table")
        self.assertNotIn("seconds", body)

    def test_prompt_change_preserves_request_and_recorded_task(self):
        provider = self.provider()
        request = provider._request_snapshot()
        provider.set_task("Instruction B")
        provider.set_seconds(2.4)

        body = provider._build_request_body(request)
        self.assertEqual(body["task"], "Instruction A")
        self.assertEqual(body["subtask"], "Instruction A")
        self.assertEqual(body["seconds"], 1.6)
        self.assertLess(body["prompt_gen"], provider.snapshot()["prompt_epoch"])

        record = provider.request_recorder.begin(
            "wm", "http://unused/wm", body,
            context={key: value for key, value in request.items() if key != "ego"},
        )
        provider.request_recorder.response(record, 200, {"subgoal_jpeg": body["ego_jpeg"]})
        provider.request_recorder.finish(record, "stale_dropped")
        metadata = json.loads(record[0].read_text())
        self.assertEqual(metadata["request"]["task"], "Instruction A")
        self.assertEqual(metadata["context"]["epoch"], body["prompt_gen"])

    def test_old_reply_cannot_install_a_goal_after_prompt_change(self):
        provider = self.provider()

        def reply_after_switch(url, *, json, timeout):
            provider.set_task("Instruction B")
            return SimpleNamespace(status_code=200, json=lambda: {
                "req_id": json["req_id"], "subgoal_jpeg": json["ego_jpeg"],
                "prompt_gen": json["prompt_gen"],
                "robot_episode_session_id": json["robot_episode_session_id"],
            })

        with patch.object(provider._session, "post", side_effect=reply_after_switch):
            provider._poll_once()
        snapshot = provider.snapshot()
        self.assertIsNone(snapshot["goal"])
        self.assertIsNone(snapshot["last_wm_error"])
        files = list(Path(self.output.name).glob("requests/wm/*.json"))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text())
        self.assertEqual(record["outcome"], "stale_dropped")
        self.assertEqual(record["request"]["task"], "Instruction A")
        self.assertTrue(files[0].with_suffix(".obs.jpg").exists())
        self.assertTrue(files[0].with_suffix(".goal.jpg").exists())


if __name__ == "__main__":
    unittest.main()
