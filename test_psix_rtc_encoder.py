"""Encoder selection and deferred-goal ack checks; no sockets or robot I/O."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from psix import vla as client, robot
import psix_client as launch


def make_client(version="v1"):
    state = dict(body_q_measured=np.zeros(29), base_quat_measured=np.array([1., 0., 0., 0.]))
    return client.RTCWebSocketClient(
        "ws://unused", SimpleNamespace(get_state=lambda: state), None, None, None,
        "pick the fruit", dry_run=True, encoder_version=version,
    )


class EncoderSelectionTest(unittest.TestCase):
    def test_cli_version_selects_both_init_and_hold(self):
        for flags, version in (([], "v1"), (["--v1_1"], "v1_1"),
                               (["--encoder-version", "v1_1"], "v1_1")):
            with self.subTest(flags=flags):
                args = launch.build_parser().parse_args(flags)
                token = np.full(64, .125, dtype=np.float32)
                encoder = SimpleNamespace(encode=lambda *args: token)
                with patch.object(robot, "EncoderClient", return_value=encoder) as factory:
                    instance = make_client(args.encoder_version)
                    self.assertIs(instance._encoder, encoder)
                    action = np.linspace(-.2, .2, 80).astype(np.float32)
                    frozen = instance._freeze_action(action)
                    factory.assert_called_once_with(client.ENCODER_MODELS[version], mode=0, version=version)
                    self.assertIs(instance._hold_encoder, instance._encoder)
                    np.testing.assert_array_equal(frozen[14:78], token)
                    np.testing.assert_array_equal(frozen[:14], action[:14])
                    np.testing.assert_array_equal(frozen[78:], action[78:])

    def test_hold_retry_keeps_selected_version_after_init_load_failure(self):
        token = np.zeros(64, dtype=np.float32)
        encoder = SimpleNamespace(encode=lambda *args: token)
        with patch.object(robot, "EncoderClient", side_effect=[RuntimeError("temporary failure"), encoder]) as factory:
            instance = make_client("v1_1")
            self.assertIsNone(instance._encoder)
            instance._freeze_action(np.zeros(80, dtype=np.float32))
            self.assertEqual(factory.call_count, 2)
            for call in factory.call_args_list:
                self.assertEqual(call.args, (client.ENCODER_MODELS["v1_1"],))
                self.assertEqual(call.kwargs, dict(mode=0, version="v1_1"))

    def test_old_goal_actions_remain_acceptable_until_new_chunk_ack(self):
        with patch.object(robot, "EncoderClient", return_value=SimpleNamespace()):
            instance = make_client()
        instance._vla_session_id = "test-session"
        instance._active_condition = dict(sid="test-session", cid=0, hash="a" * 64, prompt_epoch=0)
        instance._candidate_condition = dict(sid="test-session", cid=1, hash="b" * 64, prompt_epoch=0)
        instance._wm = SimpleNamespace(snapshot=lambda: {'prompt_epoch': 0})
        # An action is executable once an observation for this prompt has gone out,
        # and only for versions the server numbered after that.
        self.assertFalse(instance._accept_action(1, None)[0])
        instance._observation_sent()
        self.assertTrue(instance._accept_action(2, None)[0])
        self.assertFalse(instance._accept_action(3, 99)[0])   # wrong prompt epoch (HTTP pairing)


if __name__ == "__main__":
    unittest.main()
