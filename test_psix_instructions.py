"""Instruction switching, rollback and holds with real WM/condition state.

Only the network and the robot I/O are mocked: the WM provider, the VLA
condition machinery and the instruction manager are the real objects.
"""

from base64 import b64decode
from contextlib import ExitStack
import json
from pathlib import Path
import os
import pty
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from psix import robot, vla
from psix.hlp import HlpClient, HlpConfig
from psix.instructions import InstructionManager
from psix.keyboard import KeyboardInput
from psix.wm import WmClient


class Backend:
    """A scripted HLP server speaking the deployed wire."""

    def __init__(self):
        self.revision = 1
        self.memory = []
        self.atomic = []
        self.next = 'A'
        self.level = 'instruction'
        self.awaiting = True
        self.done = False

    @property
    def items(self):
        return self.atomic if self.level == 'atomic' else self.memory

    def post(self, url, *, json, timeout):
        route = url.rsplit('/', 1)[-1]
        if route == 'hlp' and not self.done:
            if self.next is not None and (self.awaiting or not self.items or self.items[-1] != self.next):
                self.items.append(self.next)
                self.awaiting = False
                self.revision += 1
        elif route == 'undo':
            if not self.done and len(self.items) > 1:
                self.items.pop()
            self.done = self.awaiting = False
            self.revision += 1
        elif route == 'resume':
            # Past a done latch the server marks the current text finished.
            self.done = False
            self.awaiting = True
            self.revision += 1
        elif route == 'override':
            self.items.append(json['subtask'])
            self.done = self.awaiting = False
            self.revision += 1
        elif route == 'reset':
            self.memory.clear()
            self.atomic.clear()
            self.done = False
            self.awaiting = True
            self.revision += 1
        return SimpleNamespace(status_code=200, json=self.reply)

    def reply(self):
        current = self.items[-1] if self.items else None
        return {'revision': self.revision, 'instruction': None if self.done or self.awaiting else current,
                'level': self.level, 'parent': self.memory[-1] if self.level == 'atomic' and self.memory else None,
                'done': self.done,
                'reason': 'episode_done' if self.done else 'awaiting_instruction' if self.awaiting else None,
                'memory': {'instruction': list(self.memory), 'atomic': list(self.atomic)},
                'can_undo': bool(self.items), 'model': {'decision': 'switch'}}


class InstructionControlTest(unittest.TestCase):
    def rig(self, transport=vla.RTCWebSocketClient, neck=False, mode='hlp'):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        wm = WmClient('http://unused', task='Macro', dump_dir=directory)
        stack.callback(wm.stop)
        wm.update_latest_ego(np.full((24, 32, 3), 80, np.uint8))
        state = SimpleNamespace(get_state=lambda: dict(body_q_measured=np.zeros(29), base_quat_measured=[1, 0, 0, 0]), age=lambda: 0)
        token = np.full(64, 0.125, np.float32)
        stack.enter_context(patch.object(robot, 'EncoderClient', return_value=SimpleNamespace(encode=lambda *a: token)))
        publisher, neck_publisher = Mock(), Mock()
        client = transport('http://unused' if transport is vla.HttpChunkClient else 'ws://unused', state,
                           SimpleNamespace(age=lambda: 0), publisher, wm, 'Macro',
                           dry_run=False, include_neck=neck, neck_publisher=neck_publisher if neck else None,
                           neck_state_reader=SimpleNamespace(age=lambda: 0) if neck else None,
                           encoder_version='v1_1' if neck else 'v1')
        stack.callback(client.stop)
        client._last_observation_at = time.monotonic()
        client._reset_condition_session()
        hlp = HlpClient(HlpConfig(task='Macro'), wm.request_recorder)
        backend = Backend()
        stack.enter_context(patch.object(hlp._http, 'post', side_effect=backend.post))
        stack.enter_context(patch.object(wm._session, 'post', side_effect=lambda url, *, json, timeout:
            SimpleNamespace(status_code=200, json=lambda: {'req_id': json['req_id'], 'subgoal_jpeg': json['ego_jpeg']})))
        if mode != 'hlp':
            client.set_instruction('Macro', reason='manual')
        manager = InstructionManager(client, wm, prompt='Macro', mode=mode, hlp=hlp)
        stack.callback(manager.stop)
        return SimpleNamespace(client=client, wm=wm, hlp=hlp, backend=backend, manager=manager,
                               publisher=publisher, neck=neck_publisher, directory=directory, token=token)

    def poll(self, rig, text=None, *, level=None, done=None):
        """One HLP round trip on a fresh frame."""
        if text is not None:
            rig.backend.next = text
        if level is not None:
            rig.backend.level = level
        if done is not None:
            rig.backend.done = done
        rig.wm.update_latest_ego(rig.wm.latest_ego()[0])
        rig.manager.poll_once()
        self.assertIsNone(rig.manager.snapshot()['error'])
        return rig.manager.snapshot()

    def run_command(self, rig, name, text=None):
        self.assertTrue(rig.manager.command(name, text))
        rig.manager._run_command(*rig.manager._commands.get_nowait())
        return rig.manager.snapshot()

    def goal_and_action(self, rig, text, version):
        """Land a WM goal for `text`, mark an observation sent, get one action executed."""
        rig.wm._poll_once()
        epoch = int(rig.wm.snapshot()['prompt_epoch'])
        rig.client._observation_sent()
        action = np.linspace(-.2, .2, 80 if rig.client._include_neck else 78, dtype=np.float32)
        rig.client._last_observation_at = time.monotonic()
        if isinstance(rig.client, vla.HttpChunkClient):
            meta = {'epoch': epoch, 'seq': version, 'infer_ms': 0}
            self.assertEqual(rig.client._publish_action(action, meta, 0, False, time.monotonic()), 'ok')
        else:
            data = {'action': robot.numpy_serialize(action[None]), 'version': version}
            rig.client._on_message(None, json.dumps(data))
        self.assertTrue(rig.client.condition_snapshot()['executing'])
        return {'epoch': epoch, 'version': version}, action

    # ------------------------------------------------------------- switching

    def test_rollback_updates_both_transports_memory_goals_and_actions(self):
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            for neck in (False, True):
                with self.subTest(transport=transport.__name__, neck=neck):
                    rig = self.rig(transport, neck)
                    self.poll(rig, 'A')
                    self.goal_and_action(rig, 'A', 1)
                    self.poll(rig, 'B')
                    old, last_action = self.goal_and_action(rig, 'B', 2)
                    self.assertTrue(rig.manager.command('prev'))
                    self.assertFalse(rig.client.condition_snapshot()['executing'])
                    self.assertIsNone(rig.wm.snapshot()['goal'])
                    self.assertIsNone(rig.wm._request_snapshot())
                    rig.client._check_action_liveness()
                    rig.client.hold_tick()
                    self.assertFalse(any(call.kwargs.get('stop') for call in rig.publisher.send_command.call_args_list))
                    wire = rig.publisher.publish_token.call_args.args[0]
                    np.testing.assert_array_equal(wire[:64], rig.token)
                    np.testing.assert_array_equal(wire[64:], last_action[:14])
                    if neck:
                        np.testing.assert_array_equal(rig.neck.publish.call_args.args, last_action[78:])
                    rig.manager._run_command(*rig.manager._commands.get_nowait())
                    snap = rig.manager.snapshot()
                    self.assertEqual((snap['current'], rig.client._task), ('A', 'A'))
                    self.assertEqual(rig.wm._request_snapshot()['task'], 'A')
                    self.assertEqual(snap['hlp']['memory']['instruction'], ['A'])
                    self.assertTrue(snap['paused'])
                    # Nothing is executable again until an observation for the new prompt goes out.
                    self.assertIsNone(rig.client._action_floor)
                    self.goal_and_action(rig, 'A', 4)
                    self.assertFalse(rig.client._holding)
                    self.assertEqual(rig.publisher.send_command.call_count, 1)

    def test_atomic_steps_steer_the_wm_while_the_vla_keeps_the_instruction(self):
        """The VLA was trained with the instruction-level sentence as its Task;
        an atomic step only conditions the WM goal image (and, with
        --subtask-prompt, the Subtask clause that training also used)."""
        parent = 'Pick up the table cloth, walk to the kitchen island, clean the surface.'
        step = 'Wipe the table with the table cloth'
        for subtask_prompt in (False, True):
            with self.subTest(subtask_prompt=subtask_prompt):
                rig = self.rig()
                rig.client._subtask_prompt = subtask_prompt
                frame = np.full((24, 32, 3), 90, np.uint8)
                rig.client._camera = SimpleNamespace(age=lambda: 0, get_frame=lambda: frame)
                sample = (dict(body_q_measured=np.zeros(29), base_quat_measured=[1, 0, 0, 0],
                               left_hand_q=np.zeros(7), right_hand_q=np.zeros(7)), time.monotonic())
                rig.client._state_sub = SimpleNamespace(age=lambda: 0, get_state=lambda: sample[0],
                                                        get_state_with_timestamp=lambda: sample)
                self.poll(rig, parent)
                self.assertEqual((rig.client._task, rig.client._subtask), (parent, None))
                self.assertEqual(rig.wm._request_snapshot()['task'], parent)

                self.poll(rig, step, level='atomic')
                self.assertEqual(rig.client._task, parent)        # VLA prompt unchanged
                self.assertEqual(rig.client._subtask, step)       # only the Subtask clause
                self.assertEqual(rig.wm._request_snapshot()['task'], step)  # WM follows the step
                rig.wm._poll_once()
                self.assertEqual(rig.client._observe().instruction,
                                 f'Task: {parent.lower()}. Subtask: {step.lower()}'
                                 if subtask_prompt else f'Task: {parent.lower()}')

    def test_a_new_revision_rebuilds_the_condition_even_for_the_same_sentence(self):
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            with self.subTest(transport=transport.__name__):
                rig = self.rig(transport)
                self.poll(rig, 'Serve the drink')
                self.poll(rig, 'Pour the drink', level='atomic')
                old, action = self.goal_and_action(rig, 'Pour the drink', 1)
                # A repeated reply at the same revision must not re-freeze the robot.
                self.poll(rig)
                self.assertTrue(rig.client.condition_snapshot()['executing'])
                rig.backend.atomic.clear()
                rig.backend.awaiting = True
                rig.backend.revision += 1
                self.poll(rig)
                self.assertFalse(rig.client.condition_snapshot()['executing'])
                fresh, _ = self.goal_and_action(rig, 'Pour the drink', 2)
                self.assertGreater(fresh['epoch'], old['epoch'])
                self.assertEqual(rig.publisher.send_command.call_count, 1)

    def test_a_planner_hold_stops_the_robot_and_keeps_planning(self):
        rig = self.rig()
        self.poll(rig, 'Serve the drink')
        self.goal_and_action(rig, 'Serve the drink', 1)
        rig.backend.next = None
        rig.backend.awaiting = True
        rig.backend.revision += 1
        held = self.poll(rig)
        self.assertIsNone(held['current'])
        self.assertFalse(held['paused'])
        self.assertFalse(rig.client.condition_snapshot()['executing'])
        self.assertIsNone(rig.wm._request_snapshot())

    def test_the_planner_only_advances_once_the_robot_is_running_the_current_text(self):
        rig = self.rig()
        self.poll(rig, 'A')
        self.assertIsNone(rig.hlp._http.post.call_args.kwargs['json']['executed_revision'])
        self.goal_and_action(rig, 'A', 1)
        self.poll(rig)
        self.assertEqual(rig.hlp._http.post.call_args.kwargs['json']['executed_revision'],
                         rig.manager.snapshot()['revision'])

    def test_shadow_records_the_plan_without_touching_the_robot(self):
        rig = self.rig(mode='shadow')
        epoch = rig.wm.snapshot()['prompt_epoch']
        self.poll(rig, 'Pour the drink', level='atomic')
        self.assertEqual(rig.manager.snapshot()['current'], 'Macro')
        self.assertEqual(rig.wm.snapshot()['prompt_epoch'], epoch)

    # ------------------------------------------------------------------ done

    def test_done_pauses_and_enter_moves_on_while_p_reopens_the_last_instruction(self):
        for key, expected in (('next', None), ('prev', 'A')):
            with self.subTest(key=key):
                rig = self.rig()
                self.poll(rig, 'A')
                self.goal_and_action(rig, 'A', 1)
                finished = self.poll(rig, done=True)
                self.assertTrue(finished['done'] and finished['paused'])
                self.assertIsNone(rig.wm._request_snapshot())
                self.assertTrue(finished['can_prev'])
                snap = self.run_command(rig, key)
                self.assertFalse(snap['done'])
                self.assertFalse(rig.backend.done)
                self.assertEqual(snap['current'], expected)
                self.assertEqual(snap['paused'], key == 'prev')

    def test_enter_while_running_only_unpauses_and_never_holds_the_robot(self):
        rig = self.rig()
        self.poll(rig, 'A')
        active, _ = self.goal_and_action(rig, 'A', 1)
        rig.manager._paused = True
        self.assertTrue(rig.manager.command('next'))
        self.assertTrue(rig.manager._commands.empty())
        self.assertFalse(rig.manager.snapshot()['paused'])
        self.assertIsNone(rig.client._instruction_hold_reason)
        self.assertTrue(rig.client.condition_snapshot()['executing'])

    def test_a_failed_control_holds_the_robot_and_leaves_the_next_poll_to_resync(self):
        rig = self.rig()
        self.poll(rig, 'A')
        self.poll(rig, 'B')
        self.goal_and_action(rig, 'B', 1)
        self.assertTrue(rig.manager.command('prev'))
        with patch.object(rig.hlp._http, 'post', side_effect=requests.Timeout('unreachable')):
            rig.manager._run_command(*rig.manager._commands.get_nowait())
        snap = rig.manager.snapshot()
        self.assertIsNotNone(snap['error'])
        self.assertFalse(snap['paused'])
        self.assertIsNotNone(rig.client._instruction_hold_reason)
        self.assertIsNone(rig.wm._request_snapshot())
        self.assertEqual(self.poll(rig)['current'], rig.backend.memory[-1])

    def test_observe_gates_on_missing_state_stale_state_and_missing_goal(self):
        """Both transports share this; a gated tick holds the robot and yields nothing."""
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            with self.subTest(transport=transport.__name__):
                rig = self.rig(transport)
                frame = np.full((24, 32, 3), 90, np.uint8)
                rig.client._camera = SimpleNamespace(age=lambda: 0, get_frame=lambda: frame)
                state = dict(body_q_measured=np.zeros(29), base_quat_measured=[1, 0, 0, 0],
                             left_hand_q=np.zeros(7), right_hand_q=np.zeros(7))
                dropped = []

                def observe(sample):
                    rig.client._state_sub = SimpleNamespace(
                        age=lambda: 0, get_state=lambda: sample and sample[0],
                        get_state_with_timestamp=lambda: sample)
                    return rig.client._observe(lambda: dropped.append(1))

                self.assertIsNone(observe((None, time.monotonic())))
                self.assertIsNotNone(rig.client._instruction_hold_reason
                                     or rig.client._last_hold_reason)
                self.assertIsNone(observe((state, time.monotonic() - 5)))
                self.assertIn("stale", rig.client._last_hold_reason)
                # Fresh state but no WM goal for the current prompt.
                self.assertIsNone(observe((state, time.monotonic())))
                self.assertIn("WM goal", rig.client._last_hold_reason)
                self.assertEqual(len(dropped), 3)

                self.poll(rig, 'A')
                rig.wm._poll_once()
                got = observe((state, time.monotonic()))
                self.assertIsNotNone(got)
                self.assertEqual(got.instruction, 'Task: a')
                self.assertEqual(got.epoch, int(rig.wm.snapshot()['prompt_epoch']))
                self.assertEqual(got.goal.shape[:2], rig.client._goal_hw)
                self.assertEqual(len(dropped), 3)

    # ------------------------------------------------------------ hold safety

    def test_instruction_wait_keeps_publishing_hold_through_watchdog_ticks(self):
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            for trigger in ('manual', 'hlp', 'done'):
                with self.subTest(transport=transport.__name__, trigger=trigger):
                    rig = self.rig(transport, neck=True)
                    self.poll(rig, 'A')
                    _, last = self.goal_and_action(rig, 'A', 1)
                    if trigger == 'manual':
                        manual = InstructionManager(rig.client, rig.wm, prompt='A', next_prompts=['B'])
                        manual.command('next')
                    else:
                        self.poll(rig, 'B' if trigger == 'hlp' else None, done=trigger == 'done')
                    self.assertFalse(rig.client.condition_snapshot()['executing'])
                    before = rig.publisher.publish_token.call_count
                    started = time.monotonic()
                    for tick in range(45):
                        with patch.object(vla.time, 'monotonic', return_value=started + tick / 30):
                            rig.client._check_action_liveness()
                            rig.client.hold_tick()
                    self.assertTrue(rig.client._wbc_started)
                    self.assertEqual(rig.publisher.publish_token.call_count, before + 45)
                    self.assertFalse(any(call.kwargs.get('stop') for call in rig.publisher.send_command.call_args_list))
                    wire = rig.publisher.publish_token.call_args.args[0]
                    np.testing.assert_array_equal(wire[:64], rig.token)
                    np.testing.assert_array_equal(wire[64:], last[:14])
                    np.testing.assert_array_equal(rig.neck.publish.call_args.args, last[78:])

    def test_watchdog_still_stops_unexpected_action_loss_outside_hold(self):
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            with self.subTest(transport=transport.__name__):
                rig = self.rig(transport)
                self.poll(rig, 'A')
                self.goal_and_action(rig, 'A', 1)
                self.assertFalse(rig.client._holding)
                before = rig.publisher.publish_token.call_count
                rig.client._last_accepted_action_at = time.monotonic() - 2
                rig.client._check_action_liveness()
                self.assertFalse(rig.client._wbc_started)
                self.assertTrue(rig.publisher.send_command.call_args.kwargs['stop'])
                self.assertEqual(rig.publisher.publish_token.call_count, before)

    def test_hold_still_stops_when_robot_state_expires(self):
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            with self.subTest(transport=transport.__name__):
                rig = self.rig(transport)
                self.poll(rig, 'A')
                self.goal_and_action(rig, 'A', 1)
                rig.client.hold_instructions('waiting for rollback')
                rig.client._state_sub.age = lambda: 2.0
                before = rig.publisher.publish_token.call_count
                rig.client._check_action_liveness()
                rig.client.hold_tick()
                self.assertFalse(rig.client._running)
                self.assertTrue(rig.publisher.send_command.call_args.kwargs['stop'])
                self.assertEqual(rig.publisher.publish_token.call_count, before)

    def test_encoder_failure_never_replays_policy_action_as_a_transient_hold(self):
        rig = self.rig()
        self.poll(rig, 'A')
        self.goal_and_action(rig, 'A', 1)
        rig.client.hold_instructions('wait')
        rig.client._encoder_failed = True
        before = rig.publisher.publish_token.call_count
        rig.client.hold_tick()
        self.assertEqual(rig.publisher.publish_token.call_count, before)
        self.assertFalse(rig.client._running)
        self.assertTrue(rig.publisher.send_command.call_args.kwargs['stop'])

    def test_no_goal_mode_publishes_without_a_dataset_or_wm_image(self):
        from psix.wm import NoGoalProvider
        for transport in (vla.RTCWebSocketClient, vla.HttpChunkClient):
            with self.subTest(transport=transport.__name__):
                rig = self.rig(transport)
                rig.client._wm = NoGoalProvider(task='A')
                rig.client.set_instruction('A', reason='test')
                rig.client._observation_sent()
                result = rig.client._publish_action_for_prompt(
                    np.zeros((1, 78), np.float32), 1, None, {}, time.monotonic())
                self.assertEqual(result, 'ok')
                rig.publisher.publish_token.assert_called_once()

    # -------------------------------------------------------------- keyboard

    def test_single_key_p_needs_no_enter_and_terminal_is_restored(self):
        master, slave = pty.openpty()
        stream = os.fdopen(slave, 'r')
        import termios
        before = termios.tcgetattr(stream.fileno())
        running, called = threading.Event(), threading.Event()
        running.set()
        manager = SimpleNamespace(command=lambda name, text=None: called.set() if name == 'prev' else None)
        keyboard = KeyboardInput(manager, None, None, running, stream=stream)
        try:
            keyboard.start()
            os.write(master, b'p')
            self.assertTrue(called.wait(1))
            keyboard.stop()
            self.assertEqual(termios.tcgetattr(stream.fileno()), before)
        finally:
            keyboard.stop()
            stream.close()
            os.close(master)


if __name__ == '__main__':
    unittest.main()
