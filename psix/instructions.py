"""The one owner of the current instruction: manual sequence, HLP, and rollback."""

import queue
import threading
import time
from copy import deepcopy


class InstructionManager:
    """Runs the HLP poll loop on its own thread and applies whatever the server
    reports. Operator commands are queued onto that thread so no HTTP request
    ever runs on the keyboard thread."""

    def __init__(self, client, observations, *, prompt, next_prompts=(), mode="manual", hlp=None,
                 log_event=lambda *a, **kw: None):
        if mode not in ("manual", "hlp", "shadow"):
            raise ValueError(f"unknown instruction mode: {mode}")
        if mode != "manual" and hlp is None:
            raise ValueError("HLP mode needs an HlpClient")
        self.client, self.observations, self.hlp = client, observations, hlp
        self.mode, self.prompt = mode, prompt
        self._sequence = [prompt, *next_prompts]
        self._index = 0
        self._current = None if mode == "hlp" else prompt
        self._source = "waiting for HLP" if mode == "hlp" else "manual"
        self._planner = {}      # the HLP's last reply: what it wants executed and why
        self._revision = None
        self._epoch = None
        self._key = None
        self._history = []
        self._paused = False
        self._done = False
        self._error = None
        self._polls_ok = self._polls_failed = 0
        self._log = log_event
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._commands = queue.Queue()
        self._thread = None
        if mode == "hlp":
            client.hold_instructions("waiting for the first HLP instruction")

    # -------------------------------------------------------------- reporting

    def snapshot(self):
        with self._lock:
            return deepcopy({
                "mode": self.mode, "macro_prompt": self.prompt, "current": self._current,
                "source": self._source, "paused": self._paused, "done": self._done,
                "error": self._error, "revision": self._revision,
                "can_prev": self._can_prev(),
                "next_prompts": self._sequence[self._index + 1:],
                "switches": self._history[-100:], "hlp": self._planner,
                "polls_ok": self._polls_ok, "polls_failed": self._polls_failed})

    def _can_prev(self):
        return bool(self._planner.get("can_undo")) if self.mode == "hlp" else self._index > 0

    def _note(self, kind, source, before, text, **fields):
        event = {"time": time.time(), "kind": kind, "source": source, "from": before, "to": text, **fields}
        self._history.append(event)
        self._log("instruction_change", operation=kind, **{k: v for k, v in event.items() if k != "kind"})
        print(f"[prompt] {kind.upper()} ({source}): {before!r} -> {text!r}", flush=True)

    # --------------------------------------------------------------- commands

    def command(self, name, text=None):
        """Enter (next), p (prev), and typed text (override). Returns False when
        the command does not apply."""
        with self._lock:
            if self._stop.is_set():
                return False
            if name == "next" and self.mode != "hlp":
                if self._index + 1 >= len(self._sequence):
                    print("[prompt] no next instruction configured (--next-prompt)", flush=True)
                    return False
                name, text = "override", self._sequence[self._index + 1]
            if name == "override":
                text = str(text or "").strip()
                if not text:
                    raise ValueError("instruction must not be empty")
            elif name not in ("next", "prev"):
                raise ValueError(f"unknown instruction command: {name}")
            if name == "prev" and not self._can_prev():
                print("[prompt] no previous instruction", flush=True)
                return False
            if self.mode != "hlp":
                self._apply_manual(name, text)
            elif name == "next" and not self._done:
                # Nothing to synchronize: the robot is already running whatever
                # the last planner established, only polling was paused.
                self._paused = False
                self._wake.set()
            else:
                self._error = None
                self.client.hold_instructions(f"operator {name}: synchronizing HLP")
                self._commands.put((name, text))
                self._wake.set()
            return True

    def _apply_manual(self, name, text):
        before = self._current
        if name == "prev":
            self._index -= 1
            text = self._sequence[self._index]
        elif self._index + 1 < len(self._sequence) and text == self._sequence[self._index + 1]:
            self._index += 1
        self.client.set_instruction(text, reason="operator")
        self._current = text
        self._note("rollback" if name == "prev" else "override", "operator", before, text)

    def _run_command(self, name, text):
        """`prev` and `override` change the plan; `next` only clears a done latch."""
        try:
            if name == "prev":
                result = self.hlp.control("undo")
            elif name == "next":
                result = self.hlp.control("resume")
            else:
                result = self.hlp.control("override", subtask=text)
            with self._lock:
                self._planner = result.reply
                self._apply(result.reply, source="operator", force=True)
                self.hlp.finish(result, "applied", command=name)
                self._paused = name != "next"
                self._error = None
                if self._paused:
                    print("[prompt] HLP paused; Enter resumes", flush=True)
        except Exception as exc:
            # The next poll re-reads the server's state, so a lost planner needs no
            # retry bookkeeping here: press the key again if nothing changed.
            with self._lock:
                self._paused = False
            self._fail(exc)

    # ---------------------------------------------------------------- polling

    def poll_once(self):
        if self.mode == "manual":
            return
        with self._lock:
            if self._paused or self._stop.is_set():
                return
        result = None
        try:
            rgb, captured = self.observations.latest_ego()
            result = self.hlp.poll(rgb, self._executed_revision(captured))
            with self._lock:
                if self._stop.is_set():
                    self.hlp.finish(result, "superseded")
                    return
                self._planner = result.reply
                self._error = None
                self._polls_ok += 1
                if self.mode == "shadow":
                    self.hlp.finish(result, "shadow")
                    return
                self.hlp.finish(result, "applied" if self._apply(result.reply) else "continue")
        except Exception as exc:
            if result is not None:
                self.hlp.finish(result, "transition_error", error=str(exc))
            self._fail(exc)

    def _executed_revision(self, captured):
        """The revision the robot has actually started running, reported only from
        a frame captured after that start. The planner needs both to advance."""
        if self.mode == "shadow":
            return self._revision
        status = self.client.condition_snapshot()
        started = status["execution_started_at"]
        seen = started is not None and captured is not None and captured >= started
        return self._revision if status["executing"] and seen else None

    def _apply(self, planner, *, source="hlp", force=False):
        key = (planner["revision"], planner["instruction"])
        if key == self._key and not force:
            return False
        before, text = self._current, planner["instruction"]
        if text is None:
            self.client.hold_instructions(planner["reason"] or "HLP planning")
        else:
            # The VLA keeps the instruction-level sentence; an atomic step only
            # steers the WM goal image (and the optional Subtask clause).
            self.client.set_instruction(text, reason=source, task=planner["parent"] or text)
        self._current, self._source, self._done = text, source, planner["done"]
        self._key, self._revision = key, planner["revision"]
        self._epoch = self.client.condition_snapshot()["wm"]["prompt_epoch"]
        if self._done:
            self._paused = True
            print("[hlp] episode done; Enter moves on, p reopens the last instruction", flush=True)
        self._note(planner.get("committed") or ("hold" if text is None else "switch"), source, before, text,
                   revision=planner["revision"], level=planner["level"], parent=planner["parent"],
                   reason=planner["reason"], wm_prompt_epoch=self._epoch)
        return True

    def _fail(self, exc):
        with self._lock:
            if self._key is not None:
                self.client.hold_instructions("HLP unreachable")
            self._key = None
            self._error = str(exc)
            self._polls_failed += 1
        print(f"[hlp] failed: {exc}", flush=True)

    # ---------------------------------------------------------------- lifecycle

    def _loop(self):
        try:
            while not self._stop.is_set() and not self._planner:
                try:
                    result = self.hlp.reset()
                    with self._lock:
                        self._planner = result.reply
                    self.hlp.finish(result, "episode_started")
                    self._log("hlp_episode_started", task=self.prompt, state=result.reply)
                    print(f"[hlp] episode started: {self.prompt}", flush=True)
                except Exception as exc:
                    self._fail(exc)
                    self._stop.wait(2)
            while not self._stop.is_set():
                started = time.monotonic()
                self._wake.clear()
                try:
                    name, text = self._commands.get_nowait()
                except queue.Empty:
                    self.poll_once()
                    self._wake.wait(max(0, self.hlp.config.period - (time.monotonic() - started)))
                else:
                    self._run_command(name, text)
        finally:
            self.hlp.close()

    def start(self):
        if self.hlp is not None:
            self._thread = threading.Thread(target=self._loop, name="instructions", daemon=True)
            self._thread.start()

    def cancel(self):
        self._stop.set()
        self._wake.set()

    def stop(self):
        self.cancel()
        if self._thread is not None:
            self._thread.join(self.hlp.config.timeout + 1)
        elif self.hlp is not None:
            self.hlp.close()
