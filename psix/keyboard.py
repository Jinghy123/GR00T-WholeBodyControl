"""Single-key terminal input. Enter moves on, p goes back, i types an instruction."""

import os
import select
import sys
import termios
import threading
import tty


class KeyboardInput:
    def __init__(self, instructions, wm, client, running, stream=None):
        self.instructions, self.wm, self.client = instructions, wm, client
        self.running = running
        self.stream = sys.stdin if stream is None else stream
        self._stop = threading.Event()
        self._thread = None
        self._terminal = None

    def dispatch(self, line):
        command = line.strip()
        try:
            if command == "":
                self.instructions.command("next")
            elif command == "p":
                self.instructions.command("prev")
            elif command == "q":
                self.running.clear()
            elif command.startswith(":sec "):
                value = command[5:].strip()
                self.wm.set_seconds(None if value == "server" else float(value))
            elif command.startswith(":"):
                print(f"[prompt] unknown command {command!r}", flush=True)
            else:
                self.instructions.command("override", command)
        except (ValueError, RuntimeError, AttributeError) as exc:
            print(f"[prompt] command failed: {exc}", flush=True)

    def _read(self):
        fd = self.stream.fileno()
        editing = not self.stream.isatty()
        buffer = bytearray()
        try:
            while self.running.is_set() and not self._stop.is_set():
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                key = os.read(fd, 1)
                if not key:
                    break
                if key in (b"\r", b"\n"):
                    self.dispatch(buffer.decode("utf-8", errors="replace"))
                    buffer.clear()
                    editing = not self.stream.isatty()
                    if self.stream.isatty():
                        print(flush=True)
                elif key == b"\x1b":
                    buffer.clear()
                    editing = False
                elif editing:
                    if key in (b"\x7f", b"\x08"):
                        buffer[:] = buffer.decode("utf-8", errors="ignore")[:-1].encode("utf-8")
                    else:
                        buffer.extend(key)
                    if self.stream.isatty():
                        sys.stdout.write("\r[prompt] > " + buffer.decode("utf-8", errors="replace") + "\033[K")
                        sys.stdout.flush()
                elif key == b"i":
                    editing = True
                    print("[prompt] type instruction, Enter to apply, Esc to cancel:", flush=True)
                elif key == b":":
                    editing = True
                    buffer.extend(key)
                elif key in (b"p", b"q"):
                    self.dispatch(key.decode())
        finally:
            self._restore()

    def start(self):
        try:
            fd = self.stream.fileno()
            if self.stream.isatty():
                self._terminal = termios.tcgetattr(fd)
                tty.setcbreak(fd)
        except (AttributeError, OSError, ValueError):
            print("[client] terminal input unavailable", flush=True)
            return
        self._thread = threading.Thread(target=self._read, name="keyboard", daemon=True)
        self._thread.start()

    def _restore(self):
        if self._terminal is not None:
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._terminal)
            self._terminal = None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._restore()
