"""HLP HTTP client and request recording. No robot, WM, or keyboard ownership."""

import cv2
import numpy as np
import requests
from base64 import b64encode
from dataclasses import dataclass

# The four macro prompts the HLP was trained on, verbatim from the training pack's
# composition.json. One line each on purpose: the text on the wire must be exact.
MACRO_PROMPTS = {
    "serve_clean": "Serve the fruits and clean the table.",
    "serve_drink_clean": "Serve the fruits and the drink, and clean the table.",
    "serve_clean_collect": "Serve the fruits and clean the table, then collect clothes into the laundry basket.",
    "serve_drink_clean_collect": "Serve the fruits and the drink, and clean the table, then collect clothes into the laundry basket.",
}
DEFAULT_MACRO = "serve_clean"


def resolve_macro(value):
    """Return (sentence, preset_key); custom text passes through unchanged."""
    v = str(value or "").strip()
    if not v:
        return MACRO_PROMPTS[DEFAULT_MACRO], DEFAULT_MACRO
    if v in MACRO_PROMPTS:
        return MACRO_PROMPTS[v], v
    for key, sentence in MACRO_PROMPTS.items():
        if v == sentence:
            return sentence, key
    return v, None


@dataclass(frozen=True)
class HlpConfig:
    task: str
    host: str = "127.0.0.1"
    port: int = 8015
    period: float = 1.0
    timeout: float = 20.0
    jpeg_quality: int = 90
    atomic: bool = False

    def __post_init__(self):
        if not self.task.strip():
            raise ValueError("HLP macro prompt must not be empty")
        if not 1 <= self.port <= 65535 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("invalid HLP port or JPEG quality")
        if min(self.period, self.timeout) <= 0:
            raise ValueError("HLP timing settings must be positive")

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"


@dataclass
class HlpResult:
    reply: dict
    record: object


class HlpError(RuntimeError):
    pass


class HlpClient:
    """One sequential HTTP connection, used only by the instruction worker."""

    def __init__(self, config, recorder):
        self.config = config
        self.recorder = recorder
        self.state = {}
        self._http = requests.Session()
        self._http.trust_env = False

    def request(self, path, body):
        # What the planner said last time, stored alongside this request so a
        # recorded exchange can be read without replaying the whole run.
        previous = {k: self.state.get(k) for k in ("revision", "instruction", "level", "parent")}
        record = self.recorder.begin("hlp", self.config.base_url + path, body, context=previous)
        try:
            response = self._http.post(self.config.base_url + path, json=body, timeout=self.config.timeout)
            reply = response.json()
        except (requests.RequestException, ValueError) as exc:
            self.recorder.finish(record, "transport_error", error=str(exc))
            raise HlpError(f"{path}: {exc}") from exc
        self.recorder.response(record, response.status_code, reply)
        if response.status_code != 200 or "error" in reply:
            error = f"{path}: HTTP {response.status_code} {reply.get('error', reply)}"
            self.recorder.finish(record, "server_error", error=error)
            raise HlpError(error)
        self.state = reply
        return HlpResult(reply, record)

    def reset(self):
        return self.request("/reset", {"task": self.config.task, "hierarchical": self.config.atomic})

    def poll(self, rgb, executed_revision):
        if rgb is None:
            raise HlpError("no camera frame for HLP")
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        if not ok:
            raise HlpError("could not encode HLP observation")
        return self.request("/hlp", {
            "task": self.config.task, "executed_revision": executed_revision,
            "ego_image": {"jpeg_b64": b64encode(encoded.tobytes()).decode("ascii"),
                          "quality": self.config.jpeg_quality}})

    def control(self, name, **body):
        return self.request("/" + name, body)

    def finish(self, result, outcome, **details):
        self.recorder.finish(result.record, outcome, **details)

    def close(self):
        self._http.close()
