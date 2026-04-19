"""
Desktop client for unified_server.py.

Displays live camera feeds via cv2.imshow and optionally saves every frame
to disk when --save-dir is provided (JPEG: writes server JPEG bytes directly,
no re-encode — faster than PNG).

Receives a 4-part ZMQ multipart reply:
  Part 0 — Ego left RGB JPEG
  Part 1 — Ego stereo JPEG (side-by-side L|R)
  Part 2 — Left wrist RGB JPEG
  Part 3 — Right wrist RGB JPEG

Usage:
  python image_client_new.py --server 192.168.123.164 --port 5558
  python image_client_new.py --server 192.168.123.164 --port 5558 --save-dir ./captures
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import zmq


def start_client(args: argparse.Namespace) -> None:
    context = zmq.Context()
    addr = f"tcp://{args.server}:{args.port}"

    print(f"Connected to server at {addr}")
    print("Press 'q' to exit.")

    save_dirs: dict[str, str] | None = None
    if args.save_dir:
        base = os.path.abspath(args.save_dir)
        save_dirs = {
            "ego_rgb": os.path.join(base, "ego_rgb"),
            "ego_stereo": os.path.join(base, "ego_stereo"),
            "wrist_left": os.path.join(base, "wrist_left"),
            "wrist_right": os.path.join(base, "wrist_right"),
        }
        for d in save_dirs.values():
            os.makedirs(d, exist_ok=True)
        print(f"Saving frames to {base}")

    frame_idx = 0

    def _make_sock():
        s = context.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, 5000)
        s.setsockopt(zmq.SNDTIMEO, 5000)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(addr)
        return s

    sock = _make_sock()

    try:
        while True:
            try:
                sock.send(b"get")
                reply = sock.recv_multipart()
            except zmq.Again:
                print("Server not responding, reconnecting...")
                sock.close()
                sock = _make_sock()
                continue

            if len(reply) < 4:
                continue

            rgb_bytes = reply[0]
            stereo_bytes = reply[1]
            lw_rgb_bytes = reply[2]
            rw_rgb_bytes = reply[3]

            if not rgb_bytes:
                continue

            rgb_img = cv2.imdecode(
                np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR
            )

            if rgb_img is not None:
                cv2.imshow("Ego RGB (ZED)", rgb_img)

            lw_img = None
            if lw_rgb_bytes:
                lw_img = cv2.imdecode(
                    np.frombuffer(lw_rgb_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if lw_img is not None:
                    cv2.imshow("Left Wrist RGB", lw_img)

            rw_img = None
            if rw_rgb_bytes:
                rw_img = cv2.imdecode(
                    np.frombuffer(rw_rgb_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if rw_img is not None:
                    cv2.imshow("Right Wrist RGB", rw_img)

            stereo_img = None
            if stereo_bytes:
                stereo_img = cv2.imdecode(
                    np.frombuffer(stereo_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if args.show_stereo and stereo_img is not None:
                    cv2.imshow("Ego Stereo (L|R)", stereo_img)

            if save_dirs is not None:
                stem = f"{frame_idx:04d}"
                # Server already sends JPEG; write bytes directly (faster than PNG or re-encode).
                if rgb_bytes:
                    with open(
                        os.path.join(save_dirs["ego_rgb"], f"{stem}.jpg"), "wb"
                    ) as f:
                        f.write(rgb_bytes)
                if stereo_bytes:
                    with open(
                        os.path.join(save_dirs["ego_stereo"], f"{stem}.jpg"), "wb"
                    ) as f:
                        f.write(stereo_bytes)
                if lw_rgb_bytes:
                    with open(
                        os.path.join(save_dirs["wrist_left"], f"{stem}.jpg"), "wb"
                    ) as f:
                        f.write(lw_rgb_bytes)
                if rw_rgb_bytes:
                    with open(
                        os.path.join(save_dirs["wrist_right"], f"{stem}.jpg"), "wb"
                    ) as f:
                        f.write(rw_rgb_bytes)
                frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        cv2.destroyAllWindows()
        sock.close()
        context.term()
        if save_dirs is not None:
            print(f"Saved {frame_idx} frame(s) to {os.path.abspath(args.save_dir)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Desktop client: live display + optional save for unified_server"
    )
    p.add_argument(
        "--server", default="192.168.0.129",
        help="Server IP address",
    )
    p.add_argument(
        "--port", type=int, default=5558,
        help="Server ZMQ port",
    )
    p.add_argument(
        "--save-dir", default=None,
        help="If set, save every frame as .jpg under 4 subfolders (server JPEG, no re-encode)",
    )
    p.add_argument(
        "--show-stereo", action="store_true",
        help="Also display the stereo (L|R) view window",
    )
    args = p.parse_args()
    start_client(args)


if __name__ == "__main__":
    main()
