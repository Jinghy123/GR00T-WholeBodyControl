import os
import cv2
import json
import numpy as np
import time
from queue import Queue, Empty
from threading import Thread


# Maps color-stream keys (passed in by the recorder) to:
#   (subdirectory under episode_dir, field name in frame_record)
# The "rgb" → "image" mapping is preserved so existing replay*.py scripts
# continue to work unchanged.
_COLOR_STREAMS = {
    "rgb":         ("color",             "image"),
    "stereo":      ("color_stereo",      "stereo_image"),
    "left_wrist":  ("color_left_wrist",  "left_wrist_image"),
    "right_wrist": ("color_right_wrist", "right_wrist_image"),
}


class EpisodeWriter():
    def __init__(self, task_dir, date, episode_num, task, frequency=30, image_size=[640, 480]):
        """
        image_size: [width, height]
        """
        print("==> EpisodeWriter initializing...\n")
        self.episode_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size

        self.data = {}
        self.episode_data = []
        self.frame_id = 0

        self.date = date
        self.episode_num = episode_num
        self.task = task

        self.is_available = True
        self.item_data_queue = Queue(maxsize=100)
        self.stop_worker = False
        self.need_save = False
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        print("==> EpisodeWriter initialized successfully.\n")

    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        """
        if not self.is_available:
            print("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False

        self.episode_data = []
        self.timestamps = []

        # Create one subdir per color stream (ego rgb, ego stereo, wrists).
        self.color_dirs = {
            key: os.path.join(self.episode_dir, subdir)
            for key, (subdir, _) in _COLOR_STREAMS.items()
        }
        for d in self.color_dirs.values():
            os.makedirs(d, exist_ok=True)
        # Kept for backward compat with any code reading self.color_dir.
        self.color_dir = self.color_dirs["rgb"]

        self.json_path = os.path.join(self.episode_dir, 'data.json')

        self.is_available = False
        print(f"==> New episode created: {self.episode_dir}")
        return True

    def add_item(self, colors, states=None, actions=None, token=None):
        item_data = {
            'timestamp': time.time_ns(),
            'colors': colors,
            'states': states,
            'actions': actions,
            'token': token,
        }
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                    self.frame_id += 1
                except Exception as e:
                    print(f"Error processing item_data: {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass

            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        colors = item_data.get('colors', {})

        frame_record = {
            'timestamp': item_data['timestamp'],
            'states': None,
            'actions': None,
        }

        if colors:
            self.timestamps.append(item_data['timestamp'])
            fname = f"frame_{self.frame_id:06d}.jpg"
            for color_key, color in colors.items():
                stream = _COLOR_STREAMS.get(color_key)
                if stream is None:
                    print(f"==> Unknown color stream '{color_key}', skipping.")
                    continue
                subdir, record_key = stream
                fpath = os.path.join(self.color_dirs[color_key], fname)
                cv2.imwrite(fpath, color)
                frame_record[record_key] = f"{subdir}/{fname}"

        if item_data.get('states') is not None:
            states = item_data['states']
            frame_record['states'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in states.items()
            }

        if item_data.get('actions') is not None:
            actions = item_data['actions']
            actions_dict = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in actions.items()
            }
            # Add token to actions (token is part of what generated the action)
            if item_data.get('token') is not None:
                token = item_data['token']
                if isinstance(token, np.ndarray) and token.size > 0:
                    actions_dict['token'] = token.tolist()
                else:
                    actions_dict['token'] = []
            frame_record['actions'] = actions_dict

        if item_data.get('token') is not None:
            # Also keep token at top level for convenience (optional)
            token = item_data['token']
            if isinstance(token, np.ndarray) and token.size > 0:
                frame_record['token'] = token.tolist()
            else:
                frame_record['token'] = []

        self.episode_data.append(frame_record)

    def save_episode(self):
        """Trigger the save operation."""
        self.need_save = True
        print(f"==> Episode save triggered...")

    def _save_episode(self):
        """Save episode data to data.json as a list of timestep dicts."""
        with open(self.json_path, 'w') as f:
            json.dump(self.episode_data, f, indent=2)

        self.need_save = False
        self.is_available = True
        print(f"==> Episode saved successfully to {self.json_path} ({len(self.episode_data)} frames)")

    def close(self):
        """Stop the worker thread and ensure all tasks are completed."""
        self.item_data_queue.join()
        self.stop_worker = True
        self.worker_thread.join()
