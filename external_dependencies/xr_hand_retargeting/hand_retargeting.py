"""Local copy of xr_teleoperate hand retargeting, vendored into
GR00T-WholeBodyControl so no other repo needs to be on sys.path.

Differences from the upstream xr_teleoperate version:
  - Paths are resolved relative to this file (no cwd/chdir tricks).
  - ``logging_mp`` is replaced with the stdlib ``logging`` module so the
    package has no third-party logging dependency.
"""

from pathlib import Path
from enum import Enum
import logging
import sys

import yaml

# Make the vendored dex_retargeting importable as a top-level package
# (its internal modules import each other via ``from dex_retargeting...``).
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from dex_retargeting import RetargetingConfig  # noqa: E402

logger_mp = logging.getLogger(__name__)

_ASSETS_DIR = _PKG_DIR / "assets"


class HandType(Enum):
    UNITREE_DEX3 = str(_ASSETS_DIR / "unitree_hand" / "unitree_dex3.yml")


class HandRetargeting:
    def __init__(self, hand_type: HandType):
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_DIR))

        config_file_path = Path(hand_type.value)

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)

            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
            self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

            if hand_type == HandType.UNITREE_DEX3:
                # "Sort by message structure" of
                # https://support.unitree.com/home/en/G1_developer/dexterous_hand
                self.left_dex3_api_joint_names = [
                    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
                    'left_hand_index_0_joint', 'left_hand_index_1_joint',
                ]
                self.right_dex3_api_joint_names = [
                    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                    'right_hand_index_0_joint', 'right_hand_index_1_joint',
                ]
                self.left_dex_retargeting_to_hardware = [
                    self.left_retargeting_joint_names.index(name)
                    for name in self.left_dex3_api_joint_names
                ]
                self.right_dex_retargeting_to_hardware = [
                    self.right_retargeting_joint_names.index(name)
                    for name in self.right_dex3_api_joint_names
                ]

        except FileNotFoundError:
            logger_mp.warning(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.warning(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise
