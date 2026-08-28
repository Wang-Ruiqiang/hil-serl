"""Task-specific settings for standalone encoder pretraining.

Add a new entry here when a task has a different set of target regions. The
training script only needs ``--exp_name`` after that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


DATA_ROOT = Path("/home/ealin/workspaces/DexTacHil/data/recorded_data")


@dataclass(frozen=True)
class EncoderTaskConfig:
    data_root: Path
    target_mode: str = "mask"
    mask_files: Tuple[str, ...] = ("ball_mask.png",)
    target_names: Tuple[str, ...] = ("target",)
    gaze_json_name: str = "gaze_contact.json"
    gaze_window: int = 8
    gaze_sigma: float = 0.06


TASK_CONFIGS = {
    # The semantic filenames are canonical. The dataset loader still accepts
    # legacy mask1.png/mask2.png files for older recordings.
    "tennis_ball_pick": EncoderTaskConfig(
        data_root=DATA_ROOT / "tennis_ball_pick",
        mask_files=("ball_mask.png",),
        target_names=("ball",),
    ),
    "tennis_ball_pick_and_place": EncoderTaskConfig(
        data_root=DATA_ROOT / "tennis_ball_pick_and_place",
        mask_files=("ball_mask.png", "hand_mask.png", "basket_mask.png"),
        target_names=("ball", "hand", "basket"),
    ),
}


def get_task_config(exp_name: str) -> EncoderTaskConfig:
    try:
        return TASK_CONFIGS[exp_name]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_CONFIGS))
        raise ValueError(
            f"Unknown encoder task {exp_name!r}. Add it to task_configs.py. "
            f"Known tasks: {known}"
        ) from exc


def list_task_names() -> Tuple[str, ...]:
    return tuple(sorted(TASK_CONFIGS))
