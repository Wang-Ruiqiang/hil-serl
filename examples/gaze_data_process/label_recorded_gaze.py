import json
from pathlib import Path
import sys

from absl import app, flags

project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "serl_robot_infra").exists()
)
sys.path.insert(0, str(project_root / "serl_robot_infra"))
sys.path.insert(0, str(project_root))

from franka_env.gaze.sam2_labeler import SAM2GazeLabeler


FLAGS = flags.FLAGS
flags.DEFINE_string("metadata", "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-5-27-0/recording_metadata.json", "Path to recording_metadata.json.")
flags.DEFINE_integer("sam2_batch_episodes", 1, "Number of episodes per SAM2 labeling batch.")
flags.DEFINE_integer("sam2_y_prompts_et", 20, "ET keyframes to annotate per episode.")
flags.DEFINE_integer("sam2_y_prompts_rs", 10, "RS keyframes to annotate per episode.")
flags.DEFINE_integer("start_episode", 0, "0-based metadata episode_index to start labeling from.")
flags.DEFINE_boolean("success_only", True, "Label only successful episodes from metadata.")


def _load_ranges(metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    records = metadata.get("episode_ranges", [])
    if FLAGS.success_only:
        records = [rec for rec in records if bool(rec.get("success", False))]

    episodes = []
    for rec in records:
        kept_ranges = rec.get("kept_frame_ranges") or [
            {
                "start_frame": rec["start_frame"],
                "end_frame": rec["end_frame"],
            }
        ]
        ranges = [
            (int(rng["start_frame"]), int(rng["end_frame"]))
            for rng in kept_ranges
            if int(rng["end_frame"]) >= int(rng["start_frame"])
        ]
        if not ranges:
            continue
        episodes.append(
            {
                "episode_index": int(rec.get("episode_index", len(episodes))),
                "ranges": ranges,
            }
        )
    return metadata, episodes


def main(_):
    if FLAGS.metadata is None:
        raise ValueError("Please pass --metadata=/path/to/recording_metadata.json")

    metadata_path = Path(FLAGS.metadata).expanduser().resolve()
    metadata, episodes = _load_ranges(metadata_path)
    if FLAGS.start_episode > 0:
        episodes = [ep for ep in episodes if ep["episode_index"] >= FLAGS.start_episode]
    if not episodes:
        print(f"[label_recorded_gaze] no episode ranges found in {metadata_path}")
        return

    frame_root = Path(metadata.get("frame_root", metadata_path.parent)).expanduser().resolve()
    labeler = SAM2GazeLabeler(
        frame_root=str(frame_root),
        et_mirror_dir=str(frame_root / "et_images"),
        rs_mirror_dir=str(frame_root / "rs_images"),
    )

    batch_size = max(1, int(FLAGS.sam2_batch_episodes))
    for start in range(0, len(episodes), batch_size):
        chunk = episodes[start : start + batch_size]
        frame_ranges = [rng for ep in chunk for rng in ep["ranges"]]
        prompt_frame_groups = [ep["ranges"] for ep in chunk]
        print(
            f"[label_recorded_gaze] labeling batch {start // batch_size + 1}: "
            f"episodes={[ep['episode_index'] for ep in chunk]} ranges={frame_ranges}"
        )
        labeler.run_inline_v2(
            frame_ranges=frame_ranges,
            prompt_frame_groups=prompt_frame_groups,
            y_prompts_et=FLAGS.sam2_y_prompts_et,
            y_prompts_rs=FLAGS.sam2_y_prompts_rs,
            rs_select_uses_same_keyset=False,
            random_seed=42 + start,
        )


if __name__ == "__main__":
    app.run(main)
