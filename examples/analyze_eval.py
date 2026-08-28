#!/usr/bin/env python3
"""Score recorded eval episodes against the checkpoint that produced them.

The recorder keeps every observation but computes no critic values, because a
critic forward per control step overruns the robot's 10 Hz budget. This runs
them here instead, batched and off-robot, and prints the comparisons that
distinguish this task's failure modes from one another.

    python examples/analyze_eval.py \
        --run_dir examples/experiments/tennis_ball_pick_and_place/<run> \
        --encoder_checkpoint examples/encoder_training/runs/<enc>/best.msgpack

Reads <run_dir>/eval_recordings/episode_*.npz and writes analysis.json beside
them. Pass --no_critic to skip the checkpoint entirely and get only the
quantities derivable from the recording itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, default=None)
    parser.add_argument("--encoder_checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Defaults to the step recorded in each episode.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_critic", action="store_true")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Only analyse the newest N episodes.")
    return parser.parse_args()


def softmax_maps(logits):
    flat = logits.reshape(len(logits), -1).astype(np.float64)
    weights = np.exp(flat - flat.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


def to_cells(mask_images, grid):
    """Binary mask -> the token grid the grounding loss scores."""
    import jax
    import jax.numpy as jnp

    mask = np.asarray(mask_images)
    while mask.ndim > 3:
        mask = mask[:, -1] if mask.shape[1] <= 4 else mask[..., 0]
    if mask.ndim == 4:
        mask = mask[..., 0]
    binary = (mask > 127).astype(np.float32)
    resized = np.asarray(jax.device_get(jax.image.resize(
        jnp.asarray(binary)[:, None], (len(binary), 1, *grid), method="linear")))[:, 0]
    return (resized > 0.04).astype(np.float64).reshape(len(binary), -1)


def load_agent(encoder_checkpoint, run_dir, step):
    import jax
    import numpy as np
    from flax.training import checkpoints

    from benchmark_learner_speed import ACTION_DIM, IMAGE_KEYS, make_sample_observation
    from serl_launcher.utils.launcher import (
        make_gaze_sac_pixel_agent_hybrid_single_arm,
    )

    agent = make_gaze_sac_pixel_agent_hybrid_single_arm(
        seed=0, sample_obs=make_sample_observation(1),
        sample_action=np.zeros((ACTION_DIM,), np.float32), image_keys=list(IMAGE_KEYS),
        encoder_type="vit-grounded", tactile_encoder_type="cnn", discount=0.97,
        mask_pick_place_phase_control=True,
        encoder_checkpoint_path=str(encoder_checkpoint) if encoder_checkpoint else None,
    )
    restored = checkpoints.restore_checkpoint(
        str(Path(run_dir).resolve()), agent.state, step=step
    )
    print(f"restored RL checkpoint step {step} from {run_dir}")
    return agent.replace(state=restored)


def critic_values(agent, data, batch_size):
    """Q at the taken action, and grasp Q at the taken / open / closed gripper."""
    import jax
    import jax.numpy as jnp

    keys = [k[len("obs_"):] for k in data.files if k.startswith("obs_")]
    if "state" not in keys:
        # Recordings made before the recorder stored full observations cannot
        # be re-scored: the encoder needs the masks and the phase one-hot.
        raise KeyError(
            "recording has no stored observations (obs_* keys); it predates "
            "full-observation recording and only supports --no_critic"
        )
    steps = len(data["action"])
    q, grasp = [], []
    for start in range(0, steps, batch_size):
        stop = min(start + batch_size, steps)
        obs = {"front_camera": jnp.asarray(data["frames"][start:stop])[:, None]}
        for key in keys:
            value = np.asarray(data[f"obs_{key}"][start:stop])
            obs[key] = jnp.asarray(value)
        action = jnp.asarray(data["action"][start:stop])
        rng = jax.random.PRNGKey(0)
        q.append(np.asarray(jax.device_get(
            agent.forward_critic(obs, action[:, :-1], rng=rng, train=False))).T)
        row = []
        for grip in (None, -1.0, 1.0):
            values = (action[:, -1:] if grip is None
                      else jnp.full((stop - start, 1), grip, jnp.float32))
            row.append(np.asarray(jax.device_get(
                agent.forward_grasp_critic(obs, values, rng=rng, train=False))).reshape(-1))
        grasp.append(np.stack(row, axis=-1))
    return np.concatenate(q), np.concatenate(grasp)


def describe(path, data, critic):
    state = data["state"]
    phase = state[:, -2:]
    pick = phase[:, 0] > 0.5
    action = data["action"]
    report = {
        "file": path.name,
        "steps": int(len(action)),
        "return": float(np.nansum(data["reward"])),
        "phase_flips": int((np.abs(np.diff(phase, axis=0)).sum(1) > 0).sum()),
        "pick_steps": int(pick.sum()),
        "place_steps": int((~pick).sum()),
    }

    if "attention_logits" in data.files:
        weights = softmax_maps(data["attention_logits"])
        grid = data["attention_logits"].shape[-2:]
        report["attention_inside_mean"] = float(np.mean(data["attention_inside"]))
        if "obs_front_camera_mask1" in data.files and "obs_front_camera_mask2" in data.files:
            ball = to_cells(data["obs_front_camera_mask1"], grid)
            basket = to_cells(data["obs_front_camera_mask2"], grid)
            other = np.where(pick[:, None], basket, ball)
            leak = (weights * other).sum(1)
            report["leak_to_other_object"] = float(leak.mean())
            if pick.any():
                report["leak_during_pick"] = float(leak[pick].mean())

    mean = data.get("pretanh_mean")
    if mean is not None:
        report["pretanh_abs_max"] = float(np.abs(mean).max())
        report["saturated_fraction"] = float(np.mean(np.abs(np.tanh(mean)) > 0.9))
    report["action_abs_mean"] = [round(float(v), 3) for v in np.abs(action).mean(0)]
    report["gripper_range"] = [round(float(action[:, -1].min()), 3),
                               round(float(action[:, -1].max()), 3)]

    if "wall_time" in data.files and len(data["wall_time"]) > 2:
        dt = np.diff(data["wall_time"])
        report["control_hz_median"] = round(float(1.0 / np.median(dt)), 2)
        report["control_dt_p05_p95_ms"] = [round(float(1000*np.percentile(dt, 5)), 1),
                                           round(float(1000*np.percentile(dt, 95)), 1)]

    tcp = state[:, :3]
    step_mm = np.linalg.norm(np.diff(tcp, axis=0), axis=1) * 1000
    report["tcp_step_mm_p50_p95_max"] = [round(float(np.percentile(step_mm, 50)), 2),
                                         round(float(np.percentile(step_mm, 95)), 2),
                                         round(float(step_mm.max()), 2)]
    centred = step_mm - step_mm.mean()
    # A period-2 alternation (negative lag-1, positive lag-2) means the arm is
    # not advancing one commanded increment per control step.
    report["tcp_step_autocorr_lag1"] = round(
        float(np.corrcoef(centred[:-1], centred[1:])[0, 1]), 3)

    if "tactile_stats" in data.files:
        tactile = data["tactile_stats"]
        report["tactile_sd_peak"] = [round(float(tactile[:, 1].max()), 2),
                                     round(float(tactile[:, 3].max()), 2)]
        live = (tactile[:, 1] + tactile[:, 3]) > 0
        report["tactile_contact_steps"] = int(live.sum())
        pick_live = live & pick
        report["tactile_contact_during_pick"] = int(pick_live.sum())

    if critic is not None:
        q, grasp = critic
        report["q_mean"] = round(float(q.mean()), 4)
        report["q_std_over_time"] = round(float(q.mean(1).std()), 4)
        report["q_ensemble_spread"] = round(float(q.std(1).mean()), 4)
        prefers_close = grasp[:, 2] > grasp[:, 1]
        report["grasp_prefers_close_fraction"] = round(float(prefers_close.mean()), 3)
        if pick.any():
            report["grasp_prefers_close_during_pick"] = round(
                float(prefers_close[pick].mean()), 3)
        report["grasp_q_gap_close_minus_open"] = round(
            float((grasp[:, 2] - grasp[:, 1]).mean()), 4)
    return report


def main():
    args = parse_args()
    directory = args.recordings or (args.run_dir / "eval_recordings")
    files = sorted(directory.glob("episode_*.npz"))
    if args.episodes:
        files = files[-args.episodes:]
    if not files:
        raise SystemExit(f"no episode_*.npz under {directory}")
    print(f"{len(files)} episode(s) in {directory}")

    agent, loaded_step = None, None
    reports = []
    for path in files:
        data = np.load(path)
        critic = None
        if not args.no_critic:
            step = args.checkpoint_step
            if step is None:
                summary = json.loads((directory / "summary.json").read_text())
                match = next((r for r in summary if r.get("npz") == path.name), {})
                step = match.get("checkpoint_step")
            if step is None:
                print(f"  {path.name}: no checkpoint step known, skipping critic")
            else:
                if agent is None or loaded_step != step:
                    agent = load_agent(args.encoder_checkpoint, args.run_dir, step)
                    loaded_step = step
                try:
                    critic = critic_values(agent, data, args.batch_size)
                except KeyError as error:
                    print(f"  {path.name}: {error}")
        report = describe(path, data, critic)
        reports.append(report)
        print(f"\n=== {path.name} ===")
        for key, value in report.items():
            if key != "file":
                print(f"  {key:<34} {value}")

    (directory / "analysis.json").write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {directory / 'analysis.json'}")


if __name__ == "__main__":
    main()
