#!/usr/bin/env python3
"""Render the attention the RL agent actually sees, on real demo frames.

visualize_encoder.py goes through ViTPretrainModel and draws one panel per
segmentation channel, so it shows the objects separately and never shows the
single merged map the policy is handed. This script goes through the RL path
instead -- the real agent, the real ViTGroundedEncodingWrapper, observations
taken straight out of the demo pickle -- and draws that one map.

It uses the same display transform as the live actor overlay
(`_attention_heatmap_overlay` with mode="softmax"), so what appears here is
what the actor window shows during a run.

    python examples/encoder_training/visualize_rl_attention.py \
        --checkpoint examples/encoder_training/runs/<run>/best.msgpack \
        --demo_path examples/demo_data/<demos>.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--per_phase", type=int, default=12,
                        help="Frames to render from each of pick and place.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=256)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.35,
        help="Extra panel with attention**gamma. The linear panel matches the "
             "actor window but hides the secondary object: the ball occupies "
             "~4 cells and the basket ~40, so in pick the peak is 6x higher "
             "and the hand renders at ~10%% of it (invisible on JET) even "
             "though it holds more attention mass there than in place.",
    )
    return parser.parse_args()


def label(image, text, height=18):
    """Caption along the bottom.

    The top is already taken: _attention_heatmap_overlay stamps "peak p=" there
    at the source resolution, and a bar drawn after the upscale would cover it.
    """
    out = image.copy()
    bottom = out.shape[0]
    cv2.rectangle(out, (0, bottom - height), (out.shape[1], bottom), (0, 0, 0), -1)
    cv2.putText(out, text, (4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp

    from benchmark_learner_speed import (
        ACTION_DIM,
        IMAGE_KEYS,
        make_sample_observation,
    )
    from serl_launcher.utils.launcher import (
        make_gaze_sac_pixel_agent_hybrid_single_arm,
    )
    from train_rlpd import _attention_heatmap_overlay

    out_dir = args.out_dir or (args.checkpoint.parent / "rl_attention")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.demo_path, "rb") as handle:
        demos = pickle.load(handle)

    agent = make_gaze_sac_pixel_agent_hybrid_single_arm(
        seed=args.seed,
        sample_obs=make_sample_observation(1),
        sample_action=np.zeros((ACTION_DIM,), dtype=np.float32),
        image_keys=list(IMAGE_KEYS),
        encoder_type="vit-grounded",
        tactile_encoder_type="cnn",
        discount=0.97,
        mask_pick_place_phase_control=True,
        encoder_checkpoint_path=str(args.checkpoint),
    )

    # Truncate first, then read the one-hot off the end. Slicing a fixed
    # offset instead would silently read a proprio column once the demos are
    # re-recorded at the narrower state width.
    state_dim = make_sample_observation(1)["state"].shape[-1]
    phases = np.stack([
        np.asarray(t["observations"]["state"])[-1, :state_dim][-2:] for t in demos
    ])
    rng = np.random.default_rng(args.seed)
    chosen = []
    for name, column in (("pick", 0), ("place", 1)):
        pool = np.flatnonzero(phases[:, column] > 0.5)
        take = rng.choice(pool, min(args.per_phase, len(pool)), replace=False)
        chosen.extend((name, int(i)) for i in sorted(take))

    observations = {}
    for key in demos[0]["observations"]:
        stacked = np.stack(
            [np.asarray(demos[i]["observations"][key]) for _, i in chosen]
        )
        # Demos predate the two-slot phase one-hot; drop the trailing dead
        # column, exactly as train_rlpd does when filling the buffers.
        if key == "state" and stacked.shape[-1] > state_dim:
            stacked = stacked[..., :state_dim]
        observations[key] = jnp.asarray(stacked)
    _, attention = agent.state.apply_fn(
        {"params": agent.state.params},
        observations,
        name="actor",
        train=False,
        return_attention=True,
    )
    attention = np.asarray(jax.device_get(attention))

    # The cells the RL-time CGL loss would score: the phase-selected slot only,
    # discretized exactly the way the agent does it.
    target_cells = np.asarray(
        jax.device_get(
            agent._mask_to_attention_shape(
                observations["front_camera_mask"], jnp.asarray(attention)
            )
        )
    )

    size = (args.scale, args.scale)
    panels = []
    for index, (phase, demo_index) in enumerate(chosen):
        rgb = np.asarray(demos[demo_index]["observations"]["front_camera"])[-1]
        heat = _attention_heatmap_overlay(rgb, attention[index], 0.45, mode="softmax")
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        flat = attention[index].reshape(-1)
        probabilities = np.exp(flat - flat.max())
        probabilities /= probabilities.sum()
        cells = target_cells[index].reshape(-1)
        inside = float((probabilities * cells).sum())

        # Same map, dynamic range compressed, so the weaker object shows up.
        compressed = probabilities.reshape(attention[index].shape)
        compressed = (compressed / compressed.max()) ** args.gamma
        boosted = cv2.applyColorMap(
            cv2.resize((compressed * 255).astype(np.uint8), size,
                       interpolation=cv2.INTER_LINEAR),
            cv2.COLORMAP_JET,
        )[:, :, ::-1]
        boosted = cv2.addWeighted(
            cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR), 0.55,
            boosted, 0.45, 0,
        )

        # Outline the scored target cells so coverage is judgeable by eye.
        grid = cv2.resize(
            (target_cells[index] * 255).astype(np.uint8), size,
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heat = cv2.resize(heat, size, interpolation=cv2.INTER_LINEAR)
        cv2.drawContours(heat, contours, -1, (255, 255, 255), 1)
        cv2.drawContours(boosted, contours, -1, (255, 255, 255), 1)

        base = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        pair = np.hstack([
            label(base, f"{phase}  demo idx {demo_index}"),
            label(heat, f"as the actor shows it   inside(obj)={inside:.2f}"),
            label(boosted, f"same map, gamma {args.gamma:g}"),
        ])
        panels.append(pair)
        cv2.imwrite(str(out_dir / f"{phase}_{index:03d}.png"),
                    cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

    columns = 1
    rows = (len(panels) + columns - 1) // columns
    height, width = panels[0].shape[:2]
    sheet = np.zeros((rows * height, columns * width, 3), np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = panel
    path = out_dir / "contact_sheet.png"
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(panels)} panels and {path}")
    print("white outline = the cells the RL CGL loss scores (phase-selected "
          "object only; the hand is grounded but not scored at RL time)")


if __name__ == "__main__":
    main()
