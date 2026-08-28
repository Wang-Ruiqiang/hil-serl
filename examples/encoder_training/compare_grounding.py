#!/usr/bin/env python3
"""Compare grounding checkpoints on the failures that actually matter.

eval_encoder.py reports an average over every supervised frame, which hides the
two things worth knowing before a checkpoint is trusted:

leakage near the phase boundary
    Attention on the *other* phase's object. Averaged over a whole episode this
    looks small, because it is confined to the last few dozen pick frames --
    exactly where a premature move to the basket is most expensive.

occlusion frames
    The reported real-robot failure: the hand covers the ball, a 2D camera
    cannot tell that from a completed grasp, and an unconditioned query slides
    onto the basket while nothing has been picked up. These frames are selected
    here by hand/ball mask overlap and scored separately.

Both are compared against a per-phase mean-attention map. That baseline matters
because the basket barely moves between episodes (centroid std ~1 px vs ~11 px
for the ball), so a model that ignores the image entirely and emits one fixed
map per phase already scores well on place. Beating the baseline on *pick* is
the only evidence that the image is being used.

    python examples/encoder_training/compare_grounding.py \
        --run_dir runs/a --run_dir runs/b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples/encoder_training"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", default="best.msgpack")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument(
        "--boundary_frames",
        type=int,
        default=40,
        help="Pick frames this close to the phase switch form the "
             "'near boundary' slice, where leakage concentrates.",
    )
    parser.add_argument(
        "--occlusion_iou",
        type=float,
        default=0.30,
        help="Fraction of the ball mask covered by the hand mask for a frame "
             "to count as occluded.",
    )
    return parser.parse_args()


def load(run_dir, checkpoint, split):
    """Run one checkpoint over a split, returning per-frame arrays."""
    import jax
    import jax.numpy as jnp
    from flax import serialization

    from train_encoder import (
        TaskDemoFrameDataset,
        ViTPretrainModel,
        _frame_id,
        find_episode_demos,
        find_phase_demos,
        split_demos,
    )

    config = json.loads((run_dir / "config.json").read_text())
    phase_dim = int(config.get("grounding_phase_dim", 0))
    target_names = list(config["target_names"])
    data_root = Path(config["data_root"])
    scan = Path(config["phase_scan"]) if config.get("phase_scan") else None
    demos = (
        find_phase_demos(data_root, 1, scan)
        if scan and scan.is_file()
        else find_episode_demos(data_root, 1)
    )
    splits = dict(
        zip(
            ("train", "val", "test"),
            split_demos(demos, int(config["val_demos"]),
                        int(config["test_demos"]), int(config["seed"])),
        )
    )

    params = serialization.msgpack_restore(
        (run_dir / checkpoint).read_bytes()
    )["full_params"]
    model = ViTPretrainModel(
        num_targets=len(target_names),
        image_size=(int(config["image_size"]), int(config["image_size"])),
        patch_size=(int(config["patch_size"]), int(config["patch_size"])),
        hidden_dim=int(config["vit_hidden_dim"]),
        num_layers=int(config["vit_num_layers"]),
        num_heads=int(config["vit_num_heads"]),
        output_dim=int(config["output_dim"]),
        num_spatial_blocks=int(config["num_spatial_blocks"]),
        grounding_phase_dim=phase_dim,
    )

    def run(images, phase):
        return model.apply(
            {"params": params}, jnp.asarray(images),
            phase=jnp.asarray(phase, jnp.float32) if phase_dim else None,
            train=False,
        )["grounding_logits"]

    out = dict(phase_dim=phase_dim, name=run_dir.name,
               maps=[], swapped=[], cells=[], rel=[], phase=[])
    for demo in splits[split]:
        dataset = TaskDemoFrameDataset(
            [demo], image_size=int(config["input_size"]),
            mask_files=list(config["mask_files"]),
            target_names=target_names, sample_stride=1,
        )
        items = [dataset[i] for i in range(len(dataset))]
        images = np.stack(
            [s["image"].numpy().transpose(1, 2, 0) for s in items]
        ).astype(np.float32)
        targets = np.stack([s["target"].numpy() for s in items])
        is_pick = np.array(
            [float(s["phase"] in ("pick", "all")) for s in items], np.float32
        )
        onehot = np.stack([is_pick, 1.0 - is_pick], axis=-1)

        logits = np.asarray(jax.device_get(run(images, onehot)), np.float64)
        n, gh, gw = logits.shape
        flat = logits.reshape(n, -1)
        maps = np.exp(flat - flat.max(1, keepdims=True))
        maps /= maps.sum(1, keepdims=True)

        swapped = None
        if phase_dim:
            other = np.asarray(
                jax.device_get(run(images, 1.0 - onehot)), np.float64
            ).reshape(n, -1)
            swapped = np.exp(other - other.max(1, keepdims=True))
            swapped /= swapped.sum(1, keepdims=True)

        cells = np.asarray(jax.device_get(jax.image.resize(
            jnp.asarray(targets), (n, 3, gh, gw), method="linear")))
        cells = (cells > 0.04).astype(np.float64).reshape(n, 3, -1)

        out["maps"].append(maps)
        out["swapped"].append(swapped if swapped is not None else np.zeros_like(maps))
        out["cells"].append(cells)
        out["rel"].append(
            np.array([_frame_id(Path(s["frame"])) for s in items])
            - demo.first_place_frame
        )
        out["phase"].append(is_pick)
        # Pixel-level overlap, computed once (identical across checkpoints).
        ball_px = targets[:, 0] > 0.5
        hand_px = targets[:, 1] > 0.5
        overlap = (ball_px & hand_px).sum((1, 2)) / np.maximum(ball_px.sum((1, 2)), 1)
        out.setdefault("occlusion", []).append(overlap)
        out.setdefault("grid", []).append((gh, gw))

    for key in ("maps", "swapped", "cells", "rel", "phase", "occlusion"):
        out[key] = np.concatenate(out[key])
    return out


def report(result, args):
    maps, cells = result["maps"], result["cells"]
    ball, hand, basket = cells[:, 0], cells[:, 1], cells[:, 2]
    pick = result["phase"] > 0.5
    place = ~pick
    rel, occ = result["rel"], result["occlusion"]

    def mass(m, region):
        return (m * region).sum(1)

    target_cells = np.where(pick[:, None], ball, basket)
    inside = mass(maps, target_cells)
    other_cells = np.where(pick[:, None], basket, ball)
    leak = mass(maps, other_cells)

    # Per-phase mean attention map: what a model that ignores the image but
    # knows the phase would produce. Fit on this same split, so it is an
    # optimistic baseline -- the model has to beat it clearly to matter.
    baseline = {}
    for name, sel in (("pick", pick), ("place", place)):
        if sel.any():
            fixed = maps[sel].mean(0)
            baseline[name] = float(
                (fixed[None] * target_cells[sel]).sum(1).mean()
            )

    near = pick & (rel >= -args.boundary_frames)
    far = pick & (rel < -args.boundary_frames)
    occluded = pick & (occ >= args.occlusion_iou)

    print(f"\n===== {result['name']} "
          f"({'phase-conditioned' if result['phase_dim'] else 'unconditioned'}) =====")
    print(f"  inside(target)     pick {inside[pick].mean():.3f}   "
          f"place {inside[place].mean():.3f}")
    print(f"  fixed-map baseline pick {baseline.get('pick', float('nan')):.3f}   "
          f"place {baseline.get('place', float('nan')):.3f}"
          f"   -> pick gain {inside[pick].mean() / max(baseline.get('pick', 1e-9), 1e-9):.2f}x")
    print(f"  LEAK to other obj  pick->basket {leak[pick].mean():.3f}   "
          f"place->ball {leak[place].mean():.3f}")
    print(f"    far from switch (rel < -{args.boundary_frames}, n={int(far.sum())})"
          f"  : {leak[far].mean():.3f}")
    print(f"    near switch (rel >= -{args.boundary_frames}, n={int(near.sum())})"
          f"    : {leak[near].mean():.3f}   <- where premature moves start")
    if occluded.any():
        print(f"    ball >= {args.occlusion_iou:.0%} occluded by hand "
              f"(n={int(occluded.sum())}): leak {leak[occluded].mean():.3f}  "
              f"inside(ball) {inside[occluded].mean():.3f}   <- reported failure")
    else:
        print(f"    no pick frame has the ball >= {args.occlusion_iou:.0%} occluded")

    if result["phase_dim"]:
        swapped = result["swapped"]
        keep = other_cells.sum(1) > 0
        print(f"  counterfactual (same pixels, phase flipped):")
        print(f"    inside(other object) {mass(swapped, other_cells)[keep].mean():.3f}"
              f"   map L1 {0.5 * np.abs(maps - swapped).sum(1).mean():.3f}"
              f"   (1.0 = disjoint)")


def main():
    args = parse_args()
    for run_dir in args.run_dir:
        report(load(run_dir, args.checkpoint, args.split), args)
    print("\ninside = softmax mass on the phase's object; leak = mass on the "
          "other phase's object.\nBoth are computed on the token grid the CGL "
          "loss scores, not on pixels.")


if __name__ == "__main__":
    main()
