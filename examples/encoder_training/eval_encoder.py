#!/usr/bin/env python3
"""Evaluate a pretrained vit-grounded encoder on the held-out test demos.

The test split is the one train_encoder.py carved out with the same seed and
never trained or model-selected on, so these numbers are the honest estimate
of what the RL run starts from.

Reports:
  grounding_inside   attention mass landing on the ball cells (chance is
                     ball_cells / total_cells, printed alongside)
  grounding_hit      fraction of frames whose attention argmax is a ball cell
  center_px          RMSE of each object's predicted center, in pixels of the
                     128x128 observation
  presence_acc       accuracy of the object-present head
  inverse_rmse       RMSE of the predicted action
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from torch.utils.data import DataLoader

try:
    from .train_encoder import (
        TaskDemoFrameDataset,
        ViTPretrainModel,
        _batch_to_jax,
        _mask_geometry,
        find_episode_demos,
        find_phase_demos,
        split_demos,
    )
except ImportError:  # Allows running this file directly from its directory.
    from train_encoder import (
        TaskDemoFrameDataset,
        ViTPretrainModel,
        _batch_to_jax,
        _mask_geometry,
        find_episode_demos,
        find_phase_demos,
        split_demos,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.msgpack")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    target_names = list(config["target_names"])
    input_size = int(config["input_size"])

    data_root = config.get("data_root")
    if not data_root:
        from task_configs import get_task_config
        data_root = get_task_config(config["exp_name"]).data_root
    data_root = Path(data_root)
    phase_scan = Path(config["phase_scan"]) if config.get("phase_scan") else None
    stride = int(config["frame_stride"])
    if phase_scan and phase_scan.is_file():
        demos = find_phase_demos(data_root, stride, phase_scan)
    else:
        demos = find_episode_demos(data_root, stride)
    train_demos, val_demos, test_demos = split_demos(
        demos, int(config["val_demos"]), int(config["test_demos"]), int(config["seed"])
    )
    chosen = {"train": train_demos, "val": val_demos, "test": test_demos}[args.split]
    print(f"{args.split} split: {len(chosen)} demos")

    dataset = TaskDemoFrameDataset(
        chosen,
        image_size=input_size,
        mask_files=list(config["mask_files"]),
        target_names=target_names,
        sample_stride=stride,
    )
    loader = DataLoader(
        dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"{args.split} frames: {len(dataset)}")

    payload = serialization.msgpack_restore(
        (args.run_dir / args.checkpoint).read_bytes()
    )
    params = payload["full_params"]
    # A phase-conditioned checkpoint stores a (2, 1, D) query table; an
    # unconditioned one stores (1, 1, D). Building the wrong one fails to load
    # rather than loading something subtly wrong.
    phase_dim = int(config.get("grounding_phase_dim", 0))
    print(f"grounding query  : "
          + ("phase-conditioned (phase_dim=%d)" % phase_dim if phase_dim
             else "unconditioned (single constant query)"))
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

    @jax.jit
    def forward(batch, phase):
        return model.apply(
            {"params": params},
            batch["image"],
            future_image=batch["future_image"],
            state=batch["state"],
            phase=phase,
            train=False,
        )

    inside, hit, ground_n = [], [], 0
    per_phase = {"pick": ([], []), "place": ([], [])}
    hand_inside = []
    center_errors = [[] for _ in target_names]
    presence_correct, presence_n = 0, 0
    inverse_se, inverse_n = 0.0, 0
    chance = []

    # Leakage: attention that lands on the *other* phase's object. This is the
    # number the phase conditioning exists to fix -- with a 2D camera the
    # unconditioned query mistakes an occluding hand for a completed grasp and
    # puts mass on the basket while the robot is still picking.
    leak = {"basket_in_pick": [], "ball_in_place": []}
    # Counterfactual: same pixels, phase flipped. If the switch is genuinely
    # hard, forcing "place" on a pick frame moves the map onto the basket and
    # vice versa, independent of what the image looks like.
    swap = {"target_inside": [], "l1_distance": []}

    for raw in loader:
        batch = _batch_to_jax(raw)
        phase = batch["phase_onehot"] if phase_dim else None
        out = jax.device_get(forward(batch, phase))
        grounding = np.asarray(out["grounding_logits"], dtype=np.float64)
        n, gh, gw = grounding.shape

        # The phase's object is what the RL-time CGL loss can actually score
        # (no hand mask exists at RL time), so that is the headline number.
        # The hand is a pretrain-only target and is reported separately.
        pick = np.asarray(batch["pick_phase"]) > 0
        place = np.asarray(batch["place_phase"]) > 0
        target = jnp.asarray(batch["target"])
        phase_mask = jnp.where(
            jnp.asarray(pick)[:, None, None], target[:, 0], target[:, 2]
        )
        # 128 is not divisible by the 14x14 token grid, so resize the mask the
        # same way _grounding_kl_loss does instead of block-reshaping it.
        target_cells = jax.image.resize(
            phase_mask[:, None], (n, 1, gh, gw), method="linear"
        )[:, 0]
        target_cells = np.asarray(target_cells > 0.04, dtype=np.float64)
        hand_cells = np.asarray(
            jax.image.resize(target[:, 1][:, None], (n, 1, gh, gw), method="linear")[:, 0]
            > 0.04,
            dtype=np.float64,
        ).reshape(n, -1)

        flat = grounding.reshape(n, -1)
        attention = np.exp(flat - flat.max(axis=1, keepdims=True))
        attention /= attention.sum(axis=1, keepdims=True)
        cells = target_cells.reshape(n, -1)

        def _cells_of(channel):
            return np.asarray(
                jax.image.resize(
                    target[:, channel][:, None], (n, 1, gh, gw), method="linear"
                )[:, 0] > 0.04,
                dtype=np.float64,
            ).reshape(n, -1)

        ball_cells, basket_cells = _cells_of(0), _cells_of(2)
        if pick.any():
            leak["basket_in_pick"].append(
                (attention * basket_cells).sum(axis=1)[pick]
            )
        if place.any():
            leak["ball_in_place"].append((attention * ball_cells).sum(axis=1)[place])

        if phase_dim:
            swapped = jax.device_get(forward(batch, 1.0 - phase))
            other = np.asarray(swapped["grounding_logits"], dtype=np.float64)
            other = other.reshape(n, -1)
            other = np.exp(other - other.max(axis=1, keepdims=True))
            other /= other.sum(axis=1, keepdims=True)
            # Under the flipped phase the correct target is the other object.
            swap_cells = np.where(pick[:, None], basket_cells, ball_cells)
            keep = swap_cells.sum(axis=1) > 0
            if keep.any():
                swap["target_inside"].append((other * swap_cells).sum(axis=1)[keep])
            swap["l1_distance"].append(
                0.5 * np.abs(attention - other).sum(axis=1)
            )
        valid = (cells.sum(axis=1) > 0) & (pick | place)
        hand_valid = (hand_cells.sum(axis=1) > 0) & (pick | place)
        if hand_valid.any():
            hand_inside.append((attention * hand_cells).sum(axis=1)[hand_valid])
        if valid.any():
            sample_inside = (attention * cells).sum(axis=1)
            peak = attention.argmax(axis=1)
            sample_hit = cells[np.arange(n), peak]
            inside.append(sample_inside[valid])
            hit.append(sample_hit[valid])
            chance.append((cells.sum(axis=1) / cells.shape[1])[valid])
            ground_n += int(valid.sum())
            for name, sel in (("pick", pick & valid), ("place", place & valid)):
                if sel.any():
                    per_phase[name][0].append(sample_inside[sel])
                    per_phase[name][1].append(sample_hit[sel])

        geometry_target, present = _mask_geometry(batch["target"])
        geometry_target = np.asarray(geometry_target)
        present = np.asarray(present)
        predicted = np.asarray(out["geometry_predictions"])
        for index in range(len(target_names)):
            keep = present[:, index] > 0
            if keep.any():
                error = predicted[keep, index, :2] - geometry_target[keep, index, :2]
                # centers live in [-1, 1] across the observation
                center_errors[index].append(error * (input_size / 2.0))

        presence_correct += int(
            ((np.asarray(out["presence_logits"]) > 0) == (present > 0.5)).sum()
        )
        presence_n += present.size

        transition = np.asarray(batch["transition_valid"]) > 0
        if transition.any():
            error = np.asarray(out["inverse_action"])[transition] - np.asarray(
                batch["action"]
            )[transition]
            inverse_se += float((error**2).sum())
            inverse_n += int(error.size)

    print(f"\n=== {args.run_dir.name} / {args.checkpoint} / {args.split} split ===")
    if ground_n:
        inside_all = np.concatenate(inside)
        print(
            f"grounding_inside : {inside_all.mean():.3f}"
            f"   (chance {np.concatenate(chance).mean():.4f}, "
            f"{inside_all.mean() / max(np.concatenate(chance).mean(), 1e-9):.0f}x)"
        )
        print(f"grounding_hit    : {np.concatenate(hit).mean():.3f}   ({ground_n} supervised frames)")
        if hand_inside:
            merged = np.concatenate(hand_inside)
            print(f"  hand          : inside={merged.mean():.3f}  n={len(merged)}"
                  f"   (pretrain-only; RL has no hand mask)")
        if leak["basket_in_pick"]:
            print(f"  LEAK basket during pick : "
                  f"{np.concatenate(leak['basket_in_pick']).mean():.3f}"
                  f"   (the failure mode phase conditioning targets)")
        if leak["ball_in_place"]:
            print(f"  LEAK ball during place  : "
                  f"{np.concatenate(leak['ball_in_place']).mean():.3f}")
        if swap["l1_distance"]:
            print(f"  counterfactual (phase flipped, same pixels):")
            print(f"    inside(other object)  : "
                  f"{np.concatenate(swap['target_inside']).mean():.3f}"
                  f"   (high = the phase input, not the image, picks the target)")
            print(f"    map L1 distance       : "
                  f"{np.concatenate(swap['l1_distance']).mean():.3f}"
                  f"   (1.0 = disjoint maps, 0.0 = phase ignored)")
        for name, (ins, hts) in per_phase.items():
            if ins:
                target = "ball" if name == "pick" else "basket"
                print(
                    f"  {name:<5} ({target:<6}): inside={np.concatenate(ins).mean():.3f} "
                    f"hit={np.concatenate(hts).mean():.3f}  n={len(np.concatenate(ins))}"
                )
    for index, name in enumerate(target_names):
        if center_errors[index]:
            error = np.concatenate(center_errors[index])
            print(
                f"center_px[{name:<7}]: {np.sqrt((error**2).sum(axis=1).mean()):.2f} px "
                f"(of {input_size})"
            )
    print(f"presence_acc     : {presence_correct / max(presence_n, 1):.3f}")
    if inverse_n:
        print(f"inverse_rmse     : {np.sqrt(inverse_se / inverse_n):.4f}")


if __name__ == "__main__":
    main()
