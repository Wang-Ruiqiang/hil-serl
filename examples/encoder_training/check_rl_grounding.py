#!/usr/bin/env python3
"""Preflight: does a pretrained encoder still ground once it is inside the RL agent?

eval_encoder.py scores the checkpoint through the *pretraining* model. This
scores it through the *RL* code path -- the real agent, the real
`ViTGroundedEncodingWrapper`, the real `_mask_grounding_loss`, fed real demo
transitions straight out of the replay pickle.

The two can disagree, and when they do the checkpoint loads without complaint
and then quietly destroys the run. The failure this was written for: the
pretraining dataset handed the ViT images in [0, 1] while the RL wrapper hands
it raw 0..255 pixels, so the trunk was fit to a 255x smaller input range.
eval_encoder reported grounding_inside 0.857; through the RL path it was 0.112.

Run this after every pretraining, before starting a learner.

    python examples/encoder_training/check_rl_grounding.py \
        --checkpoint examples/encoder_training/runs/<run>/best.msgpack \
        --demo_path examples/demo_data/<demos>.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--demo_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min_inside",
        type=float,
        default=0.5,
        help="Fail below this grounding inside-mass. Pretraining reaches ~0.85; "
        "anything near the ~0.02 chance level means the checkpoint is not "
        "reaching the trunk, or is being fed a different input distribution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    from flax.core import freeze

    from benchmark_learner_speed import (
        ACTION_DIM,
        IMAGE_KEYS,
        make_sample_observation,
    )
    from serl_launcher.utils.launcher import (
        make_gaze_sac_pixel_agent_hybrid_single_arm,
    )

    with open(args.demo_path, "rb") as handle:
        demos = pickle.load(handle)
    print(f"demo transitions: {len(demos)}")

    agent = make_gaze_sac_pixel_agent_hybrid_single_arm(
        seed=args.seed,
        sample_obs=make_sample_observation(1),
        sample_action=np.zeros((ACTION_DIM,), np.float32),
        image_keys=list(IMAGE_KEYS),
        encoder_type="vit-grounded",
        tactile_encoder_type="cnn",
        discount=0.97,
        mask_pick_place_phase_control=True,
        encoder_checkpoint_path=str(args.checkpoint),
    )
    if not agent.config.get("use_visual_aux", False):
        raise SystemExit("agent has no visual-aux loss; nothing to check")

    state_dim = make_sample_observation(1)["state"].shape[-1]

    def _match_state_dim(state):
        # Demos recorded before the phase one-hot was narrowed to two slots
        # carry an extra trailing "no target" column. It was always the last
        # one, so truncating is the exact conversion, not an approximation --
        # the same slice train_rlpd's prepare_replay_transition applies.
        if state.shape[-1] > state_dim:
            return state[..., :state_dim]
        return state

    def _match_action_dim(actions):
        if actions.shape[-1] == ACTION_DIM:
            return actions
        if actions.shape[-1] == 7 and ACTION_DIM == 4:
            return actions[..., jnp.asarray([0, 1, 2, 6])]
        raise ValueError(
            f"demo actions have width {actions.shape[-1]}, agent expects {ACTION_DIM}"
        )

    rng = np.random.default_rng(args.seed)
    networks = frozenset(
        {"critic", "grasp_critic", "actor", "temperature", "visual_aux"}
    )

    def batch(size):
        index = rng.choice(len(demos), size, replace=False)

        def stack(field, key=None):
            source = (
                [demos[i][field][key] for i in index]
                if key
                else [demos[i][field] for i in index]
            )
            stacked = jnp.asarray(np.stack([np.asarray(v) for v in source]))
            return _match_state_dim(stacked) if key == "state" else stacked

        return freeze(
            {
                "observations": {
                    k: stack("observations", k) for k in demos[0]["observations"]
                },
                "next_observations": {
                    k: stack("next_observations", k)
                    for k in demos[0]["next_observations"]
                },
                # Demos predate ArmActionSubspaceWrapper and still carry the
                # rpy slots; keep (x, y, z, grip) like prepare_replay_transition.
                "actions": _match_action_dim(stack("actions").reshape(size, -1)),
                "rewards": stack("rewards").reshape(size).astype(jnp.float32),
                "masks": stack("masks").reshape(size).astype(jnp.float32),
                "grasp_penalty": stack("grasp_penalty")
                .reshape(size)
                .astype(jnp.float32),
                "robot_arm_penalty": stack("robot_arm_penalty")
                .reshape(size)
                .astype(jnp.float32),
            }
        )

    inside, ratio, chance = [], [], []
    for step in range(args.batches):
        current = batch(args.batch_size)
        _, info = agent.update(
            current, networks_to_update=networks, train_step=step
        )
        aux = dict(info)["visual_aux"]
        # mask_mass and outside_mass sum to 1 by construction (softmax over the
        # token grid, binary mask), so "coverage" and "1 - outside" are the same
        # number -- reporting both would just look like corroboration. The
        # informative companion is the chance level: what this would score if
        # the attention were uniform, i.e. the ball's share of the token grid.
        inside.append(float(aux["mask_grounding_coverage"]))
        ratio.append(float(aux["mask_grounding_to_td_ratio"]))

        _, attention = agent.state.apply_fn(
            {"params": agent.state.params},
            current["observations"],
            name="actor",
            train=False,
            return_attention=True,
        )
        # Use the very key the agent's CGL loss reads, so the chance baseline
        # describes the same target the inside-mass was scored against.
        grounding_key = agent.config.get("mask_grounding_key") or "front_camera_mask1"
        cells = np.asarray(
            jax.device_get(
                agent._mask_to_attention_shape(
                    current["observations"][grounding_key], attention
                )
            )
        )
        cells = cells.reshape(cells.shape[0], -1)
        occupied = cells.sum(axis=1)
        if (occupied > 0).any():
            chance.append(
                float(np.mean(occupied[occupied > 0] / cells.shape[1]))
            )

    inside_mean = float(np.mean(inside))
    chance_mean = float(np.mean(chance)) if chance else float("nan")
    print(f"\n=== {args.checkpoint} through the RL agent ===")
    print(f"  grounding inside-mass  : {inside_mean:.3f}  "
          f"(min {np.min(inside):.3f} max {np.max(inside):.3f})")
    print(f"  uniform-attention level: {chance_mean:.3f}  "
          f"-> {inside_mean / max(chance_mean, 1e-9):.0f}x chance")
    print(f"  CGL / TD loss ratio    : {np.mean(ratio):.2f}")

    if inside_mean < args.min_inside:
        print(
            f"\nFAIL: inside-mass {inside_mean:.3f} < {args.min_inside}. "
            "The pretrained weights are not grounding through the RL path. "
            "Check that the pretraining fed the encoder the same pixel range "
            "the RL wrapper does (raw 0..255), and that the checkpoint's ViT "
            "shape matches the agent's."
        )
        raise SystemExit(1)
    print("\nPASS: the checkpoint grounds through the RL code path.")


if __name__ == "__main__":
    main()
