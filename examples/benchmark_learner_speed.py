#!/usr/bin/env python3
"""Measure learner throughput for the different visual encoder pipelines.

This builds the same agent that train_rlpd.py builds, but feeds it synthetic
observations so it runs without the robot, the cameras, or the SAM/gaze
checkpoints. It then times ``agent.update`` exactly the way the learner loop
calls it (``cta_ratio - 1`` critic-only updates plus one full update per
learner step) and reports the wall-clock cost of a learner step.

Examples
--------
    # ResNet RGB + small tactile CNN (current config) vs ResNet tactile
    python examples/benchmark_learner_speed.py \
        --variants resnet_tactile_cnn,resnet_tactile_resnet

    # compare the ResNet baseline against the grounded ViT
    python examples/benchmark_learner_speed.py \
        --variants resnet_tactile_cnn,vit_grounded
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np


# Observation layout of the tennis_ball_pick_and_place task. Kept literal so
# this script does not have to import the robot env (which needs ROS).
IMAGE_KEYS = (
    "front_camera",
    "tactile_data",
    "front_camera_mask",
    "front_camera_mask1",
    "front_camera_mask2",
)
IMAGE_SHAPES = {
    "front_camera": (128, 128, 3),
    "tactile_data": (128, 256, 3),
    "front_camera_mask": (128, 128, 3),
    "front_camera_mask1": (128, 128, 3),
    "front_camera_mask2": (128, 128, 3),
}
# tcp_pos(3) + tcp_ori(4) + gripper_pose(1) + pick/place/none one-hot(3)
STATE_DIM = 10  # 8 proprio + 2 phase one-hot (see PHASE_ONEHOT_DIM)
# (x, y, z, grip) once ArmActionSubspaceWrapper drops the rpy slots the robot
# never obeyed. Pass --action_dim 7 to time the old layout.
ACTION_DIM = 4

VARIANTS = {
    # name: (encoder_type, tactile_encoder_type, freeze_encoder)
    "resnet_tactile_cnn": ("resnet-pretrained", "cnn", False),
    "resnet_tactile_resnet": ("resnet-pretrained", "resnet", False),
    "vit_grounded": ("vit-grounded", "cnn", False),
    "vit_grounded_frozen": ("vit-grounded", "cnn", True),
    "vit": ("vit", "cnn", False),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="resnet_tactile_cnn,resnet_tactile_resnet",
        help=f"Comma separated subset of {sorted(VARIANTS)}.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--observation_horizon", type=int, default=1)
    parser.add_argument("--cta_ratio", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--measure_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target_steps",
        type=int,
        default=75000,
        help="Learner step count to extrapolate a wall-clock estimate for.",
    )
    parser.add_argument("--encoder_checkpoint_path", default=None)
    parser.add_argument("--action_dim", type=int, default=ACTION_DIM)
    parser.add_argument("--vit_image_size", type=int, default=None)
    parser.add_argument("--vit_hidden_dim", type=int, default=None)
    parser.add_argument("--vit_num_layers", type=int, default=None)
    parser.add_argument("--vit_num_heads", type=int, default=None)
    return parser.parse_args()


def make_sample_observation(horizon: int) -> Dict[str, np.ndarray]:
    """One unbatched observation, matching env.observation_space.sample()."""
    observation = {
        key: np.zeros((horizon, *shape), dtype=np.uint8)
        for key, shape in IMAGE_SHAPES.items()
    }
    observation["state"] = np.zeros((horizon, STATE_DIM), dtype=np.float32)
    return observation


def make_batch(batch_size: int, horizon: int, rng: np.random.Generator, action_dim: int = ACTION_DIM):
    """A replay batch with the keys the hybrid SAC update reads."""
    import jax.numpy as jnp
    from flax.core import freeze

    def images(prefix_shape):
        return {
            key: jnp.asarray(
                rng.integers(0, 256, size=(*prefix_shape, *shape), dtype=np.uint8)
            )
            for key, shape in IMAGE_SHAPES.items()
        }

    def observations():
        prefix = (batch_size, horizon)
        obs = images(prefix)
        state = rng.standard_normal((batch_size, horizon, STATE_DIM)).astype(
            np.float32
        )
        # Half the batch in pick phase, half in place phase, so the phase
        # gating branches see realistic inputs.
        state[:, :, -2:] = 0.0
        state[: batch_size // 2, :, -3] = 1.0
        state[batch_size // 2 :, :, -2] = 1.0
        obs["state"] = jnp.asarray(state)
        return obs

    return freeze(
        {
            "observations": observations(),
            "next_observations": observations(),
            "actions": jnp.asarray(
                rng.uniform(-1.0, 1.0, (batch_size, action_dim)).astype(np.float32)
            ),
            "rewards": jnp.zeros((batch_size,), dtype=jnp.float32),
            "masks": jnp.ones((batch_size,), dtype=jnp.float32),
            "grasp_penalty": jnp.zeros((batch_size,), dtype=jnp.float32),
            "robot_arm_penalty": jnp.zeros((batch_size,), dtype=jnp.float32),
        }
    )


def build_agent(variant: str, args, sample_observation):
    from serl_launcher.utils.launcher import (
        make_gaze_sac_pixel_agent_hybrid_single_arm,
    )

    encoder_type, tactile_encoder_type, freeze_encoder = VARIANTS[variant]
    kwargs = dict(
        seed=args.seed,
        sample_obs=sample_observation,
        sample_action=np.zeros((args.action_dim,), dtype=np.float32),
        image_keys=list(IMAGE_KEYS),
        encoder_type=encoder_type,
        tactile_encoder_type=tactile_encoder_type,
        freeze_encoder=freeze_encoder,
        discount=0.97,
        mask_pick_place_phase_control=True,
        mask_feature_gate_alpha=1.0,
        mask_feature_min_gate=0.1,
    )
    if args.encoder_checkpoint_path:
        kwargs["encoder_checkpoint_path"] = args.encoder_checkpoint_path
    if encoder_type == "vit-grounded":
        if args.vit_image_size:
            kwargs["vit_image_size"] = (args.vit_image_size, args.vit_image_size)
        for name in ("vit_hidden_dim", "vit_num_layers", "vit_num_heads"):
            if getattr(args, name):
                kwargs[name] = getattr(args, name)
    return make_gaze_sac_pixel_agent_hybrid_single_arm(**kwargs)


def time_updates(agent, batch, networks_to_update, steps: int) -> float:
    """Average seconds per agent.update, excluding queued async dispatch."""
    import jax

    start = time.perf_counter()
    for step in range(steps):
        agent, _info = agent.update(
            batch,
            networks_to_update=networks_to_update,
            train_step=step,
        )
    agent = jax.block_until_ready(agent)
    return (time.perf_counter() - start) / steps, agent


def benchmark_variant(variant: str, args) -> Dict[str, float]:
    import jax
    from serl_launcher.agents.continuous.sac import SACAgent

    rng = np.random.default_rng(args.seed)
    sample_observation = make_sample_observation(args.observation_horizon)
    agent = build_agent(variant, args, sample_observation)
    batch = make_batch(args.batch_size, args.observation_horizon, rng, args.action_dim)

    if isinstance(agent, SACAgent):
        critic_networks = frozenset({"critic"})
        full_networks = frozenset({"critic", "actor", "temperature"})
    else:
        critic_networks = frozenset({"critic", "grasp_critic"})
        full_networks = frozenset({"critic", "grasp_critic", "actor", "temperature"})
        if bool(agent.config.get("use_visual_aux", False)):
            critic_networks = critic_networks | frozenset({"visual_aux"})
            full_networks = full_networks | frozenset({"visual_aux"})

    # Warm up separately per update signature: each one is a distinct jit trace.
    compile_start = time.perf_counter()
    _, agent = time_updates(agent, batch, critic_networks, 1)
    _, agent = time_updates(agent, batch, full_networks, 1)
    compile_seconds = time.perf_counter() - compile_start
    if args.warmup_steps > 1:
        _, agent = time_updates(agent, batch, critic_networks, args.warmup_steps - 1)
        _, agent = time_updates(agent, batch, full_networks, args.warmup_steps - 1)

    # Two passes each, keeping the second: the first measured block still
    # absorbs lazy autotuning and GPU clock ramp-up, which otherwise makes
    # whichever set is timed first look slower than a superset of it.
    for _ in range(2):
        critic_seconds, agent = time_updates(
            agent, batch, critic_networks, args.measure_steps
        )
        full_seconds, agent = time_updates(
            agent, batch, full_networks, args.measure_steps
        )

    # Actor-side latency: one policy forward on a single observation.
    single_observation = jax.tree_util.tree_map(
        lambda value: value[0], batch["observations"]
    )
    key = jax.random.PRNGKey(args.seed)
    jax.block_until_ready(agent.sample_actions(single_observation, seed=key))
    action_start = time.perf_counter()
    for _ in range(20):
        action = agent.sample_actions(single_observation, seed=key)
    jax.block_until_ready(action)
    action_seconds = (time.perf_counter() - action_start) / 20

    step_seconds = (args.cta_ratio - 1) * critic_seconds + full_seconds
    return {
        "compile_seconds": compile_seconds,
        "critic_update_ms": critic_seconds * 1e3,
        "full_update_ms": full_seconds * 1e3,
        "learner_step_ms": step_seconds * 1e3,
        "learner_steps_per_second": 1.0 / step_seconds,
        "hours_for_target": args.target_steps * step_seconds / 3600.0,
        "sample_action_ms": action_seconds * 1e3,
    }


def main():
    args = parse_args()
    variants: List[str] = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")

    import jax

    print(f"jax devices: {jax.devices()}")
    print(
        f"batch_size={args.batch_size} horizon={args.observation_horizon} "
        f"cta_ratio={args.cta_ratio} measure_steps={args.measure_steps}"
    )

    results = {}
    for variant in variants:
        print(f"\n=== {variant} ({VARIANTS[variant][0]}, tactile={VARIANTS[variant][1]}, "
              f"frozen={VARIANTS[variant][2]}) ===", flush=True)
        results[variant] = benchmark_variant(variant, args)
        for key, value in results[variant].items():
            print(f"  {key}: {value:.3f}")

    print("\n" + "=" * 78)
    header = (
        f"{'variant':<24}{'critic ms':>11}{'full ms':>10}"
        f"{'step ms':>10}{'steps/s':>10}{'h@' + str(args.target_steps):>12}"
    )
    print(header)
    print("-" * 78)
    for variant in variants:
        row = results[variant]
        print(
            f"{variant:<24}{row['critic_update_ms']:>11.1f}"
            f"{row['full_update_ms']:>10.1f}{row['learner_step_ms']:>10.1f}"
            f"{row['learner_steps_per_second']:>10.2f}"
            f"{row['hours_for_target']:>12.2f}"
        )
    if len(variants) > 1:
        baseline = results[variants[0]]["learner_step_ms"]
        print("\nrelative to " + variants[0] + ":")
        for variant in variants[1:]:
            ratio = results[variant]["learner_step_ms"] / baseline
            print(f"  {variant}: {ratio:.2f}x  ({(ratio - 1.0) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
