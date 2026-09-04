#!/usr/bin/env python3
"""Rebuild a demo file's gaze-derived observations the way the wrapper builds them.

GazeDerivedObservationWrapper writes four things into every live observation:
front_camera_mask1, front_camera_mask2, the selected front_camera_mask, and the
two state columns the grounding query reads. It only ever sees the live env, so
a demo file keeps whatever those were when it was recorded.

tennis_ball_pick_and_place_new30_demos.pkl was recorded without a working mask
predictor: all three mask images are empty in all 5822 transitions, and the
selected index never moved, so the state columns carry a constant [1.0, 0.0].
RLPD draws half of every batch from the demo buffer, so half of every update fed
the mask CNN a blank image and told a gaze-conditioned encoder the operator was
staring at the top-right corner. The 2026-08-16 demos the successful run used
are intact by comparison: 0.0% / 2.8% / 0.0% empty.

This re-runs both predictors over the stored frames and writes what the wrapper
would have written, hysteresis included, so the demo half of a batch matches the
online half.

    python examples/relabel_demo_gaze.py \
        --demo_path examples/demo_data/<name>.pkl \
        --mask_predictor_checkpoint_path .../mask_predictor_ckpt_0825/best.pt \
        --gaze_predictor_checkpoint_path .../gaze_heatmap_ckpt_0825

The input file is never modified; the output defaults to <name>_gazexy.pkl.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serl_launcher.utils.gaze_mask_utils import (  # noqa: E402
    _predict_mask_probs,
    add_gaze_mask_image_to_obs,
    compute_all_index_target_mask_fields,
    compute_gaze_target_mask_fields,
    compute_index_target_mask_fields,
    load_mask_predictor,
    select_gaze_target_mask,
)
from serl_launcher.utils.gaze_utils import (  # noqa: E402
    infer_heatmap_shape,
    load_gaze_predictor,
)

PHASE_COLS = 2
SOURCE_KEY = "front_camera"
SELECTED_KEY = "front_camera_mask"
SLOT_KEYS = {"mask1": "front_camera_mask1", "mask2": "front_camera_mask2"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo_path", required=True)
    parser.add_argument("--gaze_predictor_checkpoint_path", required=True)
    parser.add_argument("--mask_predictor_checkpoint_path", required=True)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--dilation_px", type=int, default=2,
        help="Must match --gaze_target_mask_dilation on the run.",
    )
    parser.add_argument(
        "--gaze_source", choices=["predictor", "state"], default="predictor",
        help="Where the fixation comes from. 'predictor' runs the gaze network "
             "and overwrites the state's last two columns. 'state' keeps the "
             "columns already there -- use it on a file exported with "
             "--state_gaze_slot=gaze_xy, so the eye tracker's own fixation "
             "survives while the masks are rebuilt from the deployed mask "
             "predictor. That pairing matters: the recordings' SAM3 ball masks "
             "are empty on 33% of frames against 1.4% for the predictor, so "
             "reading masks off disk leaves the demo half of every batch with "
             "a ball that keeps vanishing while the online half has one.",
    )
    parser.add_argument(
        "--hysteresis", type=int, default=3,
        help="Must match gaze_selection_hysteresis on the wrapper.",
    )
    return parser.parse_args()


class Hysteresis:
    """The wrapper's rule: a new selection has to repeat before it takes over."""

    def __init__(self, n):
        self.n = max(1, int(n))
        self.reset()

    def reset(self):
        self.committed = None
        self.pending = None
        self.run = 0

    def __call__(self, raw):
        if raw is None:
            return self.committed
        if self.committed is None:
            self.committed = self.pending = raw
            self.run = 1
            return self.committed
        if raw == self.pending:
            self.run += 1
        else:
            self.pending, self.run = raw, 1
        if raw != self.committed and self.run >= self.n:
            self.committed = raw
        return self.committed


def main():
    args = parse_args()
    demo_path = Path(args.demo_path)
    out_path = (
        Path(args.output_path)
        if args.output_path
        else demo_path.with_name(demo_path.stem + "_gazexy.pkl")
    )
    if out_path.resolve() == demo_path.resolve():
        raise ValueError("refusing to overwrite the input demo file")

    with open(demo_path, "rb") as handle:
        transitions = pickle.load(handle)
    print(f"loaded {len(transitions)} transitions from {demo_path}")

    sample = transitions[0]["observations"]
    image_keys = list(sample.keys())
    gaze_predictor = (
        load_gaze_predictor(
            sample, image_keys, args.gaze_predictor_checkpoint_path,
            preferred_key=SOURCE_KEY,
        )
        if args.gaze_source == "predictor"
        else None
    )
    mask_predictor = load_mask_predictor(
        sample, image_keys, args.mask_predictor_checkpoint_path,
        preferred_key=SOURCE_KEY,
    )
    if mask_predictor is None:
        raise RuntimeError("the mask predictor must load")
    if args.gaze_source == "predictor" and gaze_predictor is None:
        raise RuntimeError("the gaze predictor must load")
    heatmap_shape = infer_heatmap_shape(sample, image_keys, preferred_key=SOURCE_KEY)

    for obs_key in ("observations", "next_observations"):
        hyst = Hysteresis(args.hysteresis)
        held = 0
        for i, transition in enumerate(transitions):
            obs = dict(transition[obs_key])
            if args.gaze_source == "state":
                kept = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[
                    -PHASE_COLS:
                ]
                xy = None if float(kept[0]) < 0 else kept
                selected, info = select_gaze_target_mask(
                    _predict_mask_probs(obs, mask_predictor),
                    None,
                    target_shape=tuple(heatmap_shape),
                    dilation_px=args.dilation_px,
                    gaze_xy_norm=xy,
                    return_info=True,
                )
                fields = {
                    "gaze_target_mask": selected,
                    "selected_mask_index": info["selected_mask_index"],
                    "gaze_xy_norm": xy,
                }
            else:
                fields = compute_gaze_target_mask_fields(
                    obs, gaze_predictor, mask_predictor, heatmap_shape,
                    dilation_px=args.dilation_px,
                )
            raw = fields.get("selected_mask_index")
            committed = hyst(raw)
            if committed is not None and committed != raw:
                held += 1
                swap = compute_index_target_mask_fields(
                    obs, mask_predictor, heatmap_shape,
                    selected_mask_index=committed,
                )
                fields["gaze_target_mask"] = swap["gaze_target_mask"]
            obs = add_gaze_mask_image_to_obs(
                obs, gaze_target_mask=fields["gaze_target_mask"],
                image_key=SELECTED_KEY, reference_key=SOURCE_KEY,
            )
            slots = compute_all_index_target_mask_fields(
                obs, mask_predictor, heatmap_shape,
            )
            for slot, key in SLOT_KEYS.items():
                obs = add_gaze_mask_image_to_obs(
                    obs,
                    gaze_target_mask=slots.get(slot, {}).get(
                        "gaze_target_mask",
                        np.zeros(tuple(heatmap_shape), dtype=np.float32),
                    ),
                    image_key=key, reference_key=SOURCE_KEY,
                )
            # Same convention append_gaze_xy_to_state uses: no fixation is
            # (-1, -1), outside the image box, so the conditioner can tell "no
            # gaze" from a corner. The columns are overwritten in place because
            # the state is already the right width.
            if args.gaze_source == "state":
                transition[obs_key] = obs
                if obs_key == "observations" and bool(
                    np.asarray(transition["dones"]).ravel()[0]
                ):
                    hyst.reset()
                if (i + 1) % 1000 == 0:
                    print(f"  {obs_key}: {i + 1}/{len(transitions)}", flush=True)
                continue
            xy = fields.get("gaze_xy_norm")
            state = np.asarray(obs["state"], dtype=np.float32)
            value = (
                np.asarray([-1.0, -1.0], dtype=np.float32)
                if xy is None
                else np.asarray(xy, dtype=np.float32).reshape(-1)[:PHASE_COLS]
            )
            state[..., -PHASE_COLS:] = np.broadcast_to(
                value, (*state.shape[:-1], PHASE_COLS)
            )
            obs["state"] = state
            transition[obs_key] = obs
            # An episode boundary resets the wrapper, so it resets here too.
            if obs_key == "observations" and bool(
                np.asarray(transition["dones"]).ravel()[0]
            ):
                hyst.reset()
            if (i + 1) % 1000 == 0:
                print(f"  {obs_key}: {i + 1}/{len(transitions)}", flush=True)
        print(f"  {obs_key}: done, hysteresis held {held} frames")

    with open(out_path, "wb") as handle:
        pickle.dump(transitions, handle)

    states = np.stack(
        [np.asarray(t["observations"]["state"]).reshape(-1) for t in transitions]
    )
    tail = states[:, -PHASE_COLS:]
    print(f"wrote {out_path}")
    print(
        f"  gaze columns: {len(np.unique(tail.round(4), axis=0))} distinct, "
        f"x [{tail[:, 0].min():.3f}, {tail[:, 0].max():.3f}] "
        f"y [{tail[:, 1].min():.3f}, {tail[:, 1].max():.3f}]"
    )
    for key in (SELECTED_KEY, *SLOT_KEYS.values()):
        empty = np.mean([
            int(np.count_nonzero(np.asarray(t["observations"][key]))) == 0
            for t in transitions
        ])
        print(f"  {key}: {empty:.1%} empty")


if __name__ == "__main__":
    main()
