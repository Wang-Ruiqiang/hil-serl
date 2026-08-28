#!/usr/bin/env python3
"""Check what action range the SpaceMouse actually reaches, raw and after gain.

The RL action box is [-1, 1], but a full physical deflection on this device
reads well short of that, so human demonstrations never populate the outer part
of the box the policy can reach through its tanh -- leaving the critic to
extrapolate exactly where the actor likes to operate.

Push every axis to its physical stop and read the "peak" column. With the right
gain those peaks should sit at 1.00 without spending much time clipped.

    python examples/spacemouse_test.py --gain 1.6

Axis order and signs match SpaceMouseExpert, so the printed values are the
action components that would be handed to the environment.
"""

import argparse
import time

import numpy as np
import pyspacemouse

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Scale applied before clipping to [-1, 1]. Match "
             "SPACEMOUSE_GAIN in the task config.",
    )
    parser.add_argument("--period", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pyspacemouse.open()
    if device is None:
        raise SystemExit("failed to open SpaceMouse")

    peak_raw = np.zeros(6)
    peak_gained = np.zeros(6)
    clipped_samples = 0
    total_samples = 0

    print(f"gain = {args.gain:.2f}   move every axis to its stop. Ctrl+C to stop.\n")
    try:
        while True:
            state = device.read()
            if state is None:
                time.sleep(0.01)
                continue

            # Same mapping SpaceMouseExpert applies.
            raw = np.array(
                [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
                dtype=np.float64,
            )
            gained = np.clip(raw * args.gain, -1.0, 1.0)

            peak_raw = np.maximum(peak_raw, np.abs(raw))
            peak_gained = np.maximum(peak_gained, np.abs(gained))
            total_samples += 1
            if np.any(np.abs(raw * args.gain) > 1.0):
                clipped_samples += 1

            now = "  ".join(f"{a}={v:+.2f}" for a, v in zip(AXES, gained))
            peaks = "  ".join(
                f"{a}:{r:.2f}->{g:.2f}" for a, r, g in zip(AXES, peak_raw, peak_gained)
            )
            print(
                f"\r{now}   |   peak raw->gained  {peaks}   |   "
                f"clipped {100.0 * clipped_samples / max(total_samples, 1):.1f}%",
                end="",
                flush=True,
            )
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\n")
        print(f"{'axis':<7}{'peak raw':>10}{'peak gained':>14}{'suggested gain':>17}")
        print("-" * 48)
        for axis, raw_peak in zip(AXES, peak_raw):
            suggested = 1.0 / raw_peak if raw_peak > 1e-6 else float("nan")
            print(f"{axis:<7}{raw_peak:>10.3f}{min(raw_peak * args.gain, 1.0):>14.3f}"
                  f"{suggested:>17.2f}")
        translation_peak = peak_raw[:3].max()
        if translation_peak > 1e-6:
            print(
                f"\nTranslation axes peak at {translation_peak:.3f} raw; "
                f"gain {1.0 / translation_peak:.2f} would make a full deflection "
                "reach 1.00."
            )
        print(f"clipped {100.0 * clipped_samples / max(total_samples, 1):.1f}% "
              "of samples at the current gain")


if __name__ == "__main__":
    main()
