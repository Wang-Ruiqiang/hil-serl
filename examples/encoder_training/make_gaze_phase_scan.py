#!/usr/bin/env python3
"""Build a phase scan from the operator's gaze instead of the pick classifier.

Writes the same JSON schema test_pick_classifier_phase.py writes, so
train_encoder.py consumes it through --phase_scan with no code change. That is
the point: swapping the phase source becomes a one-file substitution, and the
rest of the recipe -- mask supervision, hand_grounding_weight, the segmentation
and geometry heads -- stays byte-identical to the run that succeeded.

The transition is the first frame of the first sustained run of `--run_length`
gaze samples landing inside the basket mask. Sustained rather than first-hit
because a single saccade across the basket during the reach is not a phase
change; the same N=3 latch the gaze/mask pipeline already uses.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import cv2, numpy as np

ROOT = Path("/home/ealin/workspaces/DexTacHil/data/recorded_data/tennis_ball_pick_and_place")

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", dest="datasets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run_length", type=int, default=3)
    p.add_argument("--basket_mask", default="basket_mask.png")
    return p.parse_args()

def gaze_cell(frame: Path):
    path = frame / "gaze_contact.json"
    if not path.exists():
        return None
    try:
        g = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    uv = g.get("gaze_uv_in_realsense")
    if not g.get("hit", False) or uv is None:
        return None
    w, h = g.get("realsense_size", [640, 480])
    return (int(np.clip(uv[0] / w * 128, 0, 127)), int(np.clip(uv[1] / h * 128, 0, 127)))

def main():
    a = parse()
    results, skipped = [], []
    for ds in a.datasets:
        root = Path(ds) if os.path.isabs(ds) else ROOT / ds
        meta = json.loads((root / "recording_metadata.json").read_text())
        for ep in meta.get("episode_ranges", []):
            idx = int(ep["episode_index"])
            first, run = None, 0
            for i in range(int(ep["start_frame"]), int(ep["end_frame"]) + 1):
                frame = root / f"frame_{i}"
                cell = gaze_cell(frame)
                if cell is None:
                    run = 0
                    continue
                m = cv2.imread(str(frame / a.basket_mask), 0)
                inside = m is not None and cv2.resize(
                    m, (128, 128), interpolation=cv2.INTER_NEAREST)[cell[1], cell[0]] > 127
                run = run + 1 if inside else 0
                if run >= a.run_length and first is None:
                    first = i - (a.run_length - 1)
                    break
            if first is None:
                skipped.append((root.name, idx))
                continue
            results.append(dict(dataset=root.name, episode_index=idx,
                                start_frame=int(ep["start_frame"]),
                                end_frame=int(ep["end_frame"]),
                                first_positive_frame=int(first),
                                first_positive_offset=int(first) - int(ep["start_frame"]),
                                excluded=False, exclusion_reason=None,
                                source="gaze", run_length=a.run_length))
            print(f"[gaze] {root.name} ep{idx} 相位翻转 @ frame {first}")
    payload = dict(checkpoint="gaze:gaze_contact.json", threshold=float(a.run_length),
                   datasets=[str(ROOT / d) if not os.path.isabs(d) else d for d in a.datasets],
                   excluded_count=len(skipped), results=results)
    Path(a.output).write_text(json.dumps(payload, indent=2))
    print(f"\n[summary] 可用 {len(results)} 个 episode，判不出的 {len(skipped)} 个 {skipped}")
    print(f"[output] {a.output}")

if __name__ == "__main__":
    main()
