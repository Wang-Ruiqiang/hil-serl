"""Per-episode recording for eval runs: video plus the state needed to diagnose.

The live actor window shows one frame at a time and keeps nothing, so a failure
has to be reproduced to be studied. This writes, per episode:

    raw_<n>.mp4/.webm   the 128x128 `front_camera` observation the policy sees
                        (not the full-res camera frame -- if the policy is
                        failing on what it is fed, that is what has to be seen)
    attention_<n>.*     raw | attention | attention^gamma, with the selected
                        mask slot outlined and a HUD of the step's numbers

Each video is written twice, H.264/mp4 and VP9/WebM, because whether a file
opens depends on the codecs installed on the viewing machine rather than on the
file itself. See VIDEO_FORMATS.
    episode_<n>.npz     every per-step array, for offline analysis
    summary.json        one record per episode

The `.npz` carries the quantities that discriminate between the failure modes
this task has actually hit:

    pretanh_mean/std    a saturated mean (|mean| > ~1.5, i.e. |a| > 0.9) is the
                        signature of a policy operating outside the demo action
                        support, which reads as "entropy collapsed" but is not
Critic values are NOT recorded here. Each critic forward is an unjitted pass
through the encoder (~120 ms measured), and doing them per control step
overruns a 10 Hz budget by 3.6x, which would change the behaviour the eval is
supposed to measure. Every observation is stored instead, so analyze_eval.py
recomputes Q, grasp Q and mask leakage afterwards in batch, off-robot.
    phase / mask_slot   whether the classifier flipped early -- the hard signal
                        that can pull the arm to the basket before a grasp
    tactile_*           contact, which no camera angle can show. "reached the
                        ball and closed but came away empty" and "never closed"
                        look identical in video and are different bugs.
    wall_time           the control loop's actual period. The commanded step is
                        bounded by ACTION_SCALE, so a measured displacement
                        above that bound means either a late observation or a
                        long step -- indistinguishable without the timestamps.
    attention_inside    attention mass on the phase's own mask, per step, so a
                        drift onto the other object is visible as a time series
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HUD_HEIGHT = 62
PANEL = 256


VIDEO_FORMATS = (
    # (suffix, ffmpeg output args). Both are written for every video because
    # which one opens is a property of the viewing machine, not of the file:
    # the box these recordings come from has no H.264 decoder at all (no
    # gstreamer1.0-libav), so its mp4s cannot even be thumbnailed, while VP9 in
    # WebM plays from the stock gstreamer1.0-plugins-good. A machine set up the
    # other way round reads the mp4. Together they cost ~1 s per episode.
    (".mp4", dict(codec="libx264",
                  output_params=["-crf", "20", "-preset", "veryfast",
                                 "-movflags", "+faststart"])),
    (".webm", dict(codec="libvpx-vp9",
                   output_params=["-crf", "32", "-b:v", "0", "-row-mt", "1",
                                  "-deadline", "good", "-cpu-used", "4"])),
)


def _open_writers(base, width, height, fps):
    """Fan one RGB frame stream out to every format in VIDEO_FORMATS.

    Returns (send, close, paths). Falls back to a single OpenCV mp4v file if
    imageio-ffmpeg is missing -- OpenCV cannot encode H.264 or VP9 here (its
    only H.264 encoder is `h264_v4l2m2m`, which needs a V4L2 device), so that
    path produces a file that needs convert_eval_videos.py before it can be
    watched.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        import cv2 as _cv2

        path = base.with_suffix(".mp4")
        writer = _cv2.VideoWriter(
            str(path), _cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print("[eval recorder] imageio-ffmpeg not installed; writing mp4v. Run "
              "examples/convert_eval_videos.py to make these playable.")
        return (
            lambda frame: writer.write(_cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)),
            writer.release,
            [path],
        )

    streams, paths = [], []
    for suffix, options in VIDEO_FORMATS:
        path = base.with_suffix(suffix)
        stream = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=fps,
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            # The attention composite is 768x318, not a multiple of 16; yuv420p
            # only requires even dimensions, which both panels already satisfy.
            macro_block_size=1, quality=None, **options,
        )
        stream.send(None)
        streams.append(stream)
        paths.append(path)

    def send(frame):
        payload = np.ascontiguousarray(frame).tobytes()
        for stream in streams:
            stream.send(payload)

    def close():
        for stream in streams:
            stream.close()

    return send, close, paths


def _to_uint8_rgb(frame) -> np.ndarray:
    frame = np.asarray(frame)
    # Observations arrive stacked as (T, H, W, C); the current frame is last.
    while frame.ndim > 3:
        frame = frame[-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame[..., :3]


def _softmax_map(logits: np.ndarray) -> np.ndarray:
    flat = np.asarray(logits, dtype=np.float64).reshape(-1)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.exp(flat - flat.max())
    probabilities /= max(probabilities.sum(), 1e-12)
    return probabilities.reshape(np.asarray(logits).shape[-2:])


def _heat(rgb, weights, gamma=1.0):
    """Max-normalised JET overlay, optionally gamma-compressed for display.

    Brightness is relative to the frame's own peak, so a widely-spread object
    renders dark next to a tightly-peaked one even when it holds comparable
    mass (measured: the hand holds 0.23 during pick and still renders at 5% of
    peak, because the ball concentrates 0.67 into ~4 tokens). gamma < 1 lifts
    the low end so the whole support is visible; it changes nothing but pixels.
    """
    scaled = weights / max(float(weights.max()), 1e-12)
    if gamma != 1.0:
        scaled = scaled ** float(gamma)
    heat = cv2.applyColorMap(
        cv2.resize((scaled * 255).astype(np.uint8), (PANEL, PANEL),
                   interpolation=cv2.INTER_LINEAR),
        cv2.COLORMAP_JET,
    )[:, :, ::-1]
    base = cv2.resize(rgb, (PANEL, PANEL), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)


def _label(panel, text):
    out = panel.copy()
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


class EvalEpisodeRecorder:
    """Collect per-step arrays for one episode, then write video + npz."""

    def __init__(self, output_dir, gamma=0.35, fps=10, image_key="front_camera",
                 mask_key="front_camera_mask", tactile_key="tactile_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gamma = float(gamma)
        self.fps = int(fps)
        self.image_key = image_key
        self.mask_key = mask_key
        self.tactile_key = tactile_key
        self.summary_path = self.output_dir / "summary.json"
        self.session = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._reset()
        print(f"[eval recorder] writing to {self.output_dir}")

    def _reset(self):
        self.frames, self.masks, self.attention, self.tactile = [], [], [], []
        self.observations = []
        self.records = []

    def step(self, obs, action, extras, reward=None, done=None, info=None):
        """Record one control step. `extras` holds whatever diagnostics ran."""
        # Keep every observation key verbatim. This is what lets the critics be
        # queried afterwards instead of inside the control loop, and it is the
        # difference between an eval that can be re-analysed and one that has
        # to be re-run on the robot.
        self.observations.append(
            {key: np.asarray(value) for key, value in obs.items()}
        )
        self.frames.append(_to_uint8_rgb(obs[self.image_key]))

        mask = obs.get(self.mask_key)
        if mask is not None:
            mask = _to_uint8_rgb(mask)
            self.masks.append((mask[..., 0] > 127).astype(np.uint8))
        else:
            self.masks.append(None)

        attention = extras.get("attention")
        self.attention.append(
            None if attention is None else np.asarray(attention, np.float32)
        )

        # The tactile image is two pads side by side. Keep a small thumbnail
        # (cheap to store, enough to see the contact pattern) plus per-pad
        # scalars, rather than the full 128x256 stream.
        tactile = obs.get(self.tactile_key)
        if tactile is None:
            self.tactile.append(None)
        else:
            tactile = _to_uint8_rgb(tactile)
            self.tactile.append(
                cv2.resize(tactile, (64, 32), interpolation=cv2.INTER_AREA)
            )
            half = tactile.shape[1] // 2
            gray = tactile.mean(axis=-1)
            record_tactile = np.array(
                [gray[:, :half].mean(), gray[:, :half].std(),
                 gray[:, half:].mean(), gray[:, half:].std()], np.float32
            )

        state = np.asarray(obs.get("state"))
        while state.ndim > 1:
            state = state[-1]
        record = {
            "action": np.asarray(action, np.float32).reshape(-1),
            "state": state.astype(np.float32),
            "reward": float(reward) if reward is not None else np.nan,
            "done": bool(done) if done is not None else False,
            "intervened": bool(info and "intervene_action" in info),
            "wall_time": time.time(),
        }
        # During an intervention `action` is what the policy proposed and
        # `executed_action` is what the robot did. Keeping both is what lets a
        # recording answer "was the policy about to do the right thing" on the
        # steps a human took over.
        executed = (info or {}).get("intervene_action")
        record["executed_action"] = (
            np.asarray(executed, np.float32).reshape(-1)
            if executed is not None
            else record["action"]
        )
        if self.tactile[-1] is not None:
            record["tactile_stats"] = record_tactile
        for key in ("pretanh_mean", "pretanh_std", "q", "grasp_q"):
            value = extras.get(key)
            if value is not None:
                record[key] = np.asarray(value, np.float32).reshape(-1)
        self.records.append(record)

    # ------------------------------------------------------------------ write

    def _inside_mass(self, index):
        """Attention mass on the selected mask, on the token grid."""
        attention, mask = self.attention[index], self.masks[index]
        if attention is None or mask is None:
            return np.nan
        weights = _softmax_map(attention)
        grid = cv2.resize(mask.astype(np.float32), weights.shape[::-1],
                          interpolation=cv2.INTER_LINEAR)
        return float((weights * (grid > 0.04)).sum())

    def _hud(self, index, width):
        record = self.records[index]
        state = record["state"]
        phase = state[-2:]
        action = record["action"]
        strip = np.zeros((HUD_HEIGHT, width, 3), np.uint8)

        def put(text, row, column):
            cv2.putText(strip, text, (6 + column, 15 + row * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1,
                        cv2.LINE_AA)

        label = "PICK " if phase[0] > 0.5 else ("PLACE" if phase[-1] > 0.5 else "NONE ")
        put(f"t={index:3d}  {label}  inside={self._inside_mass(index):.2f}", 0, 0)
        put("a=[" + " ".join(f"{v:+.2f}" for v in action) + "]", 1, 0)
        mean = record.get("pretanh_mean")
        if mean is not None:
            saturated = np.abs(np.tanh(mean)) > 0.9
            put(f"|pretanh|max={np.abs(mean).max():.2f}"
                f" sat={int(saturated.sum())}/{len(mean)}", 1, width // 2)
        q = record.get("q")
        if q is not None:
            put(f"Q={q.mean():+.3f}", 0, width // 2)
        grasp = record.get("grasp_q")
        if grasp is not None and len(grasp) >= 3:
            # taken / open / closed -- "prefers" says which extreme the critic
            # rates higher, independent of what the actor emitted.
            prefers = "close" if grasp[2] > grasp[1] else "open"
            put(f"graspQ take{grasp[0]:+.2f} open{grasp[1]:+.2f} "
                f"close{grasp[2]:+.2f} -> {prefers}", 2, width // 2)
        elif grasp is not None:
            put(f"graspQ={grasp[0]:+.3f}", 2, width // 2)
        tactile = record.get("tactile_stats")
        if tactile is not None:
            # std, not mean: a pad's brightness drifts, but the spatial spread
            # is what jumps on contact.
            put(f"tactile sd {tactile[1]:5.1f} / {tactile[3]:5.1f}", 2, 0)
        if record["intervened"]:
            put("HUMAN INTERVENTION", 3, 0)
        return strip

    def save(self, episode, extra=None):
        if not self.frames:
            return None
        tag = f"{self.session}_ep{episode:03d}"

        write_raw, close_raw, raw_paths = _open_writers(
            self.output_dir / f"raw_{tag}", PANEL, PANEL, self.fps)
        for frame in self.frames:
            write_raw(cv2.resize(frame, (PANEL, PANEL),
                                 interpolation=cv2.INTER_NEAREST))
        close_raw()

        width = PANEL * 3
        write_attention, close_attention, attention_paths = _open_writers(
            self.output_dir / f"attention_{tag}", width, PANEL + HUD_HEIGHT,
            self.fps)
        for index, frame in enumerate(self.frames):
            attention = self.attention[index]
            if attention is None:
                panels = [cv2.resize(frame, (PANEL, PANEL))] * 3
            else:
                weights = _softmax_map(attention)
                panels = [
                    _label(cv2.resize(frame, (PANEL, PANEL),
                                      interpolation=cv2.INTER_LINEAR), "obs"),
                    _label(_heat(frame, weights),
                           f"attn  peak p={weights.max():.3f}"),
                    _label(_heat(frame, weights, self.gamma),
                           f"attn  gamma {self.gamma:g}"),
                ]
                mask = self.masks[index]
                if mask is not None:
                    grid = cv2.resize(mask * 255, (PANEL, PANEL),
                                      interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    for panel in panels[1:]:
                        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
            composite = np.vstack([np.hstack(panels), self._hud(index, width)])
            write_attention(composite)
        close_attention()

        arrays = {"frames": np.stack(self.frames)}
        if all(m is not None for m in self.masks):
            arrays["masks"] = np.stack(self.masks)
        if all(t is not None for t in self.tactile):
            arrays["tactile_thumbnails"] = np.stack(self.tactile)
        if all(a is not None for a in self.attention):
            arrays["attention_logits"] = np.stack(self.attention)
            # Only when a mask exists. vit-gaze carries no mask observation at
            # all, so every entry would be NaN -- which is not "the metric is
            # zero", it is "the metric is undefined", and averaging it wrote a
            # NaN into summary.json behind a RuntimeWarning.
            if any(m is not None for m in self.masks):
                arrays["attention_inside"] = np.array(
                    [self._inside_mass(i) for i in range(len(self.frames))],
                    np.float32,
                )
        for key in self.records[0]:
            if key in ("done", "intervened", "reward", "wall_time"):
                continue
            if all(key in r for r in self.records):
                arrays[key] = np.stack([r[key] for r in self.records])
        for key in self.observations[0]:
            if key in (self.image_key,):
                continue  # already stored as `frames`
            arrays[f"obs_{key}"] = np.stack(
                [observation[key] for observation in self.observations]
            )
        arrays["wall_time"] = np.array(
            [r["wall_time"] for r in self.records], np.float64
        )
        arrays["reward"] = np.array([r["reward"] for r in self.records], np.float32)
        arrays["done"] = np.array([r["done"] for r in self.records], bool)
        arrays["intervened"] = np.array([r["intervened"] for r in self.records], bool)
        npz_path = self.output_dir / f"episode_{tag}.npz"
        np.savez_compressed(npz_path, **arrays)

        record = {
            "episode": int(episode),
            "npz_mb": round(npz_path.stat().st_size / 1e6, 1),
            "session": self.session,
            "steps": len(self.frames),
            "return": float(np.nansum(arrays["reward"])),
            "intervened_steps": int(arrays["intervened"].sum()),
            "raw_video": [path.name for path in raw_paths],
            "attention_video": [path.name for path in attention_paths],
            "npz": npz_path.name,
        }
        if "attention_inside" in arrays and np.isfinite(
            arrays["attention_inside"]
        ).any():
            record["attention_inside_mean"] = float(
                np.nanmean(arrays["attention_inside"])
            )
        if extra:
            record.update(extra)
        history = []
        if self.summary_path.exists():
            history = json.loads(self.summary_path.read_text())
        history.append(record)
        self.summary_path.write_text(json.dumps(history, indent=2))

        print(f"[eval recorder] episode {episode}: {len(self.frames)} steps -> "
              + ", ".join(p.name for p in raw_paths + attention_paths)
              + f", {npz_path.name}")
        self._reset()
        return record
