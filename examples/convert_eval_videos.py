#!/usr/bin/env python3
"""Re-encode recorded eval videos into a format the local desktop can play.

Codec availability is a property of the machine, not of the file. On the box
these recordings come from, `gstreamer1.0-libav` is not installed, so Totem has
no H.264 or MPEG-4 decoder at all -- neither the original mp4v files nor H.264
re-encodes would open, and the file manager could not even thumbnail them. What
that system does have is `vp8dec`/`vp9dec` from gstreamer1.0-plugins-good, so
VP9-in-WebM plays with no extra packages.

Default output is therefore WebM alongside the untouched mp4. Use
`--format h264` on a machine that has the H.264 decoder instead.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system ffmpeg is
needed.

    python examples/convert_eval_videos.py --dir <run>/eval_recordings
    python examples/convert_eval_videos.py --dir <...> --format h264 --replace
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def frame_count(path):
    """Decoded frame count, or None if the file cannot be read."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = capture.read()
    capture.release()
    return count if ok and count > 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override playback rate. Recordings are written at the 10 Hz "
             "control rate; a lower value plays them back in slow motion, "
             "which is usually what you want for a 13-second episode.",
    )
    parser.add_argument("--crf", type=int, default=20,
                        help="Quality for h264. WebM uses --webm_crf.")
    parser.add_argument("--webm_crf", type=int, default=32)
    parser.add_argument(
        "--format",
        choices=("webm", "h264", "both"),
        default="webm",
        help="webm (default): VP9 in a WebM container, playable with only the "
             "stock gstreamer plugins. h264: mp4, needs gstreamer1.0-libav.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite the originals instead of writing *_h264.mp4. The "
             "original is only removed after the re-encode is confirmed to "
             "decode and to hold the same number of frames.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also re-encode files that already end in _h264 (normally "
             "skipped so repeated runs are cheap).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import imageio_ffmpeg
    except ImportError:
        raise SystemExit(
            "imageio-ffmpeg is not installed and no system ffmpeg was found; "
            "pip install imageio-ffmpeg"
        )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    sources = sorted(
        path for path in args.dir.glob(args.pattern)
        if args.all or not path.stem.endswith("_h264")
    )
    if not sources:
        raise SystemExit(f"no videos matching {args.pattern} under {args.dir}")

    formats = ("webm", "h264") if args.format == "both" else (args.format,)
    for source in sources:
      for fmt in formats:
        if fmt == "webm":
            target = source.with_suffix(".webm")
            codec_args = [
                "-c:v", "libvpx-vp9", "-crf", str(args.webm_crf), "-b:v", "0",
                "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                "-pix_fmt", "yuv420p",
            ]
        else:
            target = source.with_name(f"{source.stem}_h264.mp4")
            codec_args = [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            ]
        if target == source:
            continue
        command = [ffmpeg, "-y", "-loglevel", "error"]
        if args.fps is not None:
            # Before -i: reinterpret the existing frames at a new rate rather
            # than resampling them, so no frame is dropped or duplicated.
            command += ["-r", str(args.fps)]
        # Even dimensions are required by yuv420p.
        command += ["-i", str(source),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] + codec_args + [str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED {source.name} -> {fmt}: {result.stderr.strip()[:200]}",
                  file=sys.stderr)
            continue
        size = target.stat().st_size / 1e6
        if not args.replace or fmt == "webm":
            print(f"  {source.name}  ->  {target.name} ({size:.1f} MB)")
            continue

        # Never drop the original on an unverified re-encode: confirm the new
        # file decodes and carries every frame first.
        before, after = frame_count(source), frame_count(target)
        if after is None or (before is not None and after != before):
            print(f"  SKIPPED {source.name}: re-encode has {after} frames, "
                  f"original has {before}; keeping both", file=sys.stderr)
            continue
        target.replace(source)
        print(f"  {source.name}  ->  H.264 in place ({after} frames, {size:.1f} MB)")


if __name__ == "__main__":
    main()
