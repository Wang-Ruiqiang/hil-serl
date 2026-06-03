import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import List, Tuple

import cv2
import numpy as np


def _repo_pythonpath_env(extra_env: dict | None = None):
    """Return an environment where child processes can import franka_env."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    package_root = str(pathlib.Path(__file__).resolve().parents[2])
    pythonpath_parts = [package_root]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def sam2_prompt_child_main(mirror_dir: str, key_indices: list[int], mode: str, title: str, out_path: str):
    """Run OpenCV prompt UI in a child process and write prompts as JSON."""
    win = f"{title} - {mode.upper()}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    prompts = {}
    max_objects = 3
    obj_colors = {
        0: (0, 255, 255),
        1: (255, 0, 255),
        2: (0, 165, 255),
    }
    active_obj = 0
    tmp_box = []
    tmp_box_end = None
    dragging = False
    cur_i = 0

    def _draw_text_lines(image, lines, origin=(10, 28), line_height=26):
        x, y = origin
        pad = 6
        max_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            max_width = max(max_width, tw)
        box_h = line_height * len(lines) + pad
        cv2.rectangle(
            image,
            (x - pad, y - 20 - pad),
            (x + max_width + pad, y - 20 + box_h),
            (0, 0, 0),
            -1,
        )
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (x, y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

    def _on_mouse(event, x, y, flags, param):
        nonlocal dragging, tmp_box, tmp_box_end, active_obj
        idx = param["cur_idx"]
        if idx < 0:
            return
        d = prompts.setdefault(idx, {})
        if active_obj not in d:
            d[active_obj] = {"clicks": [], "labels": [], "boxes": []}
        entry = d[active_obj]
        if event == cv2.EVENT_LBUTTONDOWN:
            tmp_box = [(x, y)]
            tmp_box_end = (x, y)
            dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging and len(tmp_box) == 1:
            tmp_box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and len(tmp_box) == 1:
            tmp_box.append((x, y))
            entry["boxes"].append(list(tmp_box))
            tmp_box = []
            tmp_box_end = None
            dragging = False
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            entry["clicks"].append([x, y])
            entry["labels"].append(1)
        elif event == cv2.EVENT_MBUTTONDOWN:
            entry["clicks"].append([x, y])
            entry["labels"].append(0)

    cv2.setMouseCallback(win, _on_mouse, param={"cur_idx": -1})

    while key_indices:
        frame_idx = int(key_indices[cur_i])
        img = cv2.imread(os.path.join(mirror_dir, f"{frame_idx}.jpg"))
        if img is None:
            if cur_i < len(key_indices) - 1:
                cur_i += 1
                continue
            break

        vis = img.copy()
        for oid, rec in prompts.get(frame_idx, {}).items():
            color = obj_colors.get(int(oid), (255, 255, 255))
            for (px, py), lbl in zip(rec.get("clicks", []), rec.get("labels", [])):
                cv2.circle(vis, (px, py), 7, color, -1)
                if not lbl:
                    cv2.line(vis, (px - 6, py - 6), (px + 6, py + 6), (0, 0, 255), 2)
                    cv2.line(vis, (px - 6, py + 6), (px + 6, py - 6), (0, 0, 255), 2)
                cv2.putText(vis, str(oid), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for box in rec.get("boxes", []):
                if len(box) == 2:
                    cv2.rectangle(vis, box[0], box[1], color, 2)
                    cv2.putText(vis, f"obj{oid}", box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(tmp_box) == 1 and tmp_box_end is not None:
            cv2.rectangle(vis, tmp_box[0], tmp_box_end, obj_colors.get(active_obj, (255, 255, 255)), 1)

        cv2.setMouseCallback(win, _on_mouse, param={"cur_idx": frame_idx})
        _draw_text_lines(
            vis,
            [
                f"[{mode}] key {cur_i+1}/{len(key_indices)} frame={frame_idx} obj={active_obj}",
                "mouse: drag box | left double-click positive | middle-click negative",
                "obj: 0/1/2 switch | n next obj | u undo",
                "clear: c current obj | x whole frame",
                "nav: j prev | k next | Enter finish | q quit",
            ],
        )
        cv2.imshow(win, vis)
        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10) or key == ord("q"):
            break
        if key == ord("n"):
            active_obj = (active_obj + 1) % max_objects
        elif ord("0") <= key <= ord(str(max_objects - 1)):
            active_obj = key - ord("0")
        elif key == ord("u"):
            rec = prompts.get(frame_idx, {}).get(active_obj)
            if rec:
                if rec.get("boxes"):
                    rec["boxes"].pop()
                elif rec.get("clicks"):
                    rec["clicks"].pop()
                    if rec.get("labels"):
                        rec["labels"].pop()
        elif key == ord("c"):
            prompts.get(frame_idx, {}).pop(active_obj, None)
        elif key == ord("x"):
            prompts.pop(frame_idx, None)
        elif key == ord("j") and cur_i > 0:
            cur_i -= 1
            active_obj = 0
        elif key == ord("k") and cur_i < len(key_indices) - 1:
            cur_i += 1
            active_obj = 0

    try:
        cv2.destroyWindow(win)
    except Exception:
        pass
    with open(out_path, "w") as f:
        json.dump(prompts, f)


class SAM2GazeLabeler:
    """Interactive SAM2 labeling and gaze-contact construction."""

    def __init__(self, frame_root: str, et_mirror_dir: str, rs_mirror_dir: str):
        self.frame_root = frame_root
        self.et_mirror_dir = et_mirror_dir
        self.rs_mirror_dir = rs_mirror_dir

    def run_inline_v2(
        self,
        frame_ranges: List[Tuple[int, int]],
        prompt_frame_groups: List[List[Tuple[int, int]]] | None = None,
        y_prompts_et: int = 20,
        y_prompts_rs: int = 10,
        rs_select_uses_same_keyset: bool = False,
        random_seed: int = 0,
    ):
        selected_all = []
        for start, end in frame_ranges:
            start, end = int(start), int(end)
            if end < start:
                start, end = end, start
            selected_all.extend(range(start, end + 1))
        selected_all = sorted(set(selected_all))
        if not selected_all:
            print("[SAM2 v2] no frames in ranges; skip.")
            return

        print(f"[SAM2 v2] total frames in batch: {len(selected_all)}, ranges={frame_ranges}")
        prompt_frame_groups = prompt_frame_groups or [[rng] for rng in frame_ranges]
        key_et = self._select_keyframes_per_episode_groups(
            prompt_frame_groups,
            prompts_per_ep=y_prompts_et,
            include_ends=True,
        )
        et_prompts = self._prompt_via_subprocess(self.et_mirror_dir, key_et, mode="et") if key_et else {}
        self._propagate_and_save_with_prompts(
            "et",
            self.et_mirror_dir,
            selected_indices=selected_all,
            prompts_dict=et_prompts,
            save_union=True,
        )

        key_rs = key_et if rs_select_uses_same_keyset else self._select_keyframes_per_episode_groups(
            prompt_frame_groups,
            prompts_per_ep=y_prompts_rs,
            include_ends=True,
        )
        rs_prompts = self._prompt_via_subprocess(self.rs_mirror_dir, key_rs, mode="rs") if key_rs else {}
        self._propagate_and_save_with_prompts(
            "rs",
            self.rs_mirror_dir,
            selected_indices=selected_all,
            prompts_dict=rs_prompts,
            save_union=True,
        )

        self._build_gaze_contacts_subset(
            selected_indices=selected_all,
            et_to_rs_map={0: 0, 1: 1},
            flip_y=True,
            eye_wh=(1280, 720),
        )
        print(
            f"[SAM2 v2] done. episodes={len(prompt_frame_groups)} frames={len(selected_all)} "
            f"prompts_et={len(key_et)} prompts_rs={len(key_rs)}"
        )

    def _select_keyframes_per_episode_groups(
        self,
        frame_groups: List[List[Tuple[int, int]]],
        prompts_per_ep: int | None = 20,
        include_ends: bool = True,
    ) -> list[int]:
        selected = set()
        count = max(0, int(prompts_per_ep or 0))
        if count <= 0:
            return []

        for ranges in frame_groups:
            frames = []
            for start, end in ranges:
                start, end = int(start), int(end)
                if end < start:
                    start, end = end, start
                frames.extend(range(start, end + 1))
            frames = sorted(set(frames))
            if not frames:
                continue
            if len(frames) <= count:
                selected.update(frames)
                continue

            if include_ends:
                positions = np.rint(np.linspace(0, len(frames) - 1, num=count)).astype(int).tolist()
            else:
                stride = len(frames) / float(count + 1)
                positions = [int(round(stride * k)) for k in range(1, count + 1)]

            chosen = []
            for pos in positions:
                pos = max(0, min(len(frames) - 1, int(pos)))
                idx = frames[pos]
                if idx not in chosen:
                    chosen.append(idx)
            for idx in frames:
                if len(chosen) >= count:
                    break
                if idx not in chosen:
                    chosen.append(idx)
            selected.update(chosen[:count])
        return sorted(selected)

    def _select_keyframes_per_episode(
        self,
        frame_ranges: List[Tuple[int, int]],
        prompts_per_ep: int | None = 20,
        include_ends: bool = True,
    ) -> list[int]:
        selected = set()
        count = max(0, int(prompts_per_ep or 0))
        if count <= 0:
            return []
        for start, end in frame_ranges:
            start, end = int(start), int(end)
            if end < start:
                start, end = end, start
            length = end - start + 1
            if length <= count:
                selected.update(range(start, end + 1))
                continue

            if include_ends:
                candidates = np.rint(np.linspace(start, end, num=count)).astype(int).tolist()
            else:
                stride = length / float(count + 1)
                candidates = [int(round(start + stride * k)) for k in range(1, count + 1)]

            chosen = []
            for idx in candidates:
                idx = max(start, min(end, int(idx)))
                if idx not in chosen:
                    chosen.append(idx)
            for idx in range(start, end + 1):
                if len(chosen) >= count:
                    break
                if idx not in chosen:
                    chosen.append(idx)
            selected.update(chosen[:count])
        return sorted(selected)

    def _prompt_via_subprocess(self, mirror_dir: str, key_indices: list[int], mode: str, timeout_sec: int = 900):
        workdir = tempfile.TemporaryDirectory(prefix="sam2_ui_")
        args_path = os.path.join(workdir.name, "ui_args.json")
        out_path = os.path.join(workdir.name, "ui_out.json")
        with open(args_path, "w") as f:
            json.dump({"mirror_dir": mirror_dir, "keys": list(map(int, key_indices)), "mode": mode, "out": out_path}, f)

        pycode = (
            "import json, os; "
            "from franka_env.gaze.sam2_labeler import sam2_prompt_child_main; "
            "args=json.load(open(os.environ['SAM2_ARGS'],'r')); "
            "sam2_prompt_child_main(args['mirror_dir'], args['keys'], args['mode'], 'SAM2 Prompt', args['out'])"
        )
        env = _repo_pythonpath_env({"SAM2_ARGS": args_path})
        print(f"[SAM2][subprocess] launch child UI ({mode}) ...")
        proc = subprocess.Popen([sys.executable, "-c", pycode], env=env)
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            print("[SAM2][subprocess] UI timeout, terminating ...")
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        prompts = {}
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    prompts = json.load(f)
            except Exception:
                prompts = {}
        workdir.cleanup()
        print(f"[SAM2][subprocess] collected prompts for {mode}: frames={len(prompts)}")
        return prompts

    def _load_predictor(self, default_repo: str = "facebook/sam2-hiera-small"):
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        ckpt = os.environ.get("SAM2_CKPT", "").strip()
        model_id = ckpt if ckpt else default_repo
        print(f"[SAM2] loading from: {model_id}")
        return SAM2VideoPredictor.from_pretrained(
            model_id,
            config_overrides={"inference": {"multimask_output": False}},
        )

    def _propagate_and_save_with_prompts(
        self,
        mode: str,
        mirror_dir: str,
        selected_indices: list[int],
        prompts_dict: dict,
        save_union: bool = True,
    ):
        if not prompts_dict:
            print(f"[SAM2] no prompts for {mode}; skip propagation.")
            return

        subset_dir = tempfile.TemporaryDirectory(prefix=f"sam2_{mode}_frames_")
        original_to_video_idx = {}
        video_idx_to_original = {}
        for original_frame_idx in sorted(set(map(int, selected_indices))):
            src = pathlib.Path(mirror_dir) / f"{original_frame_idx}.jpg"
            if not src.exists():
                continue
            video_idx = len(original_to_video_idx)
            dst = pathlib.Path(subset_dir.name) / f"{video_idx}.jpg"
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            original_to_video_idx[original_frame_idx] = video_idx
            video_idx_to_original[video_idx] = original_frame_idx
        if not original_to_video_idx:
            subset_dir.cleanup()
            print(f"[SAM2] no selected frames found in {mirror_dir}; skip propagation.")
            return
        print(
            f"[SAM2] {mode} subset video: {len(original_to_video_idx)} frames "
            f"from {min(original_to_video_idx)}..{max(original_to_video_idx)}"
        )

        import torch
        from contextlib import nullcontext
        from torch.amp import autocast

        predictor = self._load_predictor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        amp_ctx = autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")) if device == "cuda" else nullcontext()
        with torch.inference_mode(), amp_ctx:
            state = predictor.init_state(
                subset_dir.name,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

            allowed_oids = set()
            normalized_prompts = {}
            for frame_idx, per_obj in prompts_dict.items():
                try:
                    original_frame_idx = int(frame_idx)
                except Exception:
                    continue
                if original_frame_idx not in original_to_video_idx:
                    print(f"[SAM2] skip prompt outside current subset: frame {original_frame_idx}")
                    continue
                frame_idx = original_to_video_idx[original_frame_idx]
                normalized_prompts[frame_idx] = per_obj
                for obj_id, data in (per_obj or {}).items():
                    if data.get("clicks") or data.get("boxes"):
                        allowed_oids.add(int(obj_id))
            if not allowed_oids:
                print(f"[SAM2] no valid objects for {mode}; skip propagation.")
                subset_dir.cleanup()
                return

            for frame_idx, per_obj in normalized_prompts.items():
                for obj_id, data in (per_obj or {}).items():
                    oid = int(obj_id)
                    clicks = data.get("clicks", []) or []
                    labels = data.get("labels", []) or []
                    if clicks:
                        predictor.add_new_points_or_box(
                            state,
                            frame_idx=frame_idx,
                            obj_id=oid,
                            points=np.asarray(clicks, dtype=np.float32),
                            labels=np.asarray(labels if len(labels) == len(clicks) else [1] * len(clicks), dtype=np.int64),
                            box=None,
                        )
                    for box in data.get("boxes", []) or []:
                        if isinstance(box, (list, tuple)) and len(box) == 2:
                            (x0, y0), (x1, y1) = box
                            predictor.add_new_points_or_box(
                                state,
                                frame_idx=frame_idx,
                                obj_id=oid,
                                points=None,
                                labels=None,
                                box=np.asarray([x0, y0, x1, y1], dtype=np.float32),
                            )

            keep = {
                original_to_video_idx[int(idx)]
                for idx in selected_indices
                if int(idx) in original_to_video_idx
            }
            prefix = "et_" if mode == "et" else "rs_"
            oid_remap = {old: new for new, old in enumerate(sorted(allowed_oids))}
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                frame_idx = int(frame_idx)
                if frame_idx not in keep:
                    continue
                original_frame_idx = video_idx_to_original[frame_idx]
                frame_dir = pathlib.Path(self.frame_root) / f"frame_{original_frame_idx}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                union = None
                for i, oid in enumerate(obj_ids):
                    oid = int(oid)
                    if oid not in allowed_oids:
                        continue
                    new_oid = oid_remap[oid]
                    mask = (masks[i].detach().cpu().numpy() > 0).astype(np.uint8).squeeze()
                    cv2.imwrite(str(frame_dir / f"{prefix}mask_obj{new_oid}.png"), mask * 255)
                    union = mask if union is None else (union | mask)
                if save_union and union is not None:
                    cv2.imwrite(str(frame_dir / f"{prefix}mask_union.png"), union * 255)
        subset_dir.cleanup()

    def _build_gaze_contacts_subset(
        self,
        selected_indices: list[int],
        et_to_rs_map: dict[int, int],
        flip_y: bool = True,
        eye_wh: tuple[int, int] = (1280, 720),
        gaze_hit_radius_px: int = 40,
    ):
        def _read_mask(frame_dir: pathlib.Path, oid: int):
            for prefix in ("et_", ""):
                path = frame_dir / f"{prefix}mask_obj{oid}.png"
                if path.exists():
                    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        return (mask > 0).astype(np.uint8)
            return None

        def _parse_gaze(frame_dir: pathlib.Path):
            path = frame_dir / "pupil_gaze.json"
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text())
            except Exception:
                return None
            payload = data.get("data", data)
            if "norm_pos" not in payload:
                return None
            width, height = eye_wh
            x_norm, y_norm = payload["norm_pos"]
            u = int(round(float(x_norm) * width))
            v = int(round((1.0 - float(y_norm)) * height)) if flip_y else int(round(float(y_norm) * height))
            ts = payload.get("timestamp", data.get("ts", None))
            return u, v, ts

        def _choose_hit(et_masks: dict[int, np.ndarray], u: int, v: int):
            if not et_masks:
                return None
            h, w = next(iter(et_masks.values())).shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                return None
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * gaze_hit_radius_px + 1, 2 * gaze_hit_radius_px + 1),
            )
            candidates = []
            for oid, mask in et_masks.items():
                if cv2.dilate(mask, kernel, iterations=1)[v, u] > 0:
                    dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
                    candidates.append((float(dist[v, u]), oid))
            if not candidates:
                return None
            return min(candidates)[1]

        for idx in selected_indices:
            frame_dir = pathlib.Path(self.frame_root) / f"frame_{idx}"
            if not frame_dir.exists():
                continue
            parsed = _parse_gaze(frame_dir)
            u = v = ts = None
            if parsed is not None:
                u, v, ts = parsed

            et_masks = {}
            for et_oid in et_to_rs_map.keys():
                mask = _read_mask(frame_dir, et_oid)
                if mask is not None:
                    et_masks[et_oid] = mask
            et_oid = _choose_hit(et_masks, u, v) if u is not None and v is not None else None
            rs_oid = et_to_rs_map.get(et_oid, None) if et_oid is not None else None
            label = {
                "timestamp": ts,
                "gaze_uv_in_eye": None if u is None else [int(u), int(v)],
                "et_object_id": None if et_oid is None else int(et_oid),
                "rs_object_id": None if rs_oid is None else int(rs_oid),
                "class_id": None if rs_oid is None else int(rs_oid),
                "class_name": None if rs_oid is None else f"obj{int(rs_oid)}",
                "hit": bool(et_oid is not None),
            }
            (frame_dir / "gaze_contact.json").write_text(json.dumps(label, indent=2, ensure_ascii=False))
