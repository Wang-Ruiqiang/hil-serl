from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

import cv2
import numpy as np

from serl_launcher.utils.gaze_utils import (
    compute_gaze_heatmap_fields,
    gaze_xy_norm_from_heatmap,
    latest_image,
    select_gaze_image_key,
)


@dataclass
class MaskPredictorBundle:
    model: object
    torch: object
    device: object
    image_key: str
    image_size: int
    mask_slots: tuple[str, ...]


def _build_tiny_unet(torch, base_channels: int, output_channels: int):
    nn = torch.nn

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(base_channels)
            self.enc1 = ConvBlock(3, channels)
            self.enc2 = ConvBlock(channels, channels * 2)
            self.enc3 = ConvBlock(channels * 2, channels * 4)
            self.enc4 = ConvBlock(channels * 4, channels * 8)
            self.pool = nn.MaxPool2d(2)

            self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
            self.dec3 = ConvBlock(channels * 8, channels * 4)
            self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
            self.dec2 = ConvBlock(channels * 4, channels * 2)
            self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
            self.dec1 = ConvBlock(channels * 2, channels)
            self.mask_head = nn.Conv2d(channels, output_channels, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            d3 = self.up3(e4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.mask_head(d1)

    return TinyUNet()


def load_mask_predictor(
    obs,
    image_keys: Iterable[str],
    checkpoint_path: str,
    *,
    preferred_key: str = "front_camera",
    device: str = "cuda",
    log_fn=print,
):
    """Load the frozen RGB->mask1/mask2 predictor used to build target-mask input."""
    if not checkpoint_path:
        return None

    image_key = select_gaze_image_key(obs, image_keys, preferred_key)
    if image_key is None:
        log_fn(
            "Could not load mask predictor: no RGB camera image was found in "
            f"image_keys={image_keys}."
        )
        return None

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Mask predictor requires PyTorch in the training/actor environment. "
            "If this fails in serl_env, either install a compatible torch there or "
            "run with --use_gaze_target_mask=False."
        ) from exc

    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Mask predictor is configured to run on CUDA, but torch.cuda.is_available() "
            "is False. Check the GPU environment before training."
        )

    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    mask_slots = tuple(checkpoint.get("mask_slots", ("mask1", "mask2")))
    image_size = int(config.get("image_size", 128))
    base_channels = int(config.get("base_channels", 32))

    model = _build_tiny_unet(
        torch,
        base_channels=base_channels,
        output_channels=len(mask_slots),
    )
    model.load_state_dict(checkpoint["model_state"])
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    log_fn(
        "Loading frozen mask predictor "
        f"checkpoint={checkpoint_path} image_key={image_key} "
        f"image_size={image_size} device={torch_device}"
    )
    return MaskPredictorBundle(
        model=model,
        torch=torch,
        device=torch_device,
        image_key=image_key,
        image_size=image_size,
        mask_slots=mask_slots,
    )


def _predict_mask_probs(obs, mask_predictor: MaskPredictorBundle):
    image = latest_image(obs, mask_predictor.image_key)
    if image is None:
        return None

    rgb = np.asarray(image)
    resized = cv2.resize(
        rgb,
        (mask_predictor.image_size, mask_predictor.image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    model_input = resized.astype(np.float32) / 255.0
    model_input = np.transpose(model_input, (2, 0, 1))[None]

    torch = mask_predictor.torch
    with torch.no_grad():
        tensor = torch.from_numpy(model_input).to(mask_predictor.device)
        logits = mask_predictor.model(tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)
    return probs


def _resize_2d(image, shape, method="nearest"):
    interpolation = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(
        np.asarray(image),
        (int(shape[1]), int(shape[0])),
        interpolation=interpolation,
    )


def _dilate_binary_mask(mask, radius: int):
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _dilate_prob_map(prob, radius: int):
    """Local max of a probability map over the same neighbourhood as the mask dilation.

    The containment test asks "is any mask pixel within `radius` of the gaze?",
    which on a binarised map is exactly a dilation. Scoring the candidates has
    to ask the same question of the probabilities, or the two disagree: the
    2026-09-02 run picked the basket on frames where the raw probabilities under
    the gaze pixel were 0.000 (ball) against 0.066 (basket) -- both far below
    the 0.5 threshold, so the winner was decided by noise, and the basket
    channel carries broader sub-threshold residue than a 7 px ball. A grayscale
    dilation makes the score the same evidence the candidacy used, which also
    floors it: a candidate exists only where this value clears the threshold.
    """
    if radius <= 0:
        return np.asarray(prob, dtype=np.float32)
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(np.asarray(prob, dtype=np.float32), kernel, iterations=1)


def _morph_binary_mask(mask, radius: int, op):
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.morphologyEx(mask.astype(np.uint8), op, kernel, iterations=1).astype(bool)


def _keep_largest_component(mask, min_area_px: int = 0):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    if int(areas[largest_label - 1]) < int(min_area_px):
        return np.zeros_like(mask, dtype=bool)
    return labels == largest_label


def _postprocess_slot_mask(mask, slot_name: str):
    """Clean predictable slot-specific mask artifacts without changing mask semantics."""
    mask = np.asarray(mask, dtype=bool)
    if slot_name != "mask2":
        return mask.astype(np.float32)

    # mask2 is the basket/place target. The predictor often produces tiny
    # detached fragments, so remove small islands while keeping the main target.
    mask = _morph_binary_mask(mask, radius=1, op=cv2.MORPH_OPEN)
    mask = _morph_binary_mask(mask, radius=2, op=cv2.MORPH_CLOSE)
    mask = _keep_largest_component(mask, min_area_px=64)
    return mask.astype(np.float32)


def _mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _mask_bbox_inside(inner_mask, outer_mask, margin_px: int = 2):
    inner_bbox = _mask_bbox(inner_mask)
    outer_bbox = _mask_bbox(outer_mask)
    if inner_bbox is None or outer_bbox is None:
        return False
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox
    margin_px = int(margin_px)
    return (
        ix0 >= ox0 - margin_px
        and iy0 >= oy0 - margin_px
        and ix1 <= ox1 + margin_px
        and iy1 <= oy1 + margin_px
    )


def mask_to_image(mask, reference_image=None, channels=3):
    """Convert a 2D float/bool mask to a uint8 image observation."""
    if reference_image is not None:
        reference_image = np.asarray(reference_image)
        if reference_image.ndim == 4:
            height, width = reference_image.shape[-3:-1]
            leading_shape = reference_image.shape[:-3]
        elif reference_image.ndim == 3:
            height, width = reference_image.shape[:2]
            leading_shape = ()
        else:
            raise ValueError(f"Unsupported reference image shape: {reference_image.shape}")
    else:
        mask_array = np.asarray(mask)
        while mask_array.ndim > 2:
            mask_array = mask_array[0]
        height, width = mask_array.shape
        leading_shape = ()

    mask_array = np.asarray(mask, dtype=np.float32)
    while mask_array.ndim > 2:
        mask_array = mask_array[0]
    if mask_array.shape != (height, width):
        mask_array = _resize_2d(mask_array, (height, width), method="nearest")
    mask_array = np.nan_to_num(mask_array, nan=0.0, posinf=0.0, neginf=0.0)
    mask_array = np.clip(mask_array, 0.0, 1.0)
    image = np.repeat((mask_array[..., None] * 255.0).astype(np.uint8), channels, axis=-1)
    if leading_shape:
        image = np.broadcast_to(image, (*leading_shape, height, width, channels)).copy()
    return image


# Width of the phase one-hot appended to the proprioceptive state. Every
# consumer slices `state[..., -PHASE_ONEHOT_DIM:]`, so this is the single place
# the width is defined.
#
# Two slots, not three: the old third slot meant "no target selected" and never
# fired -- all 8015 recorded demo transitions carry [1,0,0] or [0,1,0] -- so it
# was a dead input dimension. Dropping it changes the state width from 11 to 10,
# which means demos and replay buffers recorded before this are one column wider;
# train_rlpd's prepare_replay_transition truncates them with `state[..., :10]`,
# and because the dead slot was last, that truncation is exactly the right
# conversion rather than an approximation.
PHASE_ONEHOT_DIM = 2


def gaze_phase_onehot(selected_mask_index):
    """Encode selected gaze target as [mask1, mask2].

    Neither slot is set when no target is selected, which the downstream
    consumers read as "not picking" -- the same thing the old [0, 0, 1] meant.
    """
    phase = np.zeros((PHASE_ONEHOT_DIM,), dtype=np.float32)
    if selected_mask_index == 0:
        phase[0] = 1.0
    elif selected_mask_index == 1:
        phase[1] = 1.0
    return phase


def append_gaze_phase_to_state(obs, selected_mask_index):
    """Append gaze target phase one-hot to the proprioceptive state vector."""
    obs = dict(obs)
    state = np.asarray(obs["state"], dtype=np.float32)
    phase = gaze_phase_onehot(selected_mask_index).astype(np.float32)
    phase = np.broadcast_to(phase, (*state.shape[:-1], phase.shape[-1]))
    obs["state"] = np.concatenate([state, phase], axis=-1).astype(np.float32)
    return obs


def append_gaze_xy_to_state(obs, gaze_xy_norm):
    """Append the gaze position to the state, in the slot the one-hot used.

    Two numbers either way, so the observation space is unchanged and demos
    exported under the phase one-hot still load. What changes is the meaning:
    the grounding query is conditioned on where the operator is looking rather
    than on a classifier's verdict about which phase it is. Absent gaze becomes
    (-1, -1), outside the image box, so the conditioner can respond to "no
    fixation" instead of reading it as a corner.
    """
    obs = dict(obs)
    state = np.asarray(obs["state"], dtype=np.float32)
    if gaze_xy_norm is None:
        xy = np.asarray([-1.0, -1.0], dtype=np.float32)
    else:
        xy = np.asarray(gaze_xy_norm, dtype=np.float32).reshape(-1)[:2]
        if xy.shape[0] < 2:
            xy = np.asarray([-1.0, -1.0], dtype=np.float32)
    xy = np.broadcast_to(xy, (*state.shape[:-1], 2))
    obs["state"] = np.concatenate([state, xy], axis=-1).astype(np.float32)
    return obs


def add_gaze_mask_image_to_obs(
    obs,
    *,
    gaze_target_mask,
    image_key="front_camera_mask",
    reference_key="front_camera",
):
    """Add the selected target mask as a uint8 image observation."""
    obs = dict(obs)
    reference_image = latest_image(obs, reference_key)
    obs[image_key] = mask_to_image(
        gaze_target_mask,
        reference_image=reference_image,
        channels=3,
    )
    return obs


def select_gaze_target_mask(
    mask_probs,
    gaze_heatmap,
    *,
    target_shape,
    threshold: float = 0.5,
    dilation_px: int = 2,
    gaze_xy_norm=None,
    return_info: bool = False,
):
    """Return the predicted object mask containing the gaze peak, or zeros.

    ``gaze_xy_norm`` supplies the fixation directly and skips the heatmap,
    which is how the offline demo export reuses this: it has the eye tracker's
    own position and only needs the selection rule. Sharing the rule is the
    point -- the export used to carry a copy whose tie-break preferred the
    larger mask, and since the basket outweighs the ball by 60x, every
    ambiguous frame in a demo went to the basket while the same frame online
    went to whichever object the gaze was nearer.
    """
    target = np.zeros(target_shape, dtype=np.float32)
    info = {
        "selected_mask_index": None,
        "candidate_mask_indices": [],
        "gaze_xy_norm": None,
        "gaze_hit_mask": False,
    }
    if gaze_xy_norm is not None:
        xy_norm = tuple(float(v) for v in np.asarray(gaze_xy_norm).reshape(-1)[:2])
    else:
        xy_norm = gaze_xy_norm_from_heatmap(gaze_heatmap)
    info["gaze_xy_norm"] = xy_norm
    if mask_probs is None or xy_norm is None:
        return (target, info) if return_info else target

    mask_probs = np.asarray(mask_probs, dtype=np.float32)
    if mask_probs.ndim != 3 or mask_probs.shape[0] == 0:
        return (target, info) if return_info else target

    mask_height, mask_width = mask_probs.shape[-2:]
    gaze_x = int(round(np.clip(xy_norm[0], 0.0, 1.0) * (mask_width - 1)))
    gaze_y = int(round(np.clip(xy_norm[1], 0.0, 1.0) * (mask_height - 1)))
    binary_masks = mask_probs >= float(threshold)

    # One evidence source for both questions: the neighbourhood max of each
    # channel's probability at the gaze pixel. Thresholding it reproduces the
    # old dilated-mask containment test exactly, and scoring with it stops the
    # tie-break from ranking two sub-threshold noise values against each other.
    search_probs = np.stack(
        [_dilate_prob_map(prob, dilation_px) for prob in mask_probs],
        axis=0,
    )
    gaze_scores = search_probs[:, gaze_y, gaze_x]

    candidate_indices = [
        index
        for index in range(search_probs.shape[0])
        if float(gaze_scores[index]) >= float(threshold)
    ]
    info["candidate_mask_indices"] = candidate_indices
    if not candidate_indices:
        selected_index = 0
    elif len(candidate_indices) == 1:
        selected_index = candidate_indices[0]
    elif 0 in candidate_indices and 1 in candidate_indices and _mask_bbox_inside(
        binary_masks[0],
        binary_masks[1],
    ):
        selected_index = 1
    else:
        # Nearest mask wins, which is the rule the encoder's grounding target
        # is built with offline -- so the mask the policy sees and the mask the
        # query was taught to answer with agree on the ambiguous frames instead
        # of being decided by two unrelated criteria. Ranking by probability
        # was what broke here: on the frames that flipped, the values compared
        # were 0.000 (ball) against 0.066 (basket), both far under threshold,
        # because the candidacy test looks in a neighbourhood while the score
        # read the single gaze pixel. Distance is measured to the thresholded
        # silhouettes, the same ones the containment test used.
        def _gaze_distance(index):
            ys, xs = np.nonzero(binary_masks[index])
            if xs.size == 0:
                return np.inf
            return float(np.min(np.hypot(xs - gaze_x, ys - gaze_y)))

        selected_index = min(candidate_indices, key=_gaze_distance)

    selected_slot = f"mask{selected_index + 1}"
    selected = binary_masks[selected_index].astype(np.float32)
    selected = _resize_2d(selected, target_shape, method="nearest")
    selected = _postprocess_slot_mask(selected, selected_slot)
    info["selected_mask_index"] = selected_index
    # False when the fallback fired: nothing contained the fixation and index 0
    # was chosen by default, which is not the same as gaze landing on a mask.
    info["gaze_hit_mask"] = bool(candidate_indices)
    selected = selected.astype(np.float32)
    return (selected, info) if return_info else selected


def compute_gaze_target_mask_fields(
    obs,
    gaze_predictor,
    mask_predictor,
    gaze_heatmap_shape,
    *,
    dilation_px: int = 2,
    threshold: float = 0.5,
):
    """Compute gaze heatmap plus the mask selected by the gaze peak."""
    gaze_fields = compute_gaze_heatmap_fields(obs, gaze_predictor, gaze_heatmap_shape)
    target_shape = tuple(gaze_heatmap_shape)
    gaze_fields["selected_mask_index"] = None
    gaze_fields["selected_mask_slot"] = "none"
    gaze_fields["gaze_hit_mask"] = False
    gaze_fields["candidate_mask_indices"] = []
    if mask_predictor is None:
        gaze_fields["gaze_target_mask"] = np.zeros(target_shape, dtype=np.float32)
        return gaze_fields

    mask_probs = _predict_mask_probs(obs, mask_predictor)
    gaze_target_mask, select_info = select_gaze_target_mask(
        mask_probs,
        gaze_fields["gaze_heatmap"],
        target_shape=target_shape,
        threshold=threshold,
        dilation_px=dilation_px,
        return_info=True,
    )
    selected_index = select_info["selected_mask_index"]
    selected_slot = (
        mask_predictor.mask_slots[selected_index]
        if selected_index is not None and selected_index < len(mask_predictor.mask_slots)
        else "none"
    )
    gaze_fields["gaze_target_mask"] = gaze_target_mask
    gaze_fields["selected_mask_index"] = selected_index
    gaze_fields["selected_mask_slot"] = selected_slot
    gaze_fields["gaze_hit_mask"] = bool(select_info["gaze_hit_mask"])
    gaze_fields["candidate_mask_indices"] = select_info["candidate_mask_indices"]
    gaze_fields["gaze_xy_norm"] = select_info["gaze_xy_norm"]
    return gaze_fields


def compute_index_target_mask_fields(
    obs,
    mask_predictor,
    target_shape,
    *,
    selected_mask_index: int,
    threshold: float = 0.5,
):
    """Select mask1/mask2 directly by index, without using gaze."""
    fields = {
        "gaze_heatmap": None,
        "gaze_target_mask": np.zeros(tuple(target_shape), dtype=np.float32),
        "selected_mask_index": None,
        "selected_mask_slot": "none",
        "gaze_hit_mask": False,
        "candidate_mask_indices": [],
        "gaze_xy_norm": None,
    }
    if mask_predictor is None:
        return fields

    mask_probs = _predict_mask_probs(obs, mask_predictor)
    if mask_probs is None:
        return fields

    mask_probs = np.asarray(mask_probs, dtype=np.float32)
    selected_mask_index = int(selected_mask_index)
    if (
        mask_probs.ndim != 3
        or selected_mask_index < 0
        or selected_mask_index >= mask_probs.shape[0]
    ):
        return fields

    selected_slot = (
        mask_predictor.mask_slots[selected_mask_index]
        if selected_mask_index < len(mask_predictor.mask_slots)
        else f"mask{selected_mask_index + 1}"
    )
    selected_mask = (mask_probs[selected_mask_index] >= float(threshold)).astype(
        np.float32
    )
    selected_mask = _resize_2d(selected_mask, tuple(target_shape), method="nearest")
    selected_mask = _postprocess_slot_mask(selected_mask, selected_slot)
    fields["gaze_target_mask"] = selected_mask.astype(np.float32)
    fields["selected_mask_index"] = selected_mask_index
    fields["selected_mask_slot"] = selected_slot
    fields["gaze_hit_mask"] = True
    fields["candidate_mask_indices"] = [selected_mask_index]
    return fields


def compute_all_index_target_mask_fields(
    obs,
    mask_predictor,
    target_shape,
    *,
    threshold: float = 0.5,
):
    """Return mask1/mask2 fields in one RGB->mask predictor call."""
    fields_by_slot = {}
    if mask_predictor is None:
        return fields_by_slot

    mask_probs = _predict_mask_probs(obs, mask_predictor)
    if mask_probs is None:
        return fields_by_slot

    mask_probs = np.asarray(mask_probs, dtype=np.float32)
    if mask_probs.ndim != 3:
        return fields_by_slot

    for index in range(mask_probs.shape[0]):
        selected_slot = (
            mask_predictor.mask_slots[index]
            if index < len(mask_predictor.mask_slots)
            else f"mask{index + 1}"
        )
        selected_mask = (mask_probs[index] >= float(threshold)).astype(np.float32)
        selected_mask = _resize_2d(selected_mask, tuple(target_shape), method="nearest")
        selected_mask = _postprocess_slot_mask(selected_mask, selected_slot)
        fields_by_slot[selected_slot] = {
            "gaze_target_mask": selected_mask.astype(np.float32),
            "selected_mask_index": index,
            "selected_mask_slot": selected_slot,
            "gaze_hit_mask": True,
            "candidate_mask_indices": [index],
        }
    return fields_by_slot
