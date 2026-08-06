from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import cv2
import numpy as np

from serl_launcher.utils.gaze_mask_utils import (
    add_gaze_mask_image_to_obs,
    append_gaze_phase_to_state,
    compute_all_index_target_mask_fields,
    compute_gaze_target_mask_fields,
    compute_index_target_mask_fields,
    load_mask_predictor,
)
from serl_launcher.utils.gaze_utils import (
    infer_heatmap_shape,
    load_gaze_predictor,
    update_env_gaze_prediction_overlay,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAZE_PREDICTOR_PATH = str(
    REPO_ROOT / "examples" / "gaze_data_process" / "gaze_heatmap_ckpt"
)
DEFAULT_MASK_PREDICTOR_PATH = str(
    REPO_ROOT
    / "examples"
    / "gaze_data_process"
    / "SAM_process"
    / "mask_predictor_ckpt"
    / "best.pt"
)


class GazeDerivedObservationWrapper(gym.ObservationWrapper):
    """Add frozen gaze/mask predictor outputs as image observations.

    This wrapper keeps train_rlpd clean: policy code receives a complete
    observation dict and does not need to know how gaze-derived modalities are
    computed.
    """

    def __init__(
        self,
        env,
        *,
        use_gaze_target_mask=True,
        source_image_key="front_camera",
        gaze_target_mask_key="front_camera_mask",
        mask1_key="front_camera_mask1",
        mask2_key="front_camera_mask2",
        gaze_predictor_checkpoint_path=DEFAULT_GAZE_PREDICTOR_PATH,
        mask_predictor_checkpoint_path=DEFAULT_MASK_PREDICTOR_PATH,
        mask_selection_mode="gaze",
        pick_classifier_checkpoint_path="examples/reward_classifier/classifier_ckpt_ball_pick",
        pick_classifier_threshold=0.95,
        pick_classifier_image_keys=("front_camera", "tactile_data"),
        gaze_target_mask_dilation=0,
        channels=3,
        log_fn=print,
    ):
        super().__init__(env)
        self.use_gaze_target_mask = bool(use_gaze_target_mask)
        self.source_image_key = source_image_key
        self.gaze_target_mask_key = gaze_target_mask_key
        self.mask1_key = mask1_key
        self.mask2_key = mask2_key
        self.gaze_predictor_checkpoint_path = gaze_predictor_checkpoint_path
        self.mask_predictor_checkpoint_path = mask_predictor_checkpoint_path
        self.mask_selection_mode = str(mask_selection_mode)
        self.pick_classifier_checkpoint_path = pick_classifier_checkpoint_path
        self.pick_classifier_threshold = float(pick_classifier_threshold)
        self.pick_classifier_image_keys = tuple(pick_classifier_image_keys)
        self.gaze_target_mask_dilation = int(gaze_target_mask_dilation)
        self.channels = int(channels)
        self.log_fn = log_fn
        self.gaze_predictor = None
        self.mask_predictor = None
        self.pick_classifier = None
        self.predictors_loaded = False
        self.pick_latched = False
        self.last_pick_classifier_prob = 0.0
        if hasattr(self.env, "config"):
            self.config = self.env.config

        spaces = OrderedDict(self.env.observation_space.spaces)
        if self.source_image_key not in spaces:
            raise KeyError(
                f"{self.source_image_key} is required to infer gaze-derived shapes."
            )
        source_space = spaces[self.source_image_key]
        height, width = source_space.shape[:2]
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, self.channels),
            dtype=np.uint8,
        )
        if self.use_gaze_target_mask:
            for mask_key in (self.gaze_target_mask_key, self.mask1_key, self.mask2_key):
                if mask_key not in spaces:
                    spaces[mask_key] = image_space
        if self.use_gaze_target_mask and "state" in spaces:
            state_space = spaces["state"]
            state_shape = list(state_space.shape)
            state_shape[-1] += 3
            spaces["state"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=tuple(state_shape),
                dtype=state_space.dtype,
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, **kwargs):
        self.pick_latched = False
        self.last_pick_classifier_prob = 0.0
        return super().reset(**kwargs)

    def _image_keys(self):
        return list(self.observation_space.spaces.keys())

    def _load_predictors(self, obs):
        if self.predictors_loaded:
            return
        self.gaze_predictor = (
            load_gaze_predictor(
                obs,
                self._image_keys(),
                self.gaze_predictor_checkpoint_path,
                preferred_key=self.source_image_key,
                log_fn=self.log_fn,
            )
            if self.use_gaze_target_mask and self.mask_selection_mode == "gaze"
            else None
        )
        self.mask_predictor = (
            load_mask_predictor(
                obs,
                self._image_keys(),
                self.mask_predictor_checkpoint_path,
                preferred_key=self.source_image_key,
                log_fn=self.log_fn,
            )
            if self.use_gaze_target_mask
            else None
        )
        if self.use_gaze_target_mask and self.mask_selection_mode == "pick_classifier":
            import jax

            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier_keys = [
                key for key in self.pick_classifier_image_keys if key in obs
            ]
            if not classifier_keys:
                raise KeyError(
                    "pick_classifier mask selection requires at least one classifier "
                    f"image key from {self.pick_classifier_image_keys}, got obs keys={list(obs.keys())}"
                )
            self.pick_classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample={key: obs[key] for key in classifier_keys},
                image_keys=classifier_keys,
                checkpoint_path=self.pick_classifier_checkpoint_path,
            )
            self.log_fn(
                "Loading frozen pick classifier for mask selection "
                f"checkpoint={self.pick_classifier_checkpoint_path} "
                f"image_keys={classifier_keys} threshold={self.pick_classifier_threshold}"
            )
        self.predictors_loaded = True

    def _pick_classifier_prob(self, obs):
        if self.pick_classifier is None:
            return 0.0
        classifier_obs = {
            key: obs[key]
            for key in self.pick_classifier_image_keys
            if key in obs
        }
        logits = np.asarray(self.pick_classifier(classifier_obs), dtype=np.float32)
        logit = float(logits.reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _zero_image(self, key):
        space = self.observation_space[key]
        return np.zeros(space.shape, dtype=space.dtype)

    def _display_gaze_mask(self, obs, selected_slot):
        try:
            env = self.env.unwrapped
            if not getattr(env, "display_image", False) or not hasattr(env, "img_queue"):
                return
            mask_image = np.asarray(obs[self.gaze_target_mask_key]).copy()
            if mask_image.ndim == 3 and mask_image.shape[-1] == 3:
                mask_image = mask_image[..., ::-1].copy()
            cv2.putText(
                mask_image,
                str(selected_slot),
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            env.img_queue.put({self.gaze_target_mask_key: mask_image})
        except Exception as exc:
            print(f"[warn] failed to display {self.gaze_target_mask_key}: {exc}")
            return

    def observation(self, obs):
        obs = dict(obs)
        if not self.use_gaze_target_mask:
            return obs

        self._load_predictors(obs)
        heatmap_shape = infer_heatmap_shape(
            obs,
            self._image_keys(),
            preferred_key=self.source_image_key,
        )

        if self.mask_selection_mode in ("pick_only", "place_only"):
            selected_index = 0 if self.mask_selection_mode == "pick_only" else 1
            fields = compute_index_target_mask_fields(
                obs,
                self.mask_predictor,
                heatmap_shape,
                selected_mask_index=selected_index,
            )
        elif self.mask_selection_mode == "pick_classifier":
            classifier_checked = False
            if self.pick_latched:
                pick_prob = self.last_pick_classifier_prob
            else:
                pick_prob = self._pick_classifier_prob(obs)
                self.last_pick_classifier_prob = pick_prob
                classifier_checked = True
                if pick_prob >= self.pick_classifier_threshold:
                    self.pick_latched = True
                    print("pick classifier latched: switch to mask2 until next reset")
            selected_index = 1 if self.pick_latched else 0
            fields = compute_index_target_mask_fields(
                obs,
                self.mask_predictor,
                heatmap_shape,
                selected_mask_index=selected_index,
            )
            fields["pick_classifier_prob"] = pick_prob
            fields["pick_latched"] = self.pick_latched
            fields["pick_classifier_checked"] = classifier_checked
        else:
            fields = compute_gaze_target_mask_fields(
                obs,
                self.gaze_predictor,
                self.mask_predictor,
                heatmap_shape,
                dilation_px=self.gaze_target_mask_dilation,
            )
        obs = add_gaze_mask_image_to_obs(
            obs,
            gaze_target_mask=fields["gaze_target_mask"],
            image_key=self.gaze_target_mask_key,
            reference_key=self.source_image_key,
        )

        slot_fields = compute_all_index_target_mask_fields(
            obs,
            self.mask_predictor,
            heatmap_shape,
        )
        for slot_name, image_key in (("mask1", self.mask1_key), ("mask2", self.mask2_key)):
            slot_mask = slot_fields.get(slot_name, {}).get(
                "gaze_target_mask",
                np.zeros(tuple(heatmap_shape), dtype=np.float32),
            )
            obs = add_gaze_mask_image_to_obs(
                obs,
                gaze_target_mask=slot_mask,
                image_key=image_key,
                reference_key=self.source_image_key,
            )
        obs = append_gaze_phase_to_state(obs, fields.get("selected_mask_index"))

        for mask_key in (self.gaze_target_mask_key, self.mask1_key, self.mask2_key):
            obs.setdefault(mask_key, self._zero_image(mask_key))
        selected_slot = fields.get("selected_mask_slot", "none")
        self._display_gaze_mask(obs, selected_slot)

        if self.mask_selection_mode == "gaze":
            update_env_gaze_prediction_overlay(
                self.env,
                fields.get("gaze_heatmap"),
                self.gaze_predictor,
            )
        return obs
