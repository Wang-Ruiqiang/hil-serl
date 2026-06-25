from collections import OrderedDict

import gymnasium as gym
import numpy as np

from serl_launcher.utils.gaze_mask_utils import (
    add_gaze_target_mask_image_to_obs,
    compute_gaze_target_mask_fields,
    load_mask_predictor,
)
from serl_launcher.utils.gaze_utils import (
    infer_heatmap_shape,
    load_gaze_predictor,
    update_env_gaze_prediction_overlay,
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
        gaze_target_mask_key="gaze_target_mask",
        gaze_predictor_checkpoint_path="examples/gaze_data_process/gaze_heatmap_ckpt",
        mask_predictor_checkpoint_path=(
            "examples/gaze_data_process/SAM_process/mask_predictor_ckpt/best.pt"
        ),
        gaze_target_mask_dilation=2,
        channels=3,
        log_fn=print,
    ):
        super().__init__(env)
        self.use_gaze_target_mask = bool(use_gaze_target_mask)
        self.source_image_key = source_image_key
        self.gaze_target_mask_key = gaze_target_mask_key
        self.gaze_predictor_checkpoint_path = gaze_predictor_checkpoint_path
        self.mask_predictor_checkpoint_path = mask_predictor_checkpoint_path
        self.gaze_target_mask_dilation = int(gaze_target_mask_dilation)
        self.channels = int(channels)
        self.log_fn = log_fn
        self.gaze_predictor = None
        self.mask_predictor = None
        self.predictors_loaded = False
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
        if self.use_gaze_target_mask and self.gaze_target_mask_key not in spaces:
            spaces[self.gaze_target_mask_key] = image_space
        self.observation_space = gym.spaces.Dict(spaces)

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
            if self.use_gaze_target_mask
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
        self.predictors_loaded = True

    def _zero_image(self, key):
        space = self.observation_space[key]
        return np.zeros(space.shape, dtype=space.dtype)

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

        fields = compute_gaze_target_mask_fields(
            obs,
            self.gaze_predictor,
            self.mask_predictor,
            heatmap_shape,
            dilation_px=self.gaze_target_mask_dilation,
        )
        obs = add_gaze_target_mask_image_to_obs(
            obs,
            gaze_target_mask=fields["gaze_target_mask"],
            image_key=self.gaze_target_mask_key,
            reference_key=self.source_image_key,
        )

        obs.setdefault(
            self.gaze_target_mask_key,
            self._zero_image(self.gaze_target_mask_key),
        )
        print(f"gaze_target_mask = {fields.get('selected_mask_slot', 'none')}")

        update_env_gaze_prediction_overlay(
            self.env,
            fields.get("gaze_heatmap"),
            self.gaze_predictor,
        )
        return obs
