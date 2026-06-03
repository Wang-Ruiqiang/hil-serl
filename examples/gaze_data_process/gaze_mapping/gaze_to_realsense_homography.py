#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json

import cv2
import numpy as np


def _as_point_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class GazeHomography:
    """Maps eye-tracker pixel points to RealSense pixel points."""

    matrix: np.ndarray
    src_points: np.ndarray
    dst_points: np.ndarray
    src_size: tuple[int, int] | None = None
    dst_size: tuple[int, int] | None = None
    fit_label_count: int | None = None
    fit_inlier_count: int | None = None
    fit_mean_error_px: float | None = None
    fit_median_error_px: float | None = None
    fit_max_error_px: float | None = None

    @classmethod
    def from_points(
        cls,
        src_points: Sequence[Sequence[float]],
        dst_points: Sequence[Sequence[float]],
        src_size: tuple[int, int] | None = None,
        dst_size: tuple[int, int] | None = None,
    ) -> "GazeHomography":
        src = _as_point_array(src_points)
        dst = _as_point_array(dst_points)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cls(matrix=matrix, src_points=src, dst_points=dst, src_size=src_size, dst_size=dst_size)

    @classmethod
    def from_dict(cls, data: dict) -> "GazeHomography":
        matrix = np.asarray(data.get("homography"), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected homography shape (3, 3), got {matrix.shape}")
        src_points = _as_point_array(data["src_points"])
        dst_points = _as_point_array(data["dst_points"])
        src_size = tuple(data["src_size"]) if data.get("src_size") is not None else None
        dst_size = tuple(data["dst_size"]) if data.get("dst_size") is not None else None
        return cls(
            matrix=matrix,
            src_points=src_points,
            dst_points=dst_points,
            src_size=src_size,
            dst_size=dst_size,
            fit_label_count=data.get("fit_label_count"),
            fit_inlier_count=data.get("fit_inlier_count"),
            fit_mean_error_px=data.get("fit_mean_error_px"),
            fit_median_error_px=data.get("fit_median_error_px"),
            fit_max_error_px=data.get("fit_max_error_px"),
        )

    def to_dict(self) -> dict:
        return {
            "src_points": self.src_points.tolist(),
            "dst_points": self.dst_points.tolist(),
            "src_size": None if self.src_size is None else list(self.src_size),
            "dst_size": None if self.dst_size is None else list(self.dst_size),
            "fit_label_count": self.fit_label_count,
            "fit_inlier_count": self.fit_inlier_count,
            "fit_mean_error_px": self.fit_mean_error_px,
            "fit_median_error_px": self.fit_median_error_px,
            "fit_max_error_px": self.fit_max_error_px,
            "homography": self.matrix.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    def transform_point(self, point: Sequence[float]) -> tuple[float, float]:
        pt = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        mapped = cv2.perspectiveTransform(pt, self.matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def transform_points(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)
        return mapped

    def transform_point_normalized(self, point: Sequence[float], clip: bool = True) -> np.ndarray:
        if self.dst_size is None:
            raise ValueError("dst_size is required to normalize points.")
        w, h = self.dst_size
        x, y = self.transform_point(point)
        if clip:
            x = float(np.clip(x, 0.0, max(1.0, w - 1)))
            y = float(np.clip(y, 0.0, max(1.0, h - 1)))
        return np.array([x / max(1.0, w - 1), y / max(1.0, h - 1)], dtype=np.float32)


@dataclass(frozen=True)
class EpisodeHomographyMap:
    """Per-episode homographies keyed by episode_index."""

    episodes: dict[int, GazeHomography]

    def get(self, episode_index: int) -> GazeHomography | None:
        return self.episodes.get(int(episode_index))

    def transform_point(self, episode_index: int, point: Sequence[float]) -> tuple[float, float] | None:
        homography = self.get(episode_index)
        if homography is None:
            return None
        return homography.transform_point(point)

    def transform_point_normalized(
        self,
        episode_index: int,
        point: Sequence[float],
        clip: bool = True,
    ) -> np.ndarray | None:
        homography = self.get(episode_index)
        if homography is None:
            return None
        return homography.transform_point_normalized(point, clip=clip)

    def to_dict(self) -> dict:
        return {
            "type": "episode_homography_map",
            "episodes": [
                {
                    "episode_index": int(episode_index),
                    "homography": homography.to_dict(),
                }
                for episode_index, homography in sorted(self.episodes.items())
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeHomographyMap":
        episodes = {}
        for item in data.get("episodes", []):
            episode_index = int(item["episode_index"])
            homography_data = item.get("homography", item)
            episodes[episode_index] = GazeHomography.from_dict(homography_data)
        return cls(episodes=episodes)


def load_homography(path: str | Path) -> GazeHomography:
    path = Path(path)
    data = json.loads(path.read_text())
    return GazeHomography.from_dict(data)


def load_episode_homography_map(path: str | Path) -> EpisodeHomographyMap:
    path = Path(path)
    data = json.loads(path.read_text())
    return EpisodeHomographyMap.from_dict(data)


def load_homography_source(path: str | Path) -> GazeHomography | EpisodeHomographyMap:
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("type") == "episode_homography_map":
        return EpisodeHomographyMap.from_dict(data)
    if isinstance(data, dict) and "episodes" in data and isinstance(data["episodes"], list):
        return EpisodeHomographyMap.from_dict(data)
    return GazeHomography.from_dict(data)


def make_default_realsense_corners(width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [float(max(0, width - 1)), 0.0],
            [float(max(0, width - 1)), float(max(0, height - 1))],
            [0.0, float(max(0, height - 1))],
        ],
        dtype=np.float32,
    )
