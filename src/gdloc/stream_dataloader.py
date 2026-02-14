from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


FeatureExtractor = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class StreamFrame:
    frame_id: int
    split: str
    image_path: Path
    image_rgb: np.ndarray
    intrinsics: CameraIntrinsics
    init_xy: Optional[Tuple[float, float]]
    gt_xy: Optional[Tuple[float, float]]
    pose_source: str
    feature: Any
    frame_time_ms: float
    map_size_after: int


@dataclass
class MapObservation:
    frame_id: int
    xy: Tuple[float, float]
    intrinsics: CameraIntrinsics
    feature: Any
    source: str


class GlobalFeatureMap:
    """Simple incremental store for global map observations."""

    def __init__(self) -> None:
        self._obs: List[MapObservation] = []

    def add(
        self,
        frame_id: int,
        xy: Tuple[float, float],
        intrinsics: CameraIntrinsics,
        feature: Any,
        source: str,
    ) -> None:
        self._obs.append(
            MapObservation(
                frame_id=frame_id,
                xy=(float(xy[0]), float(xy[1])),
                intrinsics=intrinsics,
                feature=feature,
                source=source,
            )
        )

    def __len__(self) -> int:
        return len(self._obs)

    @property
    def observations(self) -> Sequence[MapObservation]:
        return tuple(self._obs)


class GNSSStreamDataLoader:
    """
    Stream-like loader for GNSS-denied localization experiments.

    Behavior:
    - Training split: uses GT position as simulated GNSS (`pose_source='gnss'`)
      and incrementally expands the global feature map.
    - Validation/Test split: does not use GT for initialization.
      Initial pose uses frame `id-1` estimate when available, otherwise last known estimate.
    - Measures and prints per-frame processing duration.
    """

    def __init__(
        self,
        data_root: Path,
        train_fraction: float = 0.8,
        split_seed: int = 42,
        shuffle_train_ids: bool = True,
    ) -> None:
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be between 0 and 1.")

        self.data_root = Path(data_root)
        self.train_fraction = float(train_fraction)
        self.split_seed = int(split_seed)
        self.shuffle_train_ids = bool(shuffle_train_ids)

        self.train_img_dir = self.data_root / "train_data" / "train_images"
        self.test_img_dir = self.data_root / "test_data" / "test_images"

        self.train_cam_csv = self.data_root / "train_data" / "train_cam.csv"
        self.train_pos_csv = self.data_root / "train_data" / "train_pos.csv"
        self.test_cam_csv = self.data_root / "test_data" / "test_cam.csv"

        self.global_feature_map = GlobalFeatureMap()
        self._estimated_xy_by_id: Dict[int, Tuple[float, float]] = {}
        self._last_estimate_xy: Optional[Tuple[float, float]] = None

        self.train_stream_df: pd.DataFrame
        self.val_stream_df: pd.DataFrame
        self.test_stream_df: pd.DataFrame

        self._prepare_splits()

    @staticmethod
    def default_feature_extractor(image_rgb: np.ndarray) -> np.ndarray:
        """Low-cost fallback descriptor (normalized RGB histogram)."""
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def _prepare_splits(self) -> None:
        train_cam = pd.read_csv(self.train_cam_csv)
        train_pos = pd.read_csv(self.train_pos_csv)
        test_cam = pd.read_csv(self.test_cam_csv)

        train_full = train_cam.merge(train_pos, on="id", how="inner")
        train_full["id"] = train_full["id"].astype(int)
        test_cam["id"] = test_cam["id"].astype(int)

        ids = train_full["id"].to_numpy(copy=True)
        ids = np.unique(ids)

        if self.shuffle_train_ids:
            rng = np.random.default_rng(self.split_seed)
            rng.shuffle(ids)

        n_train = int(round(len(ids) * self.train_fraction))
        n_train = max(1, min(n_train, len(ids) - 1))

        train_ids = set(int(x) for x in ids[:n_train])
        val_ids = set(int(x) for x in ids[n_train:])

        self.train_stream_df = (
            train_full[train_full["id"].isin(train_ids)]
            .copy()
            .sort_values("id")
            .reset_index(drop=True)
        )
        self.val_stream_df = (
            train_full[train_full["id"].isin(val_ids)]
            .copy()
            .sort_values("id")
            .reset_index(drop=True)
        )
        self.test_stream_df = test_cam.copy().sort_values("id").reset_index(drop=True)

    def reset_runtime_state(self) -> None:
        self.global_feature_map = GlobalFeatureMap()
        self._estimated_xy_by_id.clear()
        self._last_estimate_xy = None

    def _resolve_image_path(self, frame_id: int, is_train: bool) -> Path:
        folder = self.train_img_dir if is_train else self.test_img_dir
        id_raw = str(int(frame_id))
        id_pad = f"{int(frame_id):04d}"
        extensions = [".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"]

        for stem in (id_pad, id_raw):
            for ext in extensions:
                p = folder / f"{stem}{ext}"
                if p.exists():
                    return p

        raise FileNotFoundError(f"No image found for frame id={frame_id} in {folder}")

    def _load_rgb(self, image_path: Path) -> np.ndarray:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _prior_from_previous_id(self, frame_id: int) -> Optional[Tuple[float, float]]:
        prev_id = int(frame_id) - 1
        if prev_id in self._estimated_xy_by_id:
            return self._estimated_xy_by_id[prev_id]
        return self._last_estimate_xy

    def _iter_split(
        self,
        split: str,
        frame_df: pd.DataFrame,
        use_gt_for_init: bool,
        include_gt_in_output: bool,
        feature_extractor: Optional[FeatureExtractor],
        update_global_map: bool,
        print_timing: bool,
    ) -> Iterator[StreamFrame]:
        extractor = feature_extractor or self.default_feature_extractor

        for row in frame_df.itertuples(index=False):
            t0 = perf_counter()

            frame_id = int(row.id)
            image_path = self._resolve_image_path(frame_id, is_train=(split in {"train", "val"}))
            image_rgb = self._load_rgb(image_path)
            intr = CameraIntrinsics(float(row.fx), float(row.fy), float(row.cx), float(row.cy))

            gt_xy = None
            if hasattr(row, "x_pixel") and hasattr(row, "y_pixel"):
                gt_xy = (float(row.x_pixel), float(row.y_pixel))

            if use_gt_for_init:
                init_xy = gt_xy
                pose_source = "gnss"
            else:
                init_xy = self._prior_from_previous_id(frame_id)
                pose_source = "prev_id" if init_xy is not None else "none"

            feature = extractor(image_rgb)

            if update_global_map and init_xy is not None:
                self.global_feature_map.add(
                    frame_id=frame_id,
                    xy=init_xy,
                    intrinsics=intr,
                    feature=feature,
                    source=pose_source,
                )

            if init_xy is not None:
                self._estimated_xy_by_id[frame_id] = init_xy
                self._last_estimate_xy = init_xy

            frame_time_ms = (perf_counter() - t0) * 1000.0

            out_gt = gt_xy if include_gt_in_output else None
            frame = StreamFrame(
                frame_id=frame_id,
                split=split,
                image_path=image_path,
                image_rgb=image_rgb,
                intrinsics=intr,
                init_xy=init_xy,
                gt_xy=out_gt,
                pose_source=pose_source,
                feature=feature,
                frame_time_ms=frame_time_ms,
                map_size_after=len(self.global_feature_map),
            )

            if print_timing:
                print(
                    f"[{split}] id={frame_id} dt={frame_time_ms:.2f}ms "
                    f"init={init_xy} source={pose_source} map_size={frame.map_size_after}"
                )

            yield frame

    def stream_train(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        update_global_map: bool = True,
        print_timing: bool = True,
    ) -> Iterator[StreamFrame]:
        return self._iter_split(
            split="train",
            frame_df=self.train_stream_df,
            use_gt_for_init=True,
            include_gt_in_output=True,
            feature_extractor=feature_extractor,
            update_global_map=update_global_map,
            print_timing=print_timing,
        )

    def stream_val(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        update_global_map: bool = True,
        print_timing: bool = True,
        include_gt_in_output: bool = False,
    ) -> Iterator[StreamFrame]:
        return self._iter_split(
            split="val",
            frame_df=self.val_stream_df,
            use_gt_for_init=False,
            include_gt_in_output=include_gt_in_output,
            feature_extractor=feature_extractor,
            update_global_map=update_global_map,
            print_timing=print_timing,
        )

    def stream_test(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        update_global_map: bool = False,
        print_timing: bool = True,
    ) -> Iterator[StreamFrame]:
        return self._iter_split(
            split="test",
            frame_df=self.test_stream_df,
            use_gt_for_init=False,
            include_gt_in_output=False,
            feature_extractor=feature_extractor,
            update_global_map=update_global_map,
            print_timing=print_timing,
        )

    def run_train_val_stream(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        print_timing: bool = True,
        expand_map_with_val: bool = True,
        max_train_frames: Optional[int] = None,
        max_val_frames: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Runs train -> val stream and returns timing summary.
        Resets runtime state before execution.
        """
        self.reset_runtime_state()

        train_times: List[float] = []
        val_times: List[float] = []

        for frame in self.stream_train(
            feature_extractor=feature_extractor,
            update_global_map=True,
            print_timing=print_timing,
        ):
            train_times.append(frame.frame_time_ms)
            if max_train_frames is not None and len(train_times) >= int(max_train_frames):
                break

        for frame in self.stream_val(
            feature_extractor=feature_extractor,
            update_global_map=expand_map_with_val,
            print_timing=print_timing,
            include_gt_in_output=False,
        ):
            val_times.append(frame.frame_time_ms)
            if max_val_frames is not None and len(val_times) >= int(max_val_frames):
                break

        train_mean = float(np.mean(train_times)) if train_times else 0.0
        val_mean = float(np.mean(val_times)) if val_times else 0.0

        return {
            "train_frames": float(len(train_times)),
            "val_frames": float(len(val_times)),
            "train_mean_ms": train_mean,
            "val_mean_ms": val_mean,
            "global_map_size": float(len(self.global_feature_map)),
        }

    def describe(self) -> Dict[str, int]:
        return {
            "train_frames": int(len(self.train_stream_df)),
            "val_frames": int(len(self.val_stream_df)),
            "test_frames": int(len(self.test_stream_df)),
            "map_observations": int(len(self.global_feature_map)),
        }


__all__ = [
    "CameraIntrinsics",
    "FeatureExtractor",
    "GlobalFeatureMap",
    "GNSSStreamDataLoader",
    "MapObservation",
    "StreamFrame",
]
