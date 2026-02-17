# sfm_sift.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------
# Data structures
# -------------------------

@dataclass
class SIFTSFMConfig:
    # Image handling
    image_max_side: Optional[int] = 1000
    preprocess_mode: str = "gray_clahe"  # gray | gray_clahe | gray_denoise_clahe

    # SIFT
    sift_nfeatures: int = 6500
    sift_contrast_thr: float = 0.03
    sift_edge_thr: float = 12.0
    sift_sigma: float = 1.6
    sift_ratio_thr: float = 0.78

    # Geometry
    affine_ransac_thr: float = 3.0
    homography_ransac_thr: float = 4.0
    essential_ransac_thr: float = 1.5

    min_matches_for_geom: int = 8
    min_affine_inliers_for_motion: int = 12
    min_affine_inliers_for_model: int = 10

    # Global motion model calibration (from old sfm_sift.py)
    calib_use_bidirectional: bool = True
    calib_max_pairs: Optional[int] = 260
    robust_iters: int = 5
    huber_delta: float = 35.0
    ridge: float = 1e-4

    # Fallbacks
    use_last_good_rel: bool = True
    fallback_rel: Tuple[float, float] = (0.0, 0.0)

    # Optional: compute expensive global model (kept optional)
    enable_global_model: bool = False


@dataclass
class PairMotion:
    id0: int
    id1: int
    success: bool
    reason: str

    matches: int
    affine_inliers: int
    homography_inliers: int
    sfm_inliers: int

    affine_M_1to0: Optional[np.ndarray]
    H_1to0: Optional[np.ndarray]

    # affine decomposition (from affine_M_1to0)
    tx: float
    ty: float
    rot_deg: float
    scale: float

    # optional direction from recoverPose (unit norm)
    sfm_t: Optional[np.ndarray]


@dataclass
class MotionLinearModel:
    beta: np.ndarray  # [d+1,2]
    feature_names: List[str]
    med_err: float
    mean_err: float
    used_pairs: int


# -------------------------
# Main class
# -------------------------

class SFM_Sift:
    """
    GNSS-denied localization helper for the SE3 Labs challenge.

    Key invariants (per your spec):
    - frames[id] is the single source of truth for GT, predictions, relative path, calibration, diagnostics.
    - ranges is a list of dicts: {"raw_start_id": int, "raw_end_id": int, "inverse": bool, "closure": bool}
      raw_start_id is always the smaller ID.
    - Each range is gapless and contains at most one contiguous test sub-block.
    - Ranges maximize train context before and after the test block; train runs (except first/last)
      appear in two consecutive ranges.
    - Predictions are stored in global map.png pixel coordinates in frames[id]["pred_global"].
    """

    def __init__(
        self,
        train_images: Path,
        test_images: Path,
        train_cam_csv: Path,
        test_cam_csv: Path,
        train_pos_csv: Path,
        map_path: Optional[Path] = None,
        config: Optional[SIFTSFMConfig] = None,
    ) -> None:
        self.train_images = Path(train_images)
        self.test_images = Path(test_images)
        self.train_cam_csv = Path(train_cam_csv)
        self.test_cam_csv = Path(test_cam_csv)
        self.train_pos_csv = Path(train_pos_csv)
        self.map_path = Path(map_path) if map_path is not None else None
        self.cfg = config or SIFTSFMConfig()

        # IDs
        self.train_ids: List[int] = []
        self.test_ids: List[int] = []
        self.all_ids: List[int] = []

        # Mask: train_mask[id] = True for train, False for test
        self.train_mask: Dict[int, bool] = {}

        # Single source of truth
        self.frames: Dict[int, Dict[str, Any]] = {}

        # Ranges
        self.ranges: List[Dict[str, Any]] = []

        # Range calibration (stored also into frames; this dict just avoids re-computation)
        self._range_calib: Dict[int, Dict[str, Any]] = {}

        # Old pairwise motion-to-map model (used for reliable relative frame deltas)
        self.calib_global_model: Optional[MotionLinearModel] = None
        self._fallback_forward = np.array([0.0, 0.0], dtype=np.float64)
        self._fallback_backward = np.array([0.0, 0.0], dtype=np.float64)

        # Optional cached map image and figure handles for debugging.
        self.map_rgb: Optional[np.ndarray] = None
        self.last_figures: Dict[str, object] = {}

        # Caches (performance)
        self._img_cache: Dict[Tuple[int, Optional[int]], Tuple[np.ndarray, float]] = {}
        self._preproc_cache: Dict[Tuple[int, Optional[int], str], Tuple[np.ndarray, float]] = {}
        self._feature_cache: Dict[
            Tuple[int, Optional[int], str, Tuple],
            Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray, np.ndarray, float],
        ] = {}
        self._pair_motion_cache: Dict[Tuple[int, int, Tuple], PairMotion] = {}

    # -------------------------
    # Dataset loading
    # -------------------------

    def load_dataset(self) -> None:
        train_pos_df = pd.read_csv(self.train_pos_csv)
        train_cam_df = pd.read_csv(self.train_cam_csv)
        test_cam_df = pd.read_csv(self.test_cam_csv)

        for c in ["id", "x_pixel", "y_pixel"]:
            if c not in train_pos_df.columns:
                raise KeyError(f"Missing required column in train_pos.csv: {c}")

        for c in ["id", "fx", "fy", "cx", "cy"]:
            if c not in train_cam_df.columns:
                raise KeyError(f"Missing required column in train_cam.csv: {c}")
            if c not in test_cam_df.columns:
                raise KeyError(f"Missing required column in test_cam.csv: {c}")

        train_pos_df = train_pos_df.copy()
        train_pos_df["id"] = train_pos_df["id"].astype(int)

        train_cam_df = train_cam_df.copy()
        train_cam_df["id"] = train_cam_df["id"].astype(int)

        test_cam_df = test_cam_df.copy()
        test_cam_df["id"] = test_cam_df["id"].astype(int)

        # IDs + consistency checks across CSVs
        train_pos_ids = set(int(x) for x in train_pos_df["id"].tolist())
        train_cam_ids = set(int(x) for x in train_cam_df["id"].tolist())
        test_cam_ids = set(int(x) for x in test_cam_df["id"].tolist())

        missing_train_cam = sorted(train_pos_ids - train_cam_ids)
        if len(missing_train_cam) > 0:
            raise KeyError(f"Missing train_cam rows for train_pos IDs: {missing_train_cam[:20]}")
        extra_train_cam = sorted(train_cam_ids - train_pos_ids)
        if len(extra_train_cam) > 0:
            raise KeyError(f"train_cam has IDs without train_pos rows: {extra_train_cam[:20]}")

        overlap = sorted(train_pos_ids & test_cam_ids)
        if len(overlap) > 0:
            raise ValueError(f"IDs cannot be both train and test: {overlap[:20]}")

        self.train_ids = sorted(train_pos_ids)
        self.test_ids = sorted(test_cam_ids)
        self.all_ids = sorted(set(self.train_ids) | set(self.test_ids))

        if len(self.all_ids) == 0:
            raise RuntimeError("No IDs found in dataset CSVs.")
        if len(self.train_ids) == 0:
            raise RuntimeError("No train IDs found in train_pos.csv.")
        if len(self.test_ids) == 0:
            raise RuntimeError("No test IDs found in test_cam.csv.")

        # Mask
        train_id_set = set(self.train_ids)
        self.train_mask = {iid: (iid in train_id_set) for iid in self.all_ids}

        # Maps
        gt_map: Dict[int, np.ndarray] = {
            int(r["id"]): np.array([float(r["x_pixel"]), float(r["y_pixel"])], dtype=np.float64)
            for _, r in train_pos_df.iterrows()
        }
        train_cam_map: Dict[int, Dict[str, float]] = {
            int(r["id"]): {
                "fx": float(r["fx"]),
                "fy": float(r["fy"]),
                "cx": float(r["cx"]),
                "cy": float(r["cy"]),
            }
            for _, r in train_cam_df.iterrows()
        }
        test_cam_map: Dict[int, Dict[str, float]] = {
            int(r["id"]): {
                "fx": float(r["fx"]),
                "fy": float(r["fy"]),
                "cx": float(r["cx"]),
                "cy": float(r["cy"]),
            }
            for _, r in test_cam_df.iterrows()
        }

        # Initialize frames
        self.frames = {}
        for iid in self.all_ids:
            src = "train" if self.train_mask[iid] else "test"
            cam = train_cam_map.get(iid) if src == "train" else test_cam_map.get(iid)
            if cam is None:
                raise KeyError(f"Missing camera intrinsics for id={iid} in {src}_cam.csv")

            img_path = self._resolve_image_path(iid, source=src)

            self.frames[iid] = {
                "id": int(iid),
                "source": src,
                "image_path": img_path,
                "cam": cam,
                "gt": gt_map.get(iid, None),  # None for test
                # Predictions and relative path (filled later)
                "pred_global": None,  # np.ndarray(2,) in map pixel coordinates
                "rel_to_anchor": None,  # np.ndarray(2,) in "relative path" coordinates
                # Per-step diagnostics (prev -> this)
                "step": {
                    "prev_id": None,
                    "mode": None,
                    "quality": None,
                    "d_rel_used": None,
                    "d_global_used": None,
                    "pair_motion": None,
                    "dx_pred": None,
                    "dy_pred": None,
                },
                # Range-related info (filled later)
                "range_meta": {"ranges": []},
                "range_calib": None,
                "closure": None,
            }

        # Load map if given
        self.map_rgb = None
        if self.map_path is not None and self.map_path.exists():
            bgr = cv2.imread(str(self.map_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read map image: {self.map_path}")
            self.map_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Assertions: isolated IDs
        all_set = set(self.all_ids)
        isolated = []
        for iid in self.all_ids:
            if (iid - 1 not in all_set) and (iid + 1 not in all_set):
                isolated.append(iid)
        assert len(isolated) == 0, f"Found isolated IDs (no neighbors): {isolated[:20]}"

        # Reset caches (safe after dataset set)
        self._reset_caches()

        # Clear ranges and calibrations
        self.ranges = []
        self._range_calib = {}
        self.last_figures = {}

        # Train the old robust motion-to-map delta model on train-train neighbors
        self._fit_motion_model_from_train_pairs()

    def _fit_motion_model_from_train_pairs(self) -> None:
        """
        Old sfm_sift.py logic:
        fit robust linear model from pair motion features -> GT pixel delta.
        """
        calib_pairs: List[Tuple[int, int]] = []
        for i in range(len(self.train_ids) - 1):
            a = int(self.train_ids[i])
            b = int(self.train_ids[i + 1])
            if b != a + 1:
                continue
            calib_pairs.append((a, b))
            if self.cfg.calib_use_bidirectional:
                calib_pairs.append((b, a))

        if self.cfg.calib_max_pairs is not None and len(calib_pairs) > int(self.cfg.calib_max_pairs):
            idx = np.linspace(0, len(calib_pairs) - 1, int(self.cfg.calib_max_pairs)).astype(int)
            calib_pairs = [calib_pairs[i] for i in idx]

        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        w_list: List[float] = []

        for id0, id1 in calib_pairs:
            gt0 = self.frames[id0].get("gt", None)
            gt1 = self.frames[id1].get("gt", None)
            if gt0 is None or gt1 is None:
                continue
            gt0 = np.asarray(gt0, dtype=np.float64).reshape(2)
            gt1 = np.asarray(gt1, dtype=np.float64).reshape(2)
            gt = gt1 - gt0

            m = self._estimate_pair_motion(id0, id1)
            if m.success:
                x_list.append(self._motion_to_feature_vector(m))
                y_list.append(gt.astype(np.float64))
                w_list.append(max(1.0, float(m.affine_inliers)))

        # Fallback deltas from train GT steps
        train_step_deltas = []
        for i in range(len(self.train_ids) - 1):
            a = int(self.train_ids[i])
            b = int(self.train_ids[i + 1])
            if b == a + 1:
                ga = self.frames[a].get("gt", None)
                gb = self.frames[b].get("gt", None)
                if ga is None or gb is None:
                    continue
                train_step_deltas.append(np.asarray(gb, dtype=np.float64).reshape(2) - np.asarray(ga, dtype=np.float64).reshape(2))
        if len(train_step_deltas) == 0:
            self._fallback_forward = np.array([0.0, 0.0], dtype=np.float64)
        else:
            self._fallback_forward = np.median(np.array(train_step_deltas, dtype=np.float64), axis=0)
        self._fallback_backward = -self._fallback_forward

        if len(x_list) == 0:
            self.calib_global_model = None
            return

        x = np.vstack(x_list)
        y = np.vstack(y_list)
        w = np.array(w_list, dtype=np.float64)

        beta = self._fit_robust_linear_model(
            x,
            y,
            w,
            robust_iters=self.cfg.robust_iters,
            huber_delta=self.cfg.huber_delta,
            ridge=self.cfg.ridge,
        )
        pred = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)]) @ beta
        err = np.linalg.norm(y - pred, axis=1)

        feature_names = ["tx", "ty", "rot_rad", "log_scale", "sfm_tx", "sfm_ty", "sfm_tz", "bias"]
        self.calib_global_model = MotionLinearModel(
            beta=beta,
            feature_names=feature_names,
            med_err=float(np.median(err)),
            mean_err=float(np.mean(err)),
            used_pairs=int(len(x)),
        )

    @staticmethod
    def _motion_to_feature_vector(m: PairMotion) -> np.ndarray:
        sfm_tx, sfm_ty, sfm_tz = 0.0, 0.0, 0.0
        if m.sfm_t is not None:
            t = m.sfm_t.astype(np.float64)
            tn = float(np.linalg.norm(t))
            if tn > 1e-12:
                t = t / tn
                sfm_tx, sfm_ty, sfm_tz = float(t[0]), float(t[1]), float(t[2])

        rot_rad = 0.0 if not np.isfinite(m.rot_deg) else float(np.deg2rad(m.rot_deg))
        scale_log = 0.0 if (not np.isfinite(m.scale) or m.scale <= 1e-8) else float(np.log(m.scale))
        tx = 0.0 if not np.isfinite(m.tx) else float(m.tx)
        ty = 0.0 if not np.isfinite(m.ty) else float(m.ty)
        return np.array([tx, ty, rot_rad, scale_log, sfm_tx, sfm_ty, sfm_tz], dtype=np.float64)

    @staticmethod
    def _fit_robust_linear_model(
        x: np.ndarray,
        y: np.ndarray,
        base_w: np.ndarray,
        robust_iters: int,
        huber_delta: float,
        ridge: float,
    ) -> np.ndarray:
        n, _d = x.shape
        xb = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
        w = np.clip(base_w.astype(np.float64), 1e-6, None).copy()

        beta = np.zeros((xb.shape[1], 2), dtype=np.float64)
        for _ in range(int(robust_iters)):
            ww = np.sqrt(w).reshape(-1, 1)
            xw = xb * ww
            yw = y * ww
            xtx = xw.T @ xw
            xty = xw.T @ yw
            xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
            beta = np.linalg.solve(xtx, xty)

            pred = xb @ beta
            err = np.linalg.norm(y - pred, axis=1)
            hub = np.ones_like(err)
            bad = err > float(huber_delta)
            hub[bad] = float(huber_delta) / np.clip(err[bad], 1e-6, None)
            w = np.clip(base_w, 1e-6, None) * hub

        return beta

    def _predict_delta(self, m: PairMotion) -> np.ndarray:
        if self.calib_global_model is None:
            if np.isfinite(m.tx) and np.isfinite(m.ty):
                return np.array([-float(m.tx), -float(m.ty)], dtype=np.float64)
            return self._fallback_forward.copy()
        x = self._motion_to_feature_vector(m)
        xb = np.concatenate([x, np.array([1.0], dtype=np.float64)], axis=0)
        y = xb @ self.calib_global_model.beta
        return y.astype(np.float64)

    # -------------------------
    # Range building
    # -------------------------

    def build_ranges(self) -> None:
        """
        Builds self.ranges as list of dicts:
        {"raw_start_id": int, "raw_end_id": int, "inverse": bool, "closure": bool}

        Construction rule (your clarified intent):
        - Start with gapless ID-intervals in all_ids (split at ID jumps).
        - Within each interval, create one range per contiguous TEST block.
        - Each range contains:
            (maximal contiguous TRAIN run immediately before the test block)
            + (the test block)
            + (maximal contiguous TRAIN run immediately after the test block)
          This produces overlap of intermediate train runs in two consecutive ranges.
        - If a range begins with test in forward direction, set inverse=True so processing starts at the trailing train run.
        - closure=True iff the range has train on both sides of the test block (i.e., includes both runs).
        """
        self._require_dataset()

        self.ranges = []
        range_idx = 0

        for interval_ids in self._split_into_gapless_intervals(self.all_ids):
            # Build a boolean list for this interval: True=train, False=test
            interval_is_train = [self.train_mask[iid] for iid in interval_ids]

            # Identify contiguous test blocks: list of (start_index, end_index) in interval index space
            test_blocks = self._find_contiguous_blocks(interval_is_train, value=False)

            # If no test block exists, you may skip (no predictions needed), but still could be useful for validation.
            # We'll skip by default to avoid ranges with no test.
            if len(test_blocks) == 0:
                continue

            for tb_s, tb_e in test_blocks:
                # Find maximal contiguous train run immediately before test block
                pre_s, pre_e = self._max_train_run_adjacent_left(interval_is_train, tb_s)
                # Find maximal contiguous train run immediately after test block
                post_s, post_e = self._max_train_run_adjacent_right(interval_is_train, tb_e)

                # Range span in interval index space
                span_s = pre_s if pre_s is not None else tb_s
                span_e = post_e if post_e is not None else tb_e

                raw_start_id = int(interval_ids[span_s])
                raw_end_id = int(interval_ids[span_e])
                if raw_start_id > raw_end_id:
                    raw_start_id, raw_end_id = raw_end_id, raw_start_id

                # Determine if forward range begins with a test id -> then inverse=True
                # Range forward start id corresponds to interval_ids[span_s]
                forward_starts_with_test = not interval_is_train[span_s]
                inverse = bool(forward_starts_with_test)

                closure = (pre_s is not None) and (post_s is not None)

                # Assert range length >= 2
                assert raw_end_id - raw_start_id >= 1, (
                    f"Range must have at least two IDs, got [{raw_start_id},{raw_end_id}]"
                )
                ids_forward = self._iter_range_ids(raw_start_id, raw_end_id, inverse=False)
                assert ids_forward == list(range(raw_start_id, raw_end_id + 1)), (
                    f"Range is not gapless: [{raw_start_id},{raw_end_id}]"
                )

                # Assert at most one test block inside [span_s..span_e]
                # (by construction with adjacency, this should hold)
                self._assert_single_test_block(interval_ids, interval_is_train, span_s, span_e)

                self.ranges.append(
                    {
                        "raw_start_id": raw_start_id,
                        "raw_end_id": raw_end_id,
                        "inverse": inverse,
                        "closure": bool(closure),
                    }
                )

                # Annotate frames with range_idx (no new index system; just helpful metadata)
                for iid in self._iter_range_ids(raw_start_id, raw_end_id, inverse=False):
                    if iid in self.frames:
                        # a frame can appear in multiple ranges; store as list
                        meta = self.frames[iid].get("range_meta", {})
                        lst = meta.get("ranges", [])
                        if range_idx not in lst:
                            lst.append(range_idx)
                        meta["ranges"] = lst
                        if "range_idx" not in meta:
                            meta["range_idx"] = int(range_idx)
                        self.frames[iid]["range_meta"] = meta

                range_idx += 1

        assert len(self.ranges) > 0, "No ranges constructed. Check dataset IDs and masks."

    # -------------------------
    # Prediction pipeline
    # -------------------------

    def predict_all_ranges(
        self,
        validation: bool = False,
        compute_train_inside: bool = True,
    ) -> None:
        """
        For each range:
        1) calibrate_range(range_plan) using train prefix before test block (inverse respected).
        2) propagate_path(range_plan) starting at first (as-test) frame after prefix.
        3) apply_closure(range_plan) if closure=True, only over (as-test) frames.
        """
        self._require_dataset()
        assert len(self.ranges) > 0, "Ranges not built. Call build_ranges() first."

        prev_calib: Optional[Dict[str, Any]] = None

        for r_idx, r in enumerate(self.ranges):
            calib = self.calibrate_range(r_idx, r, validation=validation, prev_calib=prev_calib)
            prev_calib = calib if calib is not None else prev_calib

            self.propagate_path(
                r_idx,
                r,
                validation=validation,
                compute_train_inside=compute_train_inside,
            )

            if bool(r.get("closure", False)):
                self.apply_closure(r_idx, r, validation=validation)

    def export_submission_csv(self, output_path: Path) -> pd.DataFrame:
        self._require_dataset()
        output_path = Path(output_path)

        rows: List[Dict[str, Any]] = []
        missing: List[int] = []
        for tid in sorted(self.test_ids):
            pred = self.frames[tid]["pred_global"]
            if pred is None or not np.isfinite(np.asarray(pred)).all():
                missing.append(tid)
                continue
            p = np.asarray(pred, dtype=np.float64).reshape(2)
            rows.append({"id": int(tid), "x_pixel": float(p[0]), "y_pixel": float(p[1])})

        assert len(missing) == 0, f"Missing predictions for test IDs: {missing[:20]}"

        df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    # -------------------------
    # Range-local calibration
    # -------------------------

    def calibrate_range(
        self,
        range_idx: int,
        range_plan: Dict[str, Any],
        validation: bool,
        prev_calib: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Determines local calibration based on TRAIN frames before the first (as-test) frame.
        - In normal mode: all train frames are GT-enabled.
        - In validation mode: only first two train frames in traversal are GT-enabled, and (optionally)
          only last two train frames are GT-enabled for closure; all other train frames behave as test.

        Stores:
        - range calibration dict in self._range_calib[range_idx]
        - frames[anchor_id]["range_calib"] (and optionally other frames) with scale/yaw info
        - Also computes train->train prefix step predictions (calibrated) and stores them in frames.
        """
        ids = self._iter_range_ids(
            int(range_plan["raw_start_id"]),
            int(range_plan["raw_end_id"]),
            inverse=bool(range_plan["inverse"]),
        )
        ids = list(ids)
        assert len(ids) >= 2, "Range must have at least two ids."

        gt_sets = self._gt_enabled_sets_for_range(ids, validation=validation)
        gt_enabled_prefix = gt_sets["prefix"]

        # Find prefix GT-enabled TRAIN frames before first as-test frame
        prefix_train = [iid for iid in ids if (iid in gt_enabled_prefix) and self.train_mask.get(iid, False)]

        # Need at least one anchor
        assert len(prefix_train) >= 1, (
            f"Range {range_idx}: cannot start calibration without a GT-enabled train anchor."
        )

        anchor_id = prefix_train[0]
        anchor_gt = self.frames[anchor_id]["gt"]
        assert anchor_gt is not None, "GT missing for anchor train frame."
        anchor_gt = np.asarray(anchor_gt, dtype=np.float64).reshape(2)

        # If only one train in prefix, inherit calibration
        if len(prefix_train) < 2:
            calib = None
            if prev_calib is not None:
                calib = dict(prev_calib)
                calib["source"] = "inherited"
                calib["anchor_id"] = int(anchor_id)
                calib["ref_id"] = None
            else:
                calib = {
                    "source": "none",
                    "scale": 1.0,
                    "yaw_deg": 0.0,
                    "anchor_id": int(anchor_id),
                    "ref_id": None,
                }
            self._range_calib[range_idx] = calib
            # Store calib in frames (at least in anchor)
            self.frames[anchor_id]["range_calib"] = calib
            return calib

        # Estimate similarity from consecutive train-prefix pairs (more robust than one pair)
        rel_vecs: List[np.ndarray] = []
        gt_vecs: List[np.ndarray] = []
        pair_rows: List[Tuple[int, int, PairMotion, np.ndarray, np.ndarray]] = []

        for p0, p1 in zip(prefix_train[:-1], prefix_train[1:]):
            gt0 = self.frames[p0]["gt"]
            gt1 = self.frames[p1]["gt"]
            if gt0 is None or gt1 is None:
                continue
            gt0 = np.asarray(gt0, dtype=np.float64).reshape(2)
            gt1 = np.asarray(gt1, dtype=np.float64).reshape(2)
            d_gt = gt1 - gt0
            motion = self._estimate_pair_motion(p0, p1)
            if not motion.success:
                continue
            d_rel = self._pair_motion_to_rel_delta(motion)
            if float(np.linalg.norm(d_rel)) < 1e-6 or float(np.linalg.norm(d_gt)) < 1e-6:
                continue
            rel_vecs.append(d_rel.astype(np.float64))
            gt_vecs.append(d_gt.astype(np.float64))
            pair_rows.append((p0, p1, motion, d_rel, d_gt))

        if len(rel_vecs) == 0:
            if prev_calib is not None:
                calib = dict(prev_calib)
                calib["source"] = "inherited"
                calib["anchor_id"] = int(anchor_id)
                calib["ref_id"] = int(prefix_train[1])
            else:
                calib = {
                    "source": "none",
                    "scale": 1.0,
                    "yaw_deg": 0.0,
                    "anchor_id": int(anchor_id),
                    "ref_id": int(prefix_train[1]),
                }
            self._range_calib[range_idx] = calib
            self.frames[anchor_id]["range_calib"] = calib
            return calib

        scale, yaw = self._estimate_similarity_from_vectors(rel_vecs, gt_vecs)
        ref_id = int(prefix_train[1])

        calib = {
            "source": "train_prefix",
            "scale": float(scale),
            "yaw_deg": float(np.degrees(yaw)),
            "anchor_id": int(anchor_id),
            "ref_id": int(ref_id),
        }
        self._range_calib[range_idx] = calib

        # Store in frames (anchor is enough; you can replicate to all ids if desired)
        self.frames[anchor_id]["range_calib"] = calib

        # Seed prefix predictions (anchor uses GT)
        self.frames[anchor_id]["pred_global"] = anchor_gt.copy()
        self.frames[anchor_id]["rel_to_anchor"] = np.array([0.0, 0.0], dtype=np.float64)

        for p0, p1, motion, d_rel, _d_gt in pair_rows:
            gt0 = np.asarray(self.frames[p0]["gt"], dtype=np.float64).reshape(2)
            gt1 = np.asarray(self.frames[p1]["gt"], dtype=np.float64).reshape(2)
            d_global = self._apply_similarity_to_vec(d_rel, scale=scale, yaw_rad=yaw)
            pred1 = gt0 + d_global

            # Keep train prefix as exact GT seeds for stable start,
            # but store calibrated step diagnostics for validation/debug.
            self.frames[p1]["pred_global"] = gt1.copy()
            prev_rel = self.frames[p0].get("rel_to_anchor", None)
            if prev_rel is None:
                prev_rel = np.array([0.0, 0.0], dtype=np.float64)
            self.frames[p1]["rel_to_anchor"] = np.asarray(prev_rel, dtype=np.float64).reshape(2) + d_rel

            self.frames[p1]["step"] = {
                "prev_id": int(p0),
                "mode": "gt_seed",
                "quality": {
                    "matches": int(motion.matches),
                    "affine_inliers": int(motion.affine_inliers),
                    "sfm_inliers": int(motion.sfm_inliers),
                },
                "d_rel_used": d_rel.copy(),
                "d_global_used": d_global.copy(),
                "pair_motion": motion,
                "dx_pred": float(d_global[0]),
                "dy_pred": float(d_global[1]),
                "x_gt": float(gt1[0]),
                "y_gt": float(gt1[1]),
                "x_pred": float(pred1[0]),
                "y_pred": float(pred1[1]),
            }

        return calib

    # -------------------------
    # Propagation
    # -------------------------

    def propagate_path(
        self,
        range_idx: int,
        range_plan: Dict[str, Any],
        validation: bool,
        compute_train_inside: bool,
    ) -> None:
        """
        Walk the range in processing direction. Use:
        - GT if GT-enabled for that frame (seed)
        - otherwise prior prediction and step calibrated delta (scale+yaw) from local calibration

        rel_to_anchor is always stored (for overlay/mosaic).
        pred_global is always stored in map pixel coordinates.
        """
        ids = list(
            self._iter_range_ids(
                int(range_plan["raw_start_id"]),
                int(range_plan["raw_end_id"]),
                inverse=bool(range_plan["inverse"]),
            )
        )
        assert len(ids) >= 2

        calib = self._range_calib.get(range_idx, None)
        if calib is None:
            # allow if calibrate_range wasn't called (shouldn't happen)
            calib = {"source": "none", "scale": 1.0, "yaw_deg": 0.0, "anchor_id": int(ids[0]), "ref_id": None}
            self._range_calib[range_idx] = calib

        scale = float(calib["scale"])
        yaw_rad = float(np.deg2rad(calib["yaw_deg"]))

        gt_sets = self._gt_enabled_sets_for_range(ids, validation=validation)
        gt_enabled_prefix = gt_sets["prefix"]
        gt_enabled_union = set(gt_sets["prefix"]) | set(gt_sets["closure"])

        # Robust seed: interpolate as-test frames between available GT-enabled train anchors.
        interp_seed_ids = self._prefill_interp_from_gt_anchors(
            ordered_ids=ids,
            gt_enabled_union=gt_enabled_union,
            validation=validation,
        )

        # Anchor is first id; must have GT-enabled train or already predicted
        anchor_id = ids[0]
        if self.train_mask.get(anchor_id, False) and ((not validation) or (anchor_id in gt_enabled_prefix)):
            gt = self.frames[anchor_id]["gt"]
            assert gt is not None
            self.frames[anchor_id]["pred_global"] = np.asarray(gt, dtype=np.float64).reshape(2)
        else:
            # If not GT-enabled (validation edge cases), require pred already set
            assert self.frames[anchor_id]["pred_global"] is not None, (
                f"Range {range_idx}: anchor has no GT and no prior prediction."
            )
        if self.frames[anchor_id]["rel_to_anchor"] is None:
            self.frames[anchor_id]["rel_to_anchor"] = np.array([0.0, 0.0], dtype=np.float64)

        last_good_rel: Optional[np.ndarray] = None

        for prev_id, curr_id in zip(ids[:-1], ids[1:]):
            prev_pred = self.frames[prev_id]["pred_global"]
            assert prev_pred is not None and np.isfinite(np.asarray(prev_pred)).all(), (
                f"Range {range_idx}: prev_id={prev_id} has no valid pred_global."
            )
            prev_pred = np.asarray(prev_pred, dtype=np.float64).reshape(2)

            prev_rel = self.frames[prev_id]["rel_to_anchor"]
            if prev_rel is None:
                prev_rel = np.array([0.0, 0.0], dtype=np.float64)

            # Decide whether curr is treated as GT-enabled seed
            curr_is_gt_enabled = self.train_mask.get(curr_id, False) and (
                (not validation) or (curr_id in gt_enabled_prefix)
            )
            curr_is_train = self.train_mask.get(curr_id, False)

            if curr_is_gt_enabled:
                # If we do not want to compute interior train frames, just set GT and reset rel accordingly.
                gt = self.frames[curr_id]["gt"]
                assert gt is not None
                gt = np.asarray(gt, dtype=np.float64).reshape(2)
                self.frames[curr_id]["pred_global"] = gt.copy()

                # Still compute rel_to_anchor for overlay continuity (optional)
                # If you prefer to keep overlay consistent, compute motion and update rel anyway.
                motion = self._estimate_pair_motion(prev_id, curr_id)
                d_rel = self._pair_motion_to_rel_delta(motion)
                self.frames[curr_id]["rel_to_anchor"] = prev_rel + d_rel

                self.frames[curr_id]["step"] = {
                    "prev_id": int(prev_id),
                    "mode": "gt",
                    "quality": {
                        "matches": int(motion.matches),
                        "affine_inliers": int(motion.affine_inliers),
                        "sfm_inliers": int(motion.sfm_inliers),
                    },
                    "d_rel_used": d_rel.copy(),
                    "d_global_used": None,
                    "pair_motion": motion,
                }
                continue

            if curr_is_train and (not compute_train_inside):
                # Keep chain stable without introducing extra SIFT drift on interior train frames.
                if self.frames[curr_id]["pred_global"] is None:
                    self.frames[curr_id]["pred_global"] = prev_pred.copy()
                self.frames[curr_id]["rel_to_anchor"] = np.asarray(prev_rel, dtype=np.float64).reshape(2)
                self.frames[curr_id]["step"] = {
                    "prev_id": int(prev_id),
                    "mode": "train_skip",
                    "quality": None,
                    "d_rel_used": None,
                    "d_global_used": None,
                    "pair_motion": None,
                }
                continue

            # If interpolation seed exists, prefer it over noisy SIFT motion.
            if curr_id in interp_seed_ids:
                pred_curr = np.asarray(self.frames[curr_id]["pred_global"], dtype=np.float64).reshape(2)
                d_global_interp = pred_curr - prev_pred
                self.frames[curr_id]["rel_to_anchor"] = np.asarray(prev_rel, dtype=np.float64).reshape(2) + d_global_interp
                self.frames[curr_id]["step"] = {
                    "prev_id": int(prev_id),
                    "mode": "interp_seed",
                    "quality": None,
                    "d_rel_used": None,
                    "d_global_used": d_global_interp.copy(),
                    "pair_motion": None,
                }
                continue

            # Compute motion and choose relative delta
            motion = self._estimate_pair_motion(prev_id, curr_id)
            d_rel = self._pair_motion_to_rel_delta(motion)

            mode: str
            if motion.success:
                mode = "model"
                last_good_rel = d_rel.copy()
            elif self.cfg.use_last_good_rel and last_good_rel is not None:
                d_rel = last_good_rel.copy()
                mode = "last_good"
            else:
                d_rel = np.array(self.cfg.fallback_rel, dtype=np.float64).reshape(2)
                mode = "fallback"

            d_global = self._apply_similarity_to_vec(d_rel, scale=scale, yaw_rad=yaw_rad)
            pred_curr = prev_pred + d_global

            self.frames[curr_id]["pred_global"] = pred_curr.copy()
            self.frames[curr_id]["rel_to_anchor"] = prev_rel + d_rel

            self.frames[curr_id]["step"] = {
                "prev_id": int(prev_id),
                "mode": mode,
                "quality": {
                    "matches": int(motion.matches),
                    "affine_inliers": int(motion.affine_inliers),
                    "sfm_inliers": int(motion.sfm_inliers),
                },
                "d_rel_used": d_rel.copy(),
                "d_global_used": d_global.copy(),
                "pair_motion": motion,
            }

    # -------------------------
    # Closure
    # -------------------------

    def apply_closure(
        self,
        range_idx: int,
        range_plan: Dict[str, Any],
        validation: bool,
    ) -> None:
        """
        Applies a loop-closure-like correction:
        - Determine closing GT-enabled TRAIN frame at the end of traversal.
        - Compute error e = gt_close - pred_close.
        - Distribute linearly over frames treated as test (as-test) in the range.
        - Only apply to (as-test) frames (never to GT-enabled train frames).
        """
        ids = list(
            self._iter_range_ids(
                int(range_plan["raw_start_id"]),
                int(range_plan["raw_end_id"]),
                inverse=bool(range_plan["inverse"]),
            )
        )
        gt_sets = self._gt_enabled_sets_for_range(ids, validation=validation)
        gt_enabled_closure = gt_sets["closure"]
        block_info = self._range_block_info(ids)

        # Find closing GT-enabled train frame in suffix after test block.
        test_end_idx = int(block_info["test_end_idx"])
        closing_id = None
        for iid in ids[test_end_idx + 1 :]:
            if (iid in gt_enabled_closure) and self.train_mask.get(iid, False):
                closing_id = iid
                break

        if closing_id is None:
            return

        gt_close = self.frames[closing_id]["gt"]
        pred_close = self.frames[closing_id]["pred_global"]

        if gt_close is None or pred_close is None:
            return

        gt_close = np.asarray(gt_close, dtype=np.float64).reshape(2)
        pred_close = np.asarray(pred_close, dtype=np.float64).reshape(2)

        e = gt_close - pred_close
        e_norm = float(np.linalg.norm(e))

        # Frames to correct:
        # - normal mode: true test IDs in the test block
        # - validation mode: all IDs treated as test in this range
        if not validation:
            as_test_ids = [iid for iid in block_info["test_block_ids"] if iid != closing_id]
        else:
            as_test_ids = []
            for iid in ids:
                if iid == closing_id:
                    continue
                if not self._is_gt_enabled(iid, range_plan, validation=True):
                    as_test_ids.append(iid)

        n = len(as_test_ids)
        if n == 0:
            return

        # Linear distribution: alpha = (k+1)/(n+1)
        for k, iid in enumerate(as_test_ids):
            pred = self.frames[iid]["pred_global"]
            if pred is None:
                continue
            pred = np.asarray(pred, dtype=np.float64).reshape(2)
            alpha = float(k + 1) / float(n + 1)
            pred_corr = pred + alpha * e
            self.frames[iid]["pred_global"] = pred_corr

        # Store closure info (at closing frame)
        self.frames[closing_id]["closure"] = {
            "range_idx": int(range_idx),
            "closing_id": int(closing_id),
            "error_x": float(e[0]),
            "error_y": float(e[1]),
            "error_norm": float(e_norm),
            "num_corrected": int(n),
        }

    # -------------------------
    # Visualization (interactive if Plotly available)
    # -------------------------

    def plot_map_interactive(
        self,
        show_ids: bool = True,
        show_train_pred: bool = True,
        title: str = "Predictions on map.png",
    ):
        """
        Interactive plot (Plotly) with hover/zoom:
        - Train GT (blue)
        - Test predictions (red)
        - Optionally train predictions (green) for validation debugging
        """
        self._require_dataset()
        if self.map_rgb is None:
            raise FileNotFoundError("map.png not loaded or missing. Provide map_path and call load_dataset().")

        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("Plotly is required for interactive plotting. Install: pip install plotly") from e
        try:
            from PIL import Image
        except Exception as e:
            raise ImportError("Pillow is required for map overlay plotting. Install: pip install pillow") from e

        map_img = self.map_rgb
        h, w = map_img.shape[:2]

        # Collect points
        train_x, train_y, train_text = [], [], []
        test_x, test_y, test_text = [], [], []
        trainp_x, trainp_y, trainp_text = [], [], []
        prev_test_id: Optional[int] = None

        for iid in self.all_ids:
            fr = self.frames[iid]
            gt = fr["gt"]
            pred = fr["pred_global"]

            hover = self._frame_hover_text(iid)

            if fr["source"] == "train" and gt is not None:
                gt = np.asarray(gt, dtype=np.float64).reshape(2)
                train_x.append(float(gt[0]))
                train_y.append(float(gt[1]))
                train_text.append(hover)

            if fr["source"] == "test" and pred is not None and np.isfinite(np.asarray(pred)).all():
                pred = np.asarray(pred, dtype=np.float64).reshape(2)
                if prev_test_id is not None and int(iid) != int(prev_test_id) + 1:
                    # break line between disjoint ID segments
                    test_x.append(None)
                    test_y.append(None)
                    test_text.append("")
                test_x.append(float(pred[0]))
                test_y.append(float(pred[1]))
                test_text.append(hover)
                prev_test_id = int(iid)

            if show_train_pred and fr["source"] == "train" and pred is not None and np.isfinite(np.asarray(pred)).all():
                pred = np.asarray(pred, dtype=np.float64).reshape(2)
                trainp_x.append(float(pred[0]))
                trainp_y.append(float(pred[1]))
                trainp_text.append(hover)

        fig = go.Figure()

        # background image
        fig.add_layout_image(
            dict(
                source=Image.fromarray(map_img),  # <-- FIX: numpy -> PIL
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=w,
                sizey=h,
                sizing="stretch",
                layer="below",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=train_x,
                y=train_y,
                mode="markers+text" if show_ids else "markers",
                name="train GT",
                text=[str(self.frames_id_from_hover(t)) for t in train_text] if show_ids else None,
                textposition="top center",
                hovertext=train_text,
                hoverinfo="text",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=test_x,
                y=test_y,
                mode="markers+lines+text" if show_ids else "markers+lines",
                name="test pred",
                text=[str(self.frames_id_from_hover(t)) for t in test_text] if show_ids else None,
                textposition="top center",
                hovertext=test_text,
                hoverinfo="text",
            )
        )

        if show_train_pred and len(trainp_x) > 0:
            fig.add_trace(
                go.Scatter(
                    x=trainp_x,
                    y=trainp_y,
                    mode="markers",
                    name="train pred",
                    hovertext=trainp_text,
                    hoverinfo="text",
                )
            )

        # Pixel coordinate system: origin top-left, y down
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0])

        fig.update_layout(
            title=title,
            width=min(1200, w),
            height=min(900, h),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        self.last_figures["map"] = fig
        return fig

    def plot_rel_interactive(
        self,
        range_idx: int,
        title: str = "Relative path (rel_to_anchor)",
    ):
        """
        Interactive plot (Plotly) in relative coordinates for overlay debugging.
        Hover shows frame info (gt/pred/rel/motion quality).
        """
        self._require_dataset()
        assert 0 <= range_idx < len(self.ranges), "Invalid range_idx"
        r = self.ranges[range_idx]

        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("Plotly is required for interactive plotting. Install: pip install plotly") from e

        ids = list(self._iter_range_ids(r["raw_start_id"], r["raw_end_id"], inverse=bool(r["inverse"])))

        xs_t, ys_t, ht_t = [], [], []
        xs_s, ys_s, ht_s = [], [], []

        for iid in ids:
            fr = self.frames[iid]
            rel = fr["rel_to_anchor"]
            if rel is None or not np.isfinite(np.asarray(rel)).all():
                continue
            rel = np.asarray(rel, dtype=np.float64).reshape(2)
            hover = self._frame_hover_text(iid)

            if fr["source"] == "train":
                xs_t.append(float(rel[0]))
                ys_t.append(float(rel[1]))
                ht_t.append(hover)
            else:
                xs_s.append(float(rel[0]))
                ys_s.append(float(rel[1]))
                ht_s.append(hover)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs_t, y=ys_t, mode="markers+lines", name="train (rel)", hovertext=ht_t, hoverinfo="text"))
        fig.add_trace(go.Scatter(x=xs_s, y=ys_s, mode="markers+lines", name="test (rel)", hovertext=ht_s, hoverinfo="text"))
        fig.update_layout(title=title, width=900, height=700, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_yaxes(autorange="reversed")  # image-like
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        self.last_figures[f"rel_range_{range_idx}"] = fig
        return fig

    # -------------------------
    # Pair visualization (optional)
    # -------------------------

    def plot_pair_matches(self, id0: int, id1: int, max_side: Optional[int] = 900, top_k: int = 120):
        """
        Quick diagnostic: draw SIFT matches.
        Uses OpenCV drawMatches; returns RGB image as numpy array.
        """
        self._require_dataset()
        id0 = int(id0)
        id1 = int(id1)

        rgb0, _ = self._load_image_rgb(id0, max_side=max_side)
        rgb1, _ = self._load_image_rgb(id1, max_side=max_side)

        k0, d0, g0, _kmat0, _ = self._get_features_for_id(id0)
        k1, d1, g1, _kmat1, _ = self._get_features_for_id(id1)

        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            raise RuntimeError("No descriptors for one of the images.")

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(d0, d1, k=2)
        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < float(self.cfg.sift_ratio_thr) * n.distance:
                good.append(m)

        good = sorted(good, key=lambda m: m.distance)[: int(top_k)]
        out = cv2.drawMatches(
            cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR),
            k0,
            cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR),
            k1,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb

    # Cell X — Overlay plot to validate relative positions (image mosaic like your screenshot)


    def plot_range_overlay(
        self,
        range_idx: int,
        alpha: float = 0.22,
        max_side: Optional[int] = None,
        pad: int = 250,
        label_every: int = 1,
        use_center_anchors: bool = True,
        max_frames: Optional[int] = None,
        figsize=(12, 9),
    ):
        """
        Visualizes rel_to_anchor by compositing all frames of a range into one mosaic.
        Uses frames[id]["rel_to_anchor"] (in resized-image pixel units) and places each image accordingly.

        Requirements:
        - sfm.predict_all_ranges(...) must have been run
        - sfm.frames[id]["rel_to_anchor"] must exist for those ids
        """
        assert 0 <= range_idx < len(self.ranges)
        r = self.ranges[range_idx]

        ids = list(
            self._iter_range_ids(
                int(r["raw_start_id"]),
                int(r["raw_end_id"]),
                inverse=bool(r["inverse"]),
            )
        )
        if max_frames is not None and len(ids) > max_frames:
            # uniform subsample to keep plot readable
            idx = np.linspace(0, len(ids) - 1, max_frames).astype(int)
            ids = [ids[i] for i in idx]

        # Load all images in the same scale used for SIFT (important: rel_to_anchor is in that pixel space)
        if max_side is None:
            max_side = self.cfg.image_max_side

        imgs = []
        centers = []
        rels = []

        for iid in ids:
            rel = self.frames[iid].get("rel_to_anchor", None)
            if rel is None or not np.isfinite(np.asarray(rel)).all():
                continue

            rgb, _scale = self._load_image_rgb(int(iid), max_side=max_side)
            h, w = rgb.shape[:2]
            c = np.array([w * 0.5, h * 0.5], dtype=np.float64)

            imgs.append((iid, rgb))
            centers.append(c)
            rels.append(np.asarray(rel, dtype=np.float64).reshape(2))

        assert len(imgs) >= 2, "Not enough frames with valid rel_to_anchor for overlay."

        centers = np.vstack(centers)
        rels = np.vstack(rels)

        # Choose anchor placement in the mosaic canvas
        # Use first frame as anchor (rel_to_anchor ≈ (0,0))
        # We'll place anchor center at (pad + something) computed from extents.
        # Compute where each image center would land relative to anchor center.
        rel_center_offsets = rels.copy()  # in pixel units of resized images

        # Extents for canvas sizing (include image half-sizes)
        half_sizes = centers.copy()  # [w/2, h/2] per image but in (x,y) order
        min_xy = np.min(rel_center_offsets - half_sizes, axis=0)
        max_xy = np.max(rel_center_offsets + half_sizes, axis=0)

        canvas_w = int(np.ceil((max_xy[0] - min_xy[0]) + 2 * pad))
        canvas_h = int(np.ceil((max_xy[1] - min_xy[1]) + 2 * pad))
        canvas_w = max(canvas_w, 200)
        canvas_h = max(canvas_h, 200)

        # Anchor center in canvas coords so that everything fits
        anchor_center = np.array([pad - min_xy[0], pad - min_xy[1]], dtype=np.float64)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

        # Draw images (alpha blend)
        placed_centers = []
        placed_ids = []
        for (iid, rgb), c, rel in zip(imgs, centers, rels):
            img = rgb.astype(np.float32) / 255.0

            # Image center position in canvas
            cen = anchor_center + rel
            placed_centers.append(cen)
            placed_ids.append(iid)

            # top-left corner
            tl = cen - c
            x0 = int(np.floor(tl[0]))
            y0 = int(np.floor(tl[1]))

            h, w = img.shape[:2]
            x1 = x0 + w
            y1 = y0 + h

            # clip
            cx0 = max(0, x0)
            cy0 = max(0, y0)
            cx1 = min(canvas_w, x1)
            cy1 = min(canvas_h, y1)
            if cx0 >= cx1 or cy0 >= cy1:
                continue

            ix0 = cx0 - x0
            iy0 = cy0 - y0
            ix1 = ix0 + (cx1 - cx0)
            iy1 = iy0 + (cy1 - cy0)

            patch = img[iy0:iy1, ix0:ix1]
            canvas[cy0:cy1, cx0:cx1] = (1 - alpha) * canvas[cy0:cy1, cx0:cx1] + alpha * patch

        canvas_u8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

        # Plot + annotate centers and IDs
        placed_centers = np.vstack(placed_centers)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(canvas_u8)
        ax.scatter(placed_centers[:, 0], placed_centers[:, 1], s=30, c="white", edgecolors="black", linewidths=0.5)

        for i, (iid, (x, y)) in enumerate(zip(placed_ids, placed_centers)):
            if label_every > 1 and (i % label_every) != 0:
                continue
            ax.text(x + 4, y + 4, str(iid), fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none"))

        ax.set_title(f"Merged range {r['raw_start_id']} -> {r['raw_end_id']} (center anchors)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        self.last_figures[f"overlay_range_{range_idx}"] = fig


    # -------------------------
    # Internal: helpers
    # -------------------------

    def _require_dataset(self) -> None:
        if len(self.all_ids) == 0 or len(self.frames) == 0:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

    def _reset_caches(self) -> None:
        self._img_cache.clear()
        self._preproc_cache.clear()
        self._feature_cache.clear()
        self._pair_motion_cache.clear()

    @staticmethod
    def _split_into_gapless_intervals(all_ids: Sequence[int]) -> List[List[int]]:
        if len(all_ids) == 0:
            return []
        out: List[List[int]] = []
        curr = [int(all_ids[0])]
        for x in all_ids[1:]:
            x = int(x)
            if x == curr[-1] + 1:
                curr.append(x)
            else:
                out.append(curr)
                curr = [x]
        out.append(curr)
        return out

    @staticmethod
    def _find_contiguous_blocks(mask: Sequence[bool], value: bool) -> List[Tuple[int, int]]:
        """
        Returns (start_idx, end_idx) inclusive blocks where mask[idx] == value.
        """
        blocks: List[Tuple[int, int]] = []
        n = len(mask)
        i = 0
        while i < n:
            if bool(mask[i]) != bool(value):
                i += 1
                continue
            s = i
            while i + 1 < n and bool(mask[i + 1]) == bool(value):
                i += 1
            e = i
            blocks.append((s, e))
            i += 1
        return blocks

    @staticmethod
    def _max_train_run_adjacent_left(is_train: Sequence[bool], test_block_start: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (start,end) inclusive of maximal contiguous TRAIN run immediately left of test_block_start.
        If no train adjacent, returns (None,None).
        """
        if test_block_start - 1 < 0:
            return None, None
        if not bool(is_train[test_block_start - 1]):
            return None, None
        e = test_block_start - 1
        s = e
        while s - 1 >= 0 and bool(is_train[s - 1]):
            s -= 1
        return s, e

    @staticmethod
    def _max_train_run_adjacent_right(is_train: Sequence[bool], test_block_end: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (start,end) inclusive of maximal contiguous TRAIN run immediately right of test_block_end.
        If no train adjacent, returns (None,None).
        """
        if test_block_end + 1 >= len(is_train):
            return None, None
        if not bool(is_train[test_block_end + 1]):
            return None, None
        s = test_block_end + 1
        e = s
        while e + 1 < len(is_train) and bool(is_train[e + 1]):
            e += 1
        return s, e

    def _assert_single_test_block(self, interval_ids: Sequence[int], is_train: Sequence[bool], s: int, e: int) -> None:
        sub = [not bool(x) for x in is_train[s : e + 1]]  # True where test
        blocks = self._find_contiguous_blocks(sub, value=True)
        assert len(blocks) <= 1, (
            "Range violates 'single test block' constraint. "
            f"IDs span [{interval_ids[s]}..{interval_ids[e]}], test blocks={blocks}"
        )

    @staticmethod
    def _iter_range_ids(raw_start_id: int, raw_end_id: int, inverse: bool) -> List[int]:
        a = int(raw_start_id)
        b = int(raw_end_id)
        assert a <= b
        if not bool(inverse):
            return list(range(a, b + 1))
        return list(range(b, a - 1, -1))

    def _range_block_info(self, ordered_ids: Sequence[int]) -> Dict[str, Any]:
        """
        Returns structural info for an ordered range traversal.
        Assumes at most one contiguous test block.
        """
        ids = [int(i) for i in ordered_ids]
        test_idx = [k for k, iid in enumerate(ids) if not self.train_mask.get(iid, False)]
        assert len(test_idx) > 0, "Range has no test IDs; cannot derive block info."

        s = int(min(test_idx))
        e = int(max(test_idx))
        # single contiguous block invariant
        for k in range(s, e + 1):
            assert not self.train_mask.get(ids[k], False), "Range violates contiguous test-block invariant."

        prefix_train_ids = [iid for iid in ids[:s] if self.train_mask.get(iid, False)]
        suffix_train_ids = [iid for iid in ids[e + 1 :] if self.train_mask.get(iid, False)]
        return {
            "test_start_idx": s,
            "test_end_idx": e,
            "test_block_ids": ids[s : e + 1],
            "prefix_train_ids": prefix_train_ids,
            "suffix_train_ids": suffix_train_ids,
        }

    def _gt_enabled_sets_for_range(self, ordered_ids: Sequence[int], validation: bool) -> Dict[str, Set[int]]:
        """
        Returns:
        - prefix: GT frames allowed for calibration/seeding before test propagation
        - closure: GT frames allowed as closing references after test propagation
        """
        ids = [int(i) for i in ordered_ids]
        info = self._range_block_info(ids)
        train_in_order = [iid for iid in ids if self.train_mask.get(iid, False)]

        if not validation:
            prefix = set(int(i) for i in info["prefix_train_ids"])
            closure = set(int(i) for i in info["suffix_train_ids"])
            return {"prefix": prefix, "closure": closure}

        prefix = set(int(i) for i in train_in_order[:2])
        closure = set(int(i) for i in train_in_order[-2:])
        return {"prefix": prefix, "closure": closure}

    def _gt_enabled_set(self, ordered_ids: Sequence[int], validation: bool) -> Set[int]:
        """
        Backward-compatible helper:
        returns union of prefix+closure GT-enabled sets for this range.
        """
        sets = self._gt_enabled_sets_for_range(ordered_ids, validation=validation)
        return set(sets["prefix"]) | set(sets["closure"])

    def _is_gt_enabled(self, image_id: int, range_plan: Dict[str, Any], validation: bool) -> bool:
        """
        Returns whether GT is available for a frame under current validation semantics.
        """
        iid = int(image_id)
        if not self.train_mask.get(iid, False):
            return False
        if not validation:
            return True
        ordered_ids = self._iter_range_ids(
            int(range_plan["raw_start_id"]),
            int(range_plan["raw_end_id"]),
            inverse=bool(range_plan["inverse"]),
        )
        gt_sets = self._gt_enabled_sets_for_range(ordered_ids, validation=True)
        return (iid in gt_sets["prefix"]) or (iid in gt_sets["closure"])

    def _prefill_interp_from_gt_anchors(
        self,
        ordered_ids: Sequence[int],
        gt_enabled_union: Set[int],
        validation: bool,
    ) -> Set[int]:
        """
        Fills pred_global for as-test frames via linear interpolation between GT-enabled train anchors.
        Returns IDs that received interpolation seeds.
        """
        ids = [int(i) for i in ordered_ids]
        anchor_idx: List[int] = []
        for k, iid in enumerate(ids):
            if self.train_mask.get(iid, False) and iid in gt_enabled_union:
                gt = self.frames[iid].get("gt", None)
                if gt is not None and np.isfinite(np.asarray(gt, dtype=np.float64)).all():
                    anchor_idx.append(k)

        if len(anchor_idx) < 2:
            return set()

        seeded: Set[int] = set()
        def should_seed(iid: int) -> bool:
            is_train = self.train_mask.get(iid, False)
            if (not validation) and is_train:
                return False
            if validation and is_train and (iid in gt_enabled_union):
                return False
            return True

        for ka, kb in zip(anchor_idx[:-1], anchor_idx[1:]):
            ida = ids[ka]
            idb = ids[kb]
            ga = np.asarray(self.frames[ida]["gt"], dtype=np.float64).reshape(2)
            gb = np.asarray(self.frames[idb]["gt"], dtype=np.float64).reshape(2)
            span = int(kb - ka)
            if span <= 1:
                continue

            for k in range(ka + 1, kb):
                iid = ids[k]
                if not should_seed(iid):
                    continue

                alpha = float(k - ka) / float(span)
                pred = (1.0 - alpha) * ga + alpha * gb
                self.frames[iid]["pred_global"] = pred.astype(np.float64)
                seeded.add(iid)

        # Extrapolate right side (after last anchor)
        if len(anchor_idx) >= 2:
            k0 = anchor_idx[-2]
            k1 = anchor_idx[-1]
            id0 = ids[k0]
            id1 = ids[k1]
            g0 = np.asarray(self.frames[id0]["gt"], dtype=np.float64).reshape(2)
            g1 = np.asarray(self.frames[id1]["gt"], dtype=np.float64).reshape(2)
            span = float(max(1, k1 - k0))
            v = (g1 - g0) / span
            for k in range(k1 + 1, len(ids)):
                iid = ids[k]
                if not should_seed(iid):
                    continue
                pred = g1 + float(k - k1) * v
                self.frames[iid]["pred_global"] = pred.astype(np.float64)
                seeded.add(iid)

            # Extrapolate left side (before first anchor)
            k0 = anchor_idx[0]
            k1 = anchor_idx[1]
            id0 = ids[k0]
            id1 = ids[k1]
            g0 = np.asarray(self.frames[id0]["gt"], dtype=np.float64).reshape(2)
            g1 = np.asarray(self.frames[id1]["gt"], dtype=np.float64).reshape(2)
            span = float(max(1, k1 - k0))
            v = (g1 - g0) / span
            for k in range(0, k0):
                iid = ids[k]
                if not should_seed(iid):
                    continue
                pred = g0 + float(k - k0) * v
                self.frames[iid]["pred_global"] = pred.astype(np.float64)
                seeded.add(iid)

        return seeded

    @staticmethod
    def _estimate_similarity_from_vectors(rel_vecs: Sequence[np.ndarray], gt_vecs: Sequence[np.ndarray]) -> Tuple[float, float]:
        """
        Least-squares similarity estimate (scale + yaw) from vector pairs:
        gt ~= scale * R(yaw) * rel
        """
        v = np.asarray(rel_vecs, dtype=np.float64).reshape(-1, 2)
        w = np.asarray(gt_vecs, dtype=np.float64).reshape(-1, 2)
        assert v.shape == w.shape and v.shape[0] >= 1

        # Orthogonal Procrustes on vector sets (no translation)
        a = w.T @ v
        u, svals, vt = np.linalg.svd(a)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vt

        denom = float(np.sum(v * v))
        if denom < 1e-12:
            return 1.0, 0.0

        scale = float(np.sum(svals) / denom)
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        return scale, yaw

    def _infer_source(self, image_id: int) -> str:
        if self.train_mask.get(int(image_id), False):
            return "train"
        return "test"

    def _resolve_image_path(self, image_id: int, source: Optional[str] = None) -> Path:
        iid = int(image_id)
        src = source or self._infer_source(iid)
        base_dir = self.train_images if src == "train" else self.test_images
        stems = [f"{iid:04d}", str(iid)]
        exts = [".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"]
        for st in stems:
            for ext in exts:
                p = base_dir / f"{st}{ext}"
                if p.exists():
                    return p
        raise FileNotFoundError(f"Image not found for id={iid} in {base_dir}")

    def _load_image_rgb(self, image_id: int, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
        key = (int(image_id), max_side)
        if key in self._img_cache:
            return self._img_cache[key]

        path = self._resolve_image_path(int(image_id))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        scale = 1.0
        if max_side is not None:
            h, w = rgb.shape[:2]
            m = max(h, w)
            if m > int(max_side):
                scale = float(max_side) / float(m)
                nw = max(32, int(round(w * scale)))
                nh = max(32, int(round(h * scale)))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        self._img_cache[key] = (rgb, scale)
        return rgb, scale

    @staticmethod
    def _preprocess_for_sift(img_rgb: np.ndarray, mode: str) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if mode == "gray":
            return gray
        if mode == "gray_clahe":
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(gray)
        if mode == "gray_denoise_clahe":
            gray2 = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            return clahe.apply(gray2)
        raise KeyError(f"Unknown preprocess_mode: {mode}")

    def _load_preprocessed_gray(self, image_id: int, max_side: Optional[int], mode: str) -> Tuple[np.ndarray, float]:
        key = (int(image_id), max_side, str(mode))
        if key in self._preproc_cache:
            return self._preproc_cache[key]
        rgb, scale = self._load_image_rgb(int(image_id), max_side=max_side)
        gray = self._preprocess_for_sift(rgb, mode=mode)
        self._preproc_cache[key] = (gray, scale)
        return gray, scale

    def _scaled_k(self, image_id: int, image_scale: float) -> np.ndarray:
        cam = self.frames[int(image_id)]["cam"]
        fx = float(cam["fx"]) * float(image_scale)
        fy = float(cam["fy"]) * float(image_scale)
        cx = float(cam["cx"]) * float(image_scale)
        cy = float(cam["cy"]) * float(image_scale)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _get_features_for_id(
        self,
        image_id: int,
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray, np.ndarray, float]:
        cfg_key = (
            int(self.cfg.sift_nfeatures),
            float(self.cfg.sift_contrast_thr),
            float(self.cfg.sift_edge_thr),
            float(self.cfg.sift_sigma),
            str(self.cfg.preprocess_mode),
        )
        key = (int(image_id), self.cfg.image_max_side, self.cfg.preprocess_mode, cfg_key)
        if key in self._feature_cache:
            return self._feature_cache[key]

        gray, scale = self._load_preprocessed_gray(
            int(image_id),
            max_side=self.cfg.image_max_side,
            mode=self.cfg.preprocess_mode,
        )
        k = self._scaled_k(int(image_id), image_scale=scale)

        sift = cv2.SIFT_create(
            nfeatures=int(self.cfg.sift_nfeatures),
            contrastThreshold=float(self.cfg.sift_contrast_thr),
            edgeThreshold=float(self.cfg.sift_edge_thr),
            sigma=float(self.cfg.sift_sigma),
        )
        keypoints, desc = sift.detectAndCompute(gray, None)
        if keypoints is None:
            keypoints = []

        self._feature_cache[key] = (keypoints, desc, gray, k, scale)
        return self._feature_cache[key]

    def _estimate_pair_motion(self, id0: int, id1: int) -> PairMotion:
        cfg_key = (
            int(self.cfg.sift_nfeatures),
            float(self.cfg.sift_contrast_thr),
            float(self.cfg.sift_edge_thr),
            float(self.cfg.sift_sigma),
            float(self.cfg.sift_ratio_thr),
        )
        cache_key = (int(id0), int(id1), cfg_key)
        if cache_key in self._pair_motion_cache:
            return self._pair_motion_cache[cache_key]

        k0, d0, _g0, k_mat0, _ = self._get_features_for_id(int(id0))
        k1, d1, _g1, k_mat1, _ = self._get_features_for_id(int(id1))

        if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
            out = PairMotion(int(id0), int(id1), False, "no_descriptors", 0, 0, 0, 0, None, None, np.nan, np.nan, np.nan, np.nan, None)
            self._pair_motion_cache[cache_key] = out
            return out

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(d0, d1, k=2)
        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < float(self.cfg.sift_ratio_thr) * n.distance:
                good.append(m)

        if len(good) < int(self.cfg.min_matches_for_geom):
            out = PairMotion(
                int(id0),
                int(id1),
                False,
                "few_matches",
                len(good),
                0,
                0,
                0,
                None,
                None,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                None,
            )
            self._pair_motion_cache[cache_key] = out
            return out

        pts0 = np.array([k0[m.queryIdx].pt for m in good], dtype=np.float32)
        pts1 = np.array([k1[m.trainIdx].pt for m in good], dtype=np.float32)

        m_aff, inl_aff = cv2.estimateAffinePartial2D(
            pts1,
            pts0,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.cfg.affine_ransac_thr),
            maxIters=20000,
            confidence=0.999,
            refineIters=50,
        )
        if inl_aff is not None:
            inl_aff = inl_aff.ravel().astype(bool)
            n_inl_aff = int(inl_aff.sum())
        else:
            inl_aff = np.zeros((len(good),), dtype=bool)
            n_inl_aff = 0

        if m_aff is not None:
            rot_deg = float(np.degrees(np.arctan2(m_aff[1, 0], m_aff[0, 0])))
            scale = float(np.sqrt(m_aff[0, 0] ** 2 + m_aff[1, 0] ** 2))
            tx = float(m_aff[0, 2])
            ty = float(m_aff[1, 2])
        else:
            rot_deg = np.nan
            scale = np.nan
            tx = np.nan
            ty = np.nan

        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        h_mat, inl_h = cv2.findHomography(
            pts1,
            pts0,
            method=method,
            ransacReprojThreshold=float(self.cfg.homography_ransac_thr),
            maxIters=20000,
            confidence=0.999,
        )
        if inl_h is not None:
            n_inl_h = int(inl_h.ravel().astype(bool).sum())
        else:
            n_inl_h = 0

        sfm_t = None
        n_inl_pose = 0
        if len(good) >= 8:
            pts0n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), k_mat0, None).reshape(-1, 2)
            pts1n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), k_mat1, None).reshape(-1, 2)
            e_mat, inl_e = cv2.findEssentialMat(
                pts0n,
                pts1n,
                cameraMatrix=np.eye(3),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=float(self.cfg.essential_ransac_thr),
            )
            if e_mat is not None and inl_e is not None and int(inl_e.sum()) >= 8:
                _, _r, t, inl_pose = cv2.recoverPose(
                    e_mat,
                    pts0n,
                    pts1n,
                    cameraMatrix=np.eye(3),
                    mask=inl_e.astype(np.uint8).reshape(-1, 1),
                )
                sfm_t = t.reshape(3).astype(np.float64)
                if inl_pose is not None:
                    n_inl_pose = int(inl_pose.ravel().astype(bool).sum())

        success = (m_aff is not None) and (n_inl_aff >= int(self.cfg.min_affine_inliers_for_model))
        out = PairMotion(
            id0=int(id0),
            id1=int(id1),
            success=bool(success),
            reason="ok" if success else "weak_affine",
            matches=int(len(good)),
            affine_inliers=int(n_inl_aff),
            homography_inliers=int(n_inl_h),
            sfm_inliers=int(n_inl_pose),
            affine_M_1to0=m_aff,
            H_1to0=h_mat,
            tx=float(tx),
            ty=float(ty),
            rot_deg=float(rot_deg),
            scale=float(scale),
            sfm_t=sfm_t,
        )
        self._pair_motion_cache[cache_key] = out
        return out

    @staticmethod
    def _apply_similarity_to_vec(v: np.ndarray, scale: float, yaw_rad: float) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(2)
        c = float(np.cos(yaw_rad))
        s = float(np.sin(yaw_rad))
        r = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (scale * (r @ v)).astype(np.float64)

    def _pair_motion_to_rel_delta(self, m: PairMotion) -> np.ndarray:
        """
        Relative step delta from old sfm_sift.py logic.
        If the robust linear model is available, use it.
        Otherwise fallback to affine translation sign-converted.
        """
        if m.success:
            return self._predict_delta(m)
        if np.isfinite(m.tx) and np.isfinite(m.ty):
            return np.array([-float(m.tx), -float(m.ty)], dtype=np.float64)
        return np.array([0.0, 0.0], dtype=np.float64)

    def _frame_hover_text(self, iid: int) -> str:
        fr = self.frames[int(iid)]
        gt = fr["gt"]
        pred = fr["pred_global"]
        rel = fr["rel_to_anchor"]
        step = fr.get("step", {})
        calib = fr.get("range_calib", None)

        def fmt_xy(x):
            if x is None:
                return "None"
            a = np.asarray(x, dtype=np.float64).reshape(-1)
            if a.size >= 2 and np.isfinite(a[:2]).all():
                return f"({a[0]:.2f}, {a[1]:.2f})"
            return "None"

        lines = []
        lines.append(f"id={iid} | source={fr['source']}")
        lines.append(f"gt={fmt_xy(gt)}")
        lines.append(f"pred_global={fmt_xy(pred)}")
        lines.append(f"rel_to_anchor={fmt_xy(rel)}")
        if calib is not None:
            lines.append(f"calib: scale={calib.get('scale', None)} yaw_deg={calib.get('yaw_deg', None)}")
        if step and step.get("prev_id", None) is not None:
            q = step.get("quality", None)
            lines.append(f"step: prev={step.get('prev_id')} mode={step.get('mode')}")
            if isinstance(q, dict):
                lines.append(f"matches={q.get('matches')} inl_aff={q.get('affine_inliers')} sfm_inl={q.get('sfm_inliers')}")
        cl = fr.get("closure", None)
        if isinstance(cl, dict):
            lines.append(f"closure: err_norm={cl.get('error_norm'):.2f}px corrected={cl.get('num_corrected')}")
        return "<br>".join(lines)

    @staticmethod
    def frames_id_from_hover(hover_text: str) -> int:
        # Hover starts with "id=XYZ | ..."
        try:
            prefix = hover_text.split("<br>")[0]
            val = prefix.split("|")[0].strip()
            return int(val.split("=")[1])
        except Exception:
            return -1
