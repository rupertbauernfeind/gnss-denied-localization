from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SIFTSFMConfig:
    image_max_side: Optional[int] = 1000
    preprocess_mode: str = "gray_clahe"

    sift_nfeatures: int = 6500
    sift_contrast_thr: float = 0.03
    sift_edge_thr: float = 12.0
    sift_sigma: float = 1.6
    sift_ratio_thr: float = 0.78

    affine_ransac_thr: float = 3.0
    homography_ransac_thr: float = 4.0
    essential_ransac_thr: float = 1.5
    min_matches_for_geom: int = 8
    min_affine_inliers_for_model: int = 10

    calib_use_bidirectional: bool = True
    calib_max_pairs: Optional[int] = 260
    robust_iters: int = 5
    huber_delta: float = 35.0
    ridge: float = 1e-4

    auto_switch_to_next_anchor: bool = True
    anchor_switch_min_inliers: int = 16
    closure_min_inliers: int = 12

    seq_min_affine_inliers: int = 120


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
    tx: float
    ty: float
    rot_deg: float
    scale: float
    sfm_t: Optional[np.ndarray]


@dataclass
class MotionLinearModel:
    beta: np.ndarray  # [d+1,2]
    feature_names: List[str]
    med_err: float
    mean_err: float
    used_pairs: int


@dataclass
class SegmentPlan:
    segment_idx: int
    start_id: int
    end_id: int
    length: int
    anchor_train_id: int
    direction: str  # forward / backward
    ordered_test_ids: List[int]
    closing_train_id: Optional[int]
    prev_train_id: Optional[int]
    next_train_id: Optional[int]
    entry_inliers_forward: Optional[int]
    entry_inliers_backward: Optional[int]


class SFM_Sift:
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

        self._train_df: Optional[pd.DataFrame] = None
        self._train_cam_df: Optional[pd.DataFrame] = None
        self._test_cam_df: Optional[pd.DataFrame] = None

        self._train_ids: List[int] = []
        self._test_ids: List[int] = []
        self._all_ids: List[int] = []
        self._id_to_idx: Dict[int, int] = {}

        self._train_pos_map: Dict[int, np.ndarray] = {}
        self._train_cam_map: Dict[int, pd.Series] = {}
        self._test_cam_map: Dict[int, pd.Series] = {}

        # Pseudo-code requested structures.
        self._predictions: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._predictions_global: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._ground_truth: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self._train_mask: np.ndarray = np.zeros((0,), dtype=bool)
        self._test_ranges: List[Tuple[int, int]] = []

        self._model: Optional[MotionLinearModel] = None
        self._fallback_forward = np.array([0.0, 0.0], dtype=np.float64)
        self._fallback_backward = np.array([0.0, 0.0], dtype=np.float64)

        self._last_plan_df: Optional[pd.DataFrame] = None
        self._last_step_diag_df: Optional[pd.DataFrame] = None
        self._last_segment_diag_df: Optional[pd.DataFrame] = None
        self._calib_df: Optional[pd.DataFrame] = None
        self._coef_df: Optional[pd.DataFrame] = None

        self._img_cache: Dict[Tuple[int, Optional[int]], Tuple[np.ndarray, float]] = {}
        self._preproc_cache: Dict[Tuple[int, Optional[int], str], Tuple[np.ndarray, float]] = {}
        self._feature_cache: Dict[
            Tuple[int, Optional[int], str, Tuple],
            Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray, np.ndarray, float],
        ] = {}
        self._pair_motion_cache: Dict[Tuple[int, int, Tuple], PairMotion] = {}

    @property
    def plan_df(self) -> Optional[pd.DataFrame]:
        return self._last_plan_df

    @property
    def step_diag_df(self) -> Optional[pd.DataFrame]:
        return self._last_step_diag_df

    @property
    def segment_diag_df(self) -> Optional[pd.DataFrame]:
        return self._last_segment_diag_df

    @property
    def calib_df(self) -> Optional[pd.DataFrame]:
        return self._calib_df

    @property
    def coef_df(self) -> Optional[pd.DataFrame]:
        return self._coef_df

    @property
    def test_ranges(self) -> List[Tuple[int, int]]:
        return list(self._test_ranges)

    @property
    def train_ids(self) -> List[int]:
        return list(self._train_ids)

    @property
    def test_ids(self) -> List[int]:
        return list(self._test_ids)

    def _reset_caches(self) -> None:
        self._img_cache.clear()
        self._preproc_cache.clear()
        self._feature_cache.clear()
        self._pair_motion_cache.clear()

    def _require_dataset(self) -> None:
        if len(self._all_ids) == 0:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

    def _require_model(self) -> None:
        if self._model is None:
            raise RuntimeError("Calibration model missing. Call load_dataset() first.")

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

        train_df = train_cam_df.merge(train_pos_df, on="id", how="inner").copy()
        train_df["id"] = train_df["id"].astype(int)
        train_df = train_df.sort_values("id").reset_index(drop=True)

        train_cam_df = train_cam_df.copy()
        train_cam_df["id"] = train_cam_df["id"].astype(int)
        test_cam_df = test_cam_df.copy()
        test_cam_df["id"] = test_cam_df["id"].astype(int)

        self._train_df = train_df
        self._train_cam_df = train_cam_df
        self._test_cam_df = test_cam_df

        self._train_ids = sorted(train_df["id"].astype(int).tolist())
        self._test_ids = sorted(test_cam_df["id"].astype(int).tolist())
        self._all_ids = sorted(set(self._train_ids) | set(self._test_ids))
        self._id_to_idx = {iid: idx for idx, iid in enumerate(self._all_ids)}

        self._train_pos_map = {
            int(r["id"]): np.array([float(r["x_pixel"]), float(r["y_pixel"])], dtype=np.float64)
            for _, r in train_df.iterrows()
        }
        self._train_cam_map = {int(r["id"]): r for _, r in train_cam_df.iterrows()}
        self._test_cam_map = {int(r["id"]): r for _, r in test_cam_df.iterrows()}

        n = len(self._all_ids)
        self._train_mask = np.array([iid in self._train_pos_map for iid in self._all_ids], dtype=bool)
        self._ground_truth = np.zeros((n, 2), dtype=np.float64)
        for iid, xy in self._train_pos_map.items():
            self._ground_truth[self._id_to_idx[iid]] = xy
        self._predictions = np.full((n, 2), np.nan, dtype=np.float64)
        self._predictions_global = np.full((n, 2), np.nan, dtype=np.float64)

        self._test_ranges = self._build_ranges(self._test_ids)
        self._reset_caches()

        self._preprocess_data()

    def _preprocess_data(self) -> None:
        self._require_dataset()
        calib_pairs: List[Tuple[int, int]] = []

        for i in range(len(self._train_ids) - 1):
            a = int(self._train_ids[i])
            b = int(self._train_ids[i + 1])
            if b != a + 1:
                continue
            calib_pairs.append((a, b))
            if self.cfg.calib_use_bidirectional:
                calib_pairs.append((b, a))

        if self.cfg.calib_max_pairs is not None and len(calib_pairs) > int(self.cfg.calib_max_pairs):
            idx = np.linspace(0, len(calib_pairs) - 1, int(self.cfg.calib_max_pairs)).astype(int)
            calib_pairs = [calib_pairs[i] for i in idx]

        rows: List[Dict[str, object]] = []
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        w_list: List[float] = []

        for id0, id1 in calib_pairs:
            m = self._estimate_pair_motion(id0, id1)
            gt = self._train_pos_map[id1] - self._train_pos_map[id0]
            rows.append(
                {
                    "id0": id0,
                    "id1": id1,
                    "matches": m.matches,
                    "affine_inliers": m.affine_inliers,
                    "sfm_inliers": m.sfm_inliers,
                    "success": m.success,
                    "gt_dx": float(gt[0]),
                    "gt_dy": float(gt[1]),
                    "tx": m.tx,
                    "ty": m.ty,
                    "rot_deg": m.rot_deg,
                    "scale": m.scale,
                }
            )
            if m.success:
                x_list.append(self._motion_to_feature_vector(m))
                y_list.append(gt.astype(np.float64))
                w_list.append(max(1.0, float(m.affine_inliers)))

        self._calib_df = pd.DataFrame(rows)
        if len(x_list) == 0:
            raise RuntimeError("No usable calibration pairs. Cannot fit motion-to-map model.")

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
        self._model = MotionLinearModel(
            beta=beta,
            feature_names=feature_names,
            med_err=float(np.median(err)),
            mean_err=float(np.mean(err)),
            used_pairs=int(len(x)),
        )
        self._coef_df = pd.DataFrame(self._model.beta, index=feature_names, columns=["dx_coef", "dy_coef"])

        train_step_deltas = []
        for i in range(len(self._train_ids) - 1):
            a = int(self._train_ids[i])
            b = int(self._train_ids[i + 1])
            if b == a + 1:
                train_step_deltas.append(self._train_pos_map[b] - self._train_pos_map[a])
        if len(train_step_deltas) == 0:
            self._fallback_forward = np.array([0.0, 0.0], dtype=np.float64)
        else:
            self._fallback_forward = np.median(np.array(train_step_deltas, dtype=np.float64), axis=0)
        self._fallback_backward = -self._fallback_forward

    def _infer_source(self, image_id: int) -> str:
        if int(image_id) in self._train_cam_map:
            return "train"
        if int(image_id) in self._test_cam_map:
            return "test"
        raise KeyError(f"Image id {image_id} is unknown.")

    def _resolve_image_path(self, image_id: int) -> Path:
        source = self._infer_source(int(image_id))
        base_dir = self.train_images if source == "train" else self.test_images
        stems = [f"{int(image_id):04d}", str(int(image_id))]
        exts = [".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"]
        for st in stems:
            for ext in exts:
                p = base_dir / f"{st}{ext}"
                if p.exists():
                    return p
        raise FileNotFoundError(f"Image not found for id={image_id} in {base_dir}")

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

    def get_image(self, image_id: int, max_side: Optional[int] = None) -> np.ndarray:
        rgb, _ = self._load_image_rgb(int(image_id), max_side=max_side)
        return rgb

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

    def _get_cam_row(self, image_id: int) -> pd.Series:
        iid = int(image_id)
        if iid in self._train_cam_map:
            return self._train_cam_map[iid]
        if iid in self._test_cam_map:
            return self._test_cam_map[iid]
        raise KeyError(f"No camera row for id={image_id}")

    def _scaled_k(self, image_id: int, image_scale: float) -> np.ndarray:
        row = self._get_cam_row(int(image_id))
        fx = float(row["fx"]) * float(image_scale)
        fy = float(row["fy"]) * float(image_scale)
        cx = float(row["cx"]) * float(image_scale)
        cy = float(row["cy"]) * float(image_scale)
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
        self._require_model()
        x = self._motion_to_feature_vector(m)
        xb = np.concatenate([x, np.array([1.0], dtype=np.float64)], axis=0)
        y = xb @ self._model.beta
        return y.astype(np.float64)

    def _choose_fallback_delta(self, direction: str) -> np.ndarray:
        if str(direction) == "backward":
            return self._fallback_backward.copy()
        return self._fallback_forward.copy()

    def _is_train(self, image_id: int) -> bool:
        idx = self._id_to_idx[int(image_id)]
        return bool(self._train_mask[idx])

    def _has_prediction(self, image_id: int) -> bool:
        idx = self._id_to_idx[int(image_id)]
        return bool(np.isfinite(self._predictions[idx]).all())

    def _get_prediction(self, image_id: int) -> Optional[np.ndarray]:
        idx = self._id_to_idx[int(image_id)]
        p = self._predictions[idx]
        if np.isfinite(p).all():
            return p.copy()
        return None

    def _set_prediction(self, image_id: int, xy: np.ndarray) -> None:
        idx = self._id_to_idx[int(image_id)]
        self._predictions[idx] = np.asarray(xy, dtype=np.float64).reshape(2)

    def _get_global_prediction(self, image_id: int) -> Optional[np.ndarray]:
        idx = self._id_to_idx[int(image_id)]
        p = self._predictions_global[idx]
        if np.isfinite(p).all():
            return p.copy()
        return None

    def _set_global_prediction(self, image_id: int, xy: np.ndarray) -> None:
        idx = self._id_to_idx[int(image_id)]
        self._predictions_global[idx] = np.asarray(xy, dtype=np.float64).reshape(2)

    def _get_ground_truth(self, image_id: int) -> Optional[np.ndarray]:
        if int(image_id) not in self._train_pos_map:
            return None
        return self._train_pos_map[int(image_id)].copy()

    def _get_known_position(self, image_id: int) -> Optional[np.ndarray]:
        p = self._get_prediction(int(image_id))
        if p is not None:
            return p
        return self._get_ground_truth(int(image_id))

    def _predict_step(
        self,
        source_id: int,
        target_id: int,
        direction: str,
        last_good_delta: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, str, np.ndarray, PairMotion]:
        source_pos = self._get_known_position(int(source_id))
        if source_pos is None:
            raise AssertionError(
                f"source_id={source_id} has no known pose. Need train ground truth or previous prediction."
            )

        motion = self._estimate_pair_motion(int(source_id), int(target_id))
        if motion.success:
            dxy = self._predict_delta(motion)
            mode = "model"
        elif last_good_delta is not None:
            dxy = last_good_delta.copy()
            mode = "last_good"
        else:
            dxy = self._choose_fallback_delta(direction)
            mode = "fallback_median"

        pred_xy = source_pos + dxy
        self._set_prediction(int(target_id), pred_xy)
        return pred_xy, mode, dxy, motion

    def predict_frame(self, image_id: int, anchor_id: int) -> np.ndarray:
        self._require_dataset()
        self._require_model()
        iid = int(image_id)
        aid = int(anchor_id)
        if iid not in self._id_to_idx:
            raise KeyError(f"id={iid} not in dataset.")
        if aid not in self._id_to_idx:
            raise KeyError(f"anchor_id={aid} not in dataset.")
        if self._get_known_position(aid) is None:
            raise AssertionError(
                f"anchor_id={aid} has no known pose. Need train GT or previously predicted position."
            )
        direction = "forward" if iid >= aid else "backward"
        pred_xy, _mode, _dxy, _motion = self._predict_step(aid, iid, direction=direction, last_good_delta=None)
        return pred_xy

    @staticmethod
    def _build_ranges(ids: Sequence[int]) -> List[Tuple[int, int]]:
        if len(ids) == 0:
            return []
        out: List[Tuple[int, int]] = []
        s = int(ids[0])
        p = int(ids[0])
        for x in ids[1:]:
            xi = int(x)
            if xi == p + 1:
                p = xi
            else:
                out.append((s, p))
                s = xi
                p = xi
        out.append((s, p))
        return out

    def _ids_in_interval(self, start_id: int, end_id: int) -> List[int]:
        s = int(start_id)
        e = int(end_id)
        if s <= e:
            return [iid for iid in self._all_ids if s <= iid <= e]
        return [iid for iid in reversed(self._all_ids) if e <= iid <= s]

    def forward_range(self, start_id: int, end_id: int) -> pd.DataFrame:
        self._require_dataset()
        self._require_model()
        ids = self._ids_in_interval(int(start_id), int(end_id))
        if len(ids) < 2:
            raise ValueError("Range needs at least two known ids in dataset.")
        if self._get_known_position(ids[0]) is None:
            raise AssertionError(
                f"start_id={ids[0]} has no known pose. Need train GT or previous prediction."
            )

        direction = "forward" if int(end_id) >= int(start_id) else "backward"
        rows: List[Dict[str, object]] = []
        last_good_delta: Optional[np.ndarray] = None

        for source_id, target_id in zip(ids[:-1], ids[1:]):
            if not self._is_train(target_id):
                pred_xy, mode, dxy, motion = self._predict_step(
                    source_id,
                    target_id,
                    direction=direction,
                    last_good_delta=last_good_delta,
                )
                if mode == "model":
                    last_good_delta = dxy.copy()
            else:
                motion = self._estimate_pair_motion(source_id, target_id)
                pred_xy = self._train_pos_map[target_id].copy()
                mode = "skip_train"
                dxy = pred_xy - self._get_known_position(source_id)

            rows.append(
                {
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "target_is_train": bool(self._is_train(target_id)),
                    "direction": direction,
                    "predict_mode": mode,
                    "matches": int(motion.matches),
                    "affine_inliers": int(motion.affine_inliers),
                    "sfm_inliers": int(motion.sfm_inliers),
                    "pred_dx": float(dxy[0]),
                    "pred_dy": float(dxy[1]),
                    "pred_x": float(pred_xy[0]),
                    "pred_y": float(pred_xy[1]),
                }
            )

        return pd.DataFrame(rows)

    def forward_range_validation(self, start_id: int, end_id: int) -> pd.DataFrame:
        self._require_dataset()
        self._require_model()
        ids = self._ids_in_interval(int(start_id), int(end_id))
        if len(ids) < 2:
            raise ValueError("Range needs at least two known ids in dataset.")
        if self._get_known_position(ids[0]) is None:
            raise AssertionError(
                f"start_id={ids[0]} has no known pose. Need train GT or previous prediction."
            )

        direction = "forward" if int(end_id) >= int(start_id) else "backward"
        rows: List[Dict[str, object]] = []
        errors: List[float] = []
        last_good_delta: Optional[np.ndarray] = None

        for source_id, target_id in zip(ids[:-1], ids[1:]):
            pred_xy, mode, dxy, motion = self._predict_step(
                source_id,
                target_id,
                direction=direction,
                last_good_delta=last_good_delta,
            )
            if mode == "model":
                last_good_delta = dxy.copy()

            gt = self._get_ground_truth(target_id)
            err = float(np.linalg.norm(pred_xy - gt)) if gt is not None else np.nan
            if gt is not None and np.isfinite(err):
                errors.append(err)

            rows.append(
                {
                    "source_id": int(source_id),
                    "target_id": int(target_id),
                    "target_is_train": bool(self._is_train(target_id)),
                    "direction": direction,
                    "predict_mode": mode,
                    "matches": int(motion.matches),
                    "affine_inliers": int(motion.affine_inliers),
                    "sfm_inliers": int(motion.sfm_inliers),
                    "pred_dx": float(dxy[0]),
                    "pred_dy": float(dxy[1]),
                    "pred_x": float(pred_xy[0]),
                    "pred_y": float(pred_xy[1]),
                    "gt_x": float(gt[0]) if gt is not None else np.nan,
                    "gt_y": float(gt[1]) if gt is not None else np.nan,
                    "err_px": err,
                }
            )

        out = pd.DataFrame(rows)
        out["pred_x_model"] = out["pred_x"]
        out["pred_y_model"] = out["pred_y"]
        out["err_model_px"] = out["err_px"]

        # Notebook-16 style: derive global path from accumulated relative center positions.
        # For validation on train ranges this usually gives a more stable global trajectory.
        t_to_anchor: Dict[int, np.ndarray] = {int(ids[0]): np.eye(3, dtype=np.float64)}
        for prev_id, curr_id in zip(ids[:-1], ids[1:]):
            m = self._estimate_pair_motion(prev_id, curr_id)
            if (m.affine_M_1to0 is not None) and (m.affine_inliers >= int(self.cfg.seq_min_affine_inliers)):
                m_used = m.affine_M_1to0.astype(np.float64)
            else:
                m_used = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            h_curr_to_prev = self._affine2x3_to_hom(m_used)
            t_to_anchor[curr_id] = t_to_anchor[prev_id] @ h_curr_to_prev

        anchor_id = int(ids[0])
        anchor_gray, _ = self._load_preprocessed_gray(
            anchor_id,
            max_side=self.cfg.image_max_side,
            mode=self.cfg.preprocess_mode,
        )
        ah, aw = anchor_gray.shape[:2]
        anchor_center = np.array([aw * 0.5, ah * 0.5], dtype=np.float32)

        rel_by_id: Dict[int, np.ndarray] = {}
        for iid in ids:
            gray, _ = self._load_preprocessed_gray(
                iid,
                max_side=self.cfg.image_max_side,
                mode=self.cfg.preprocess_mode,
            )
            h, w = gray.shape[:2]
            center = np.array([[w * 0.5, h * 0.5]], dtype=np.float32)
            center_anchor = self._transform_points_hom(t_to_anchor[iid], center)[0]
            rel_by_id[iid] = (center_anchor - anchor_center).astype(np.float64)

        gt_by_id = {iid: self._get_ground_truth(iid) for iid in ids}
        known_by_id = {iid: self._get_known_position(iid) for iid in ids}

        anchor_pose = gt_by_id[anchor_id] if gt_by_id[anchor_id] is not None else known_by_id[anchor_id]
        ref_idx = None
        cal_source = "none"
        cal_scale = 1.0
        cal_angle = 0.0

        if anchor_pose is not None:
            if gt_by_id[anchor_id] is not None:
                cal_source = "gt"
                ref_poses = gt_by_id
            else:
                cal_source = "known"
                ref_poses = known_by_id

            for i in range(1, len(ids)):
                ref_id = int(ids[i])
                ref_pose = ref_poses.get(ref_id)
                if ref_pose is None:
                    continue
                rel = rel_by_id[ref_id]
                nr = float(np.linalg.norm(rel))
                ng = float(np.linalg.norm(ref_pose - anchor_pose))
                if nr > 1e-6 and ng > 1e-6:
                    cal_scale = ng / nr
                    cal_angle = float(
                        np.arctan2((ref_pose - anchor_pose)[1], (ref_pose - anchor_pose)[0]) - np.arctan2(rel[1], rel[0])
                    )
                    ref_idx = i
                    break

        if anchor_pose is not None:
            r_cal = self._build_rot(cal_angle)
            geom_pred_by_id = {
                iid: (anchor_pose + (r_cal @ rel_by_id[iid]) * cal_scale).astype(np.float64)
                for iid in ids
            }
        else:
            geom_pred_by_id = {}

        out["pred_x_geom"] = out["target_id"].map(lambda iid: float(geom_pred_by_id[iid][0]) if int(iid) in geom_pred_by_id else np.nan)
        out["pred_y_geom"] = out["target_id"].map(lambda iid: float(geom_pred_by_id[iid][1]) if int(iid) in geom_pred_by_id else np.nan)
        out["global_mode"] = "model"
        use_geom = np.isfinite(out["pred_x_geom"]) & np.isfinite(out["pred_y_geom"])
        out.loc[use_geom, "pred_x"] = out.loc[use_geom, "pred_x_geom"]
        out.loc[use_geom, "pred_y"] = out.loc[use_geom, "pred_y_geom"]
        out.loc[use_geom, "global_mode"] = "overlay_calibrated"
        out["err_px"] = np.sqrt((out["pred_x"] - out["gt_x"]) ** 2 + (out["pred_y"] - out["gt_y"]) ** 2)
        out.attrs["overlay_calibration"] = {
            "source": cal_source,
            "anchor_id": anchor_id,
            "ref_id": int(ids[ref_idx]) if ref_idx is not None else None,
            "scale": float(cal_scale),
            "yaw_deg": float(np.degrees(cal_angle)),
        }

        if len(errors) > 0:
            print(
                f"Validation range [{start_id}, {end_id}] | "
                f"train frames={len(errors)} | mean_err={np.nanmean(out['err_px']):.2f}px | "
                f"median_err={np.nanmedian(out['err_px']):.2f}px | "
                f"model_mean_err={np.nanmean(out['err_model_px']):.2f}px | "
                f"global_mode={'overlay_calibrated' if bool(np.any(use_geom)) else 'model'}"
            )
        else:
            print(f"Validation range [{start_id}, {end_id}] | no train frames in interval.")
        return out

    def _make_segment_plan(self, segment_idx: int, start_id: int, end_id: int) -> SegmentPlan:
        prev_train = max([t for t in self._train_ids if t < int(start_id)], default=None)
        next_train = min([t for t in self._train_ids if t > int(end_id)], default=None)

        if prev_train is None:
            if next_train is None:
                raise RuntimeError(f"Segment {start_id}-{end_id} has no train anchor.")
            anchor = int(next_train)
            direction = "backward"
            ordered = list(range(int(end_id), int(start_id) - 1, -1))
            closing = prev_train
        else:
            anchor = int(prev_train)
            direction = "forward"
            ordered = list(range(int(start_id), int(end_id) + 1))
            closing = next_train

        inl_fwd = None
        inl_bwd = None
        if self.cfg.auto_switch_to_next_anchor and (prev_train is not None) and (next_train is not None):
            m_fwd = self._estimate_pair_motion(int(prev_train), int(start_id))
            m_bwd = self._estimate_pair_motion(int(next_train), int(end_id))
            inl_fwd = int(m_fwd.affine_inliers)
            inl_bwd = int(m_bwd.affine_inliers)
            if (
                inl_fwd < int(self.cfg.anchor_switch_min_inliers)
                and inl_bwd >= int(self.cfg.anchor_switch_min_inliers)
                and inl_bwd > inl_fwd
            ):
                anchor = int(next_train)
                direction = "backward"
                ordered = list(range(int(end_id), int(start_id) - 1, -1))
                closing = prev_train

        return SegmentPlan(
            segment_idx=int(segment_idx),
            start_id=int(start_id),
            end_id=int(end_id),
            length=int(end_id - start_id + 1),
            anchor_train_id=int(anchor),
            direction=str(direction),
            ordered_test_ids=[int(x) for x in ordered],
            closing_train_id=int(closing) if closing is not None else None,
            prev_train_id=int(prev_train) if prev_train is not None else None,
            next_train_id=int(next_train) if next_train is not None else None,
            entry_inliers_forward=inl_fwd,
            entry_inliers_backward=inl_bwd,
        )

    def _segment_geom_predictions(self, plan: SegmentPlan) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
        anchor_id = int(plan.anchor_train_id)
        anchor_pose = self._train_pos_map[anchor_id].copy()

        path_ids: List[int] = [anchor_id] + [int(x) for x in plan.ordered_test_ids]
        if len(path_ids) <= 1:
            return {}, {"source": "none", "anchor_id": anchor_id, "ref_id": None, "scale": 1.0, "yaw_deg": 0.0}

        t_to_anchor: Dict[int, np.ndarray] = {anchor_id: np.eye(3, dtype=np.float64)}
        for prev_id, curr_id in zip(path_ids[:-1], path_ids[1:]):
            m = self._estimate_pair_motion(int(prev_id), int(curr_id))
            if (m.affine_M_1to0 is not None) and (m.affine_inliers >= int(self.cfg.seq_min_affine_inliers)):
                m_used = m.affine_M_1to0.astype(np.float64)
            else:
                m_used = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            h_curr_to_prev = self._affine2x3_to_hom(m_used)
            t_to_anchor[curr_id] = t_to_anchor[prev_id] @ h_curr_to_prev

        close_id = int(plan.closing_train_id) if plan.closing_train_id is not None else None
        if close_id is not None and len(path_ids) > 0:
            last_id = int(path_ids[-1])
            m_close = self._estimate_pair_motion(last_id, close_id)
            if (m_close.affine_M_1to0 is not None) and (m_close.affine_inliers >= int(self.cfg.seq_min_affine_inliers)):
                h_close_to_last = self._affine2x3_to_hom(m_close.affine_M_1to0.astype(np.float64))
                t_to_anchor[close_id] = t_to_anchor[last_id] @ h_close_to_last

        anchor_gray, _ = self._load_preprocessed_gray(
            anchor_id,
            max_side=self.cfg.image_max_side,
            mode=self.cfg.preprocess_mode,
        )
        ah, aw = anchor_gray.shape[:2]
        anchor_center = np.array([aw * 0.5, ah * 0.5], dtype=np.float32)

        rel_by_id: Dict[int, np.ndarray] = {}
        rel_ids = list(path_ids)
        if close_id is not None and close_id in t_to_anchor:
            rel_ids.append(close_id)
        for iid in rel_ids:
            gray, _ = self._load_preprocessed_gray(
                int(iid),
                max_side=self.cfg.image_max_side,
                mode=self.cfg.preprocess_mode,
            )
            h, w = gray.shape[:2]
            center = np.array([[w * 0.5, h * 0.5]], dtype=np.float32)
            center_anchor = self._transform_points_hom(t_to_anchor[int(iid)], center)[0]
            rel_by_id[int(iid)] = (center_anchor - anchor_center).astype(np.float64)

        cal_scale = 1.0
        cal_angle = 0.0
        cal_source = "none"
        ref_id = None

        if close_id is not None and close_id in rel_by_id and close_id in self._train_pos_map:
            rel = rel_by_id[close_id]
            ref_pose = self._train_pos_map[close_id]
            nr = float(np.linalg.norm(rel))
            ng = float(np.linalg.norm(ref_pose - anchor_pose))
            if nr > 1e-6 and ng > 1e-6:
                cal_scale = ng / nr
                cal_angle = float(
                    np.arctan2((ref_pose - anchor_pose)[1], (ref_pose - anchor_pose)[0]) - np.arctan2(rel[1], rel[0])
                )
                cal_source = "closing_gt"
                ref_id = int(close_id)

        r_cal = self._build_rot(cal_angle)
        out: Dict[int, np.ndarray] = {}
        for tid in plan.ordered_test_ids:
            tid_i = int(tid)
            if tid_i not in rel_by_id:
                continue
            out[tid_i] = (anchor_pose + (r_cal @ rel_by_id[tid_i]) * cal_scale).astype(np.float64)

        return out, {
            "source": cal_source,
            "anchor_id": anchor_id,
            "ref_id": ref_id,
            "scale": float(cal_scale),
            "yaw_deg": float(np.degrees(cal_angle)),
        }

    def forward_test_set(self) -> pd.DataFrame:
        self._require_dataset()
        self._require_model()
        self._predictions_global[:] = np.nan

        segment_plans: List[SegmentPlan] = []
        for i, (start_id, end_id) in enumerate(self._test_ranges):
            segment_plans.append(self._make_segment_plan(i, start_id, end_id))

        plan_rows: List[Dict[str, object]] = []
        for p in segment_plans:
            plan_rows.append(
                {
                    "segment_idx": p.segment_idx,
                    "start_id": p.start_id,
                    "end_id": p.end_id,
                    "length": p.length,
                    "anchor_train_id": p.anchor_train_id,
                    "direction": p.direction,
                    "closing_train_id": p.closing_train_id,
                    "prev_train_id": p.prev_train_id,
                    "next_train_id": p.next_train_id,
                    "entry_inliers_forward": p.entry_inliers_forward,
                    "entry_inliers_backward": p.entry_inliers_backward,
                }
            )
        self._last_plan_df = pd.DataFrame(plan_rows)

        step_diag_rows: List[Dict[str, object]] = []
        segment_diag_rows: List[Dict[str, object]] = []

        for p in segment_plans:
            segment_pred_local: Dict[int, np.ndarray] = {}
            prev_id = int(p.anchor_train_id)
            last_good_delta: Optional[np.ndarray] = None

            for step_idx, tid in enumerate(p.ordered_test_ids):
                pred_xy, mode, dxy, motion = self._predict_step(
                    prev_id,
                    int(tid),
                    direction=p.direction,
                    last_good_delta=last_good_delta,
                )
                segment_pred_local[int(tid)] = pred_xy
                if mode == "model":
                    last_good_delta = dxy.copy()

                step_diag_rows.append(
                    {
                        "segment_idx": p.segment_idx,
                        "step_idx": int(step_idx),
                        "source_id": int(prev_id),
                        "target_id": int(tid),
                        "direction": p.direction,
                        "predict_mode": mode,
                        "matches": int(motion.matches),
                        "affine_inliers": int(motion.affine_inliers),
                        "sfm_inliers": int(motion.sfm_inliers),
                        "pred_dx": float(dxy[0]),
                        "pred_dy": float(dxy[1]),
                        "pred_x": float(pred_xy[0]),
                        "pred_y": float(pred_xy[1]),
                    }
                )
                prev_id = int(tid)

            closure_used = False
            closure_err = np.array([0.0, 0.0], dtype=np.float64)
            closure_inliers = 0

            if (p.closing_train_id is not None) and len(p.ordered_test_ids) > 0:
                last_tid = int(p.ordered_test_ids[-1])
                close_tid = int(p.closing_train_id)
                m_close = self._estimate_pair_motion(last_tid, close_tid)
                closure_inliers = int(m_close.affine_inliers)
                if m_close.success and (m_close.affine_inliers >= int(self.cfg.closure_min_inliers)):
                    delta_close = self._predict_delta(m_close)
                    est_close = segment_pred_local[last_tid] + delta_close
                    gt_close = self._train_pos_map[close_tid]
                    closure_err = (gt_close - est_close).astype(np.float64)
                    n = len(p.ordered_test_ids)
                    for j, tid in enumerate(p.ordered_test_ids):
                        alpha = float(j + 1) / float(max(1, n))
                        corrected = segment_pred_local[int(tid)] + alpha * closure_err
                        segment_pred_local[int(tid)] = corrected
                        self._set_prediction(int(tid), corrected)
                    closure_used = True

            segment_diag_rows.append(
                {
                    "segment_idx": p.segment_idx,
                    "start_id": p.start_id,
                    "end_id": p.end_id,
                    "direction": p.direction,
                    "anchor_train_id": p.anchor_train_id,
                    "closing_train_id": p.closing_train_id,
                    "closure_used": bool(closure_used),
                    "closure_inliers": int(closure_inliers),
                    "closure_err_x": float(closure_err[0]),
                    "closure_err_y": float(closure_err[1]),
                    "closure_err_norm": float(np.linalg.norm(closure_err)),
                }
            )

            geom_pred_by_id, geom_meta = self._segment_geom_predictions(p)
            for tid, gxy in geom_pred_by_id.items():
                self._set_global_prediction(int(tid), gxy)
            if len(segment_diag_rows) > 0:
                segment_diag_rows[-1]["geom_source"] = geom_meta.get("source")
                segment_diag_rows[-1]["geom_ref_id"] = geom_meta.get("ref_id")
                segment_diag_rows[-1]["geom_scale"] = geom_meta.get("scale")
                segment_diag_rows[-1]["geom_yaw_deg"] = geom_meta.get("yaw_deg")

        missing = [tid for tid in self._test_ids if not self._has_prediction(int(tid))]
        if len(missing) > 0:
            for tid in missing:
                prev_train = max([t for t in self._train_ids if t < int(tid)], default=min(self._train_ids))
                self._set_prediction(int(tid), self._train_pos_map[int(prev_train)].copy())

        self._last_step_diag_df = pd.DataFrame(step_diag_rows)
        self._last_segment_diag_df = pd.DataFrame(segment_diag_rows)
        return self._last_segment_diag_df

    def export_submission_csv(self, output_path: Optional[Path] = None, use_global: bool = True) -> pd.DataFrame:
        self._require_dataset()
        if output_path is None:
            output_path = Path("notebooks/submission.csv")
        output_path = Path(output_path)

        rows: List[Dict[str, object]] = []
        for tid in sorted(self._test_ids):
            pred = self._get_global_prediction(int(tid)) if bool(use_global) else None
            if pred is None:
                pred = self._get_prediction(int(tid))
            if pred is None:
                prev_train = max([t for t in self._train_ids if t < int(tid)], default=min(self._train_ids))
                pred = self._train_pos_map[int(prev_train)].copy()
                self._set_prediction(int(tid), pred)
            rows.append({"id": int(tid), "x_pixel": float(pred[0]), "y_pixel": float(pred[1])})

        submission = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        return submission

    def plot_all_positions_on_map(
        self,
        use_global: bool = True,
        show_labels: bool = False,
        test_color: str = "red",
        figsize: Tuple[float, float] = (14, 10),
    ) -> None:
        self._require_dataset()
        if self.map_path is None or not self.map_path.exists():
            raise FileNotFoundError("map_path missing or not found.")

        map_bgr = cv2.imread(str(self.map_path), cv2.IMREAD_COLOR)
        if map_bgr is None:
            raise RuntimeError(f"Cannot read map image: {self.map_path}")
        map_rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)

        train_xy = np.array([self._train_pos_map[iid] for iid in self._train_ids], dtype=np.float64)

        test_pts: List[np.ndarray] = []
        test_ids_valid: List[int] = []
        for tid in self._test_ids:
            p = self._get_global_prediction(int(tid)) if bool(use_global) else None
            if p is None:
                p = self._get_prediction(int(tid))
            if p is None:
                continue
            test_pts.append(p.astype(np.float64))
            test_ids_valid.append(int(tid))

        test_xy = np.vstack(test_pts) if len(test_pts) > 0 else np.zeros((0, 2), dtype=np.float64)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(map_rgb)
        ax.scatter(train_xy[:, 0], train_xy[:, 1], s=9, c="deepskyblue", alpha=0.45, label="train GT")

        if test_xy.shape[0] > 0:
            ax.plot(test_xy[:, 0], test_xy[:, 1], "-", color=test_color, linewidth=1.0, alpha=0.75)
            ax.scatter(test_xy[:, 0], test_xy[:, 1], s=18, c=test_color, alpha=0.9, label="test predicted")
            if bool(show_labels):
                for i, tid in enumerate(test_ids_valid):
                    ax.text(test_xy[i, 0] + 4.0, test_xy[i, 1] + 4.0, str(tid), color=test_color, fontsize=7)

        ax.set_title(f"All positions on map | test_mode={'global' if use_global else 'raw'}")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _affine2x3_to_hom(m_aff: np.ndarray) -> np.ndarray:
        h = np.eye(3, dtype=np.float64)
        h[:2, :] = m_aff
        return h

    @staticmethod
    def _transform_points_hom(h_mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts.copy()
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        p = np.hstack([pts.astype(np.float64), ones])
        q = (h_mat @ p.T).T
        return (q[:, :2] / np.clip(q[:, 2:3], 1e-12, None)).astype(np.float32)

    def _pose_for_plot_optional(self, image_id: int, validation: bool) -> Optional[np.ndarray]:
        pred = self._get_prediction(int(image_id))
        pred_global = self._get_global_prediction(int(image_id))
        gt = self._get_ground_truth(int(image_id))
        if validation:
            if pred_global is not None:
                return pred_global
            if pred is not None:
                return pred
            if gt is not None:
                return gt
            return None
        if gt is not None:
            return gt
        if pred_global is not None:
            return pred_global
        if pred is not None:
            return pred
        return None

    def _pose_for_plot(self, image_id: int, validation: bool) -> np.ndarray:
        p = self._pose_for_plot_optional(image_id, validation=validation)
        if p is None:
            raise AssertionError(f"id={image_id} has neither prediction nor ground truth.")
        return p

    @staticmethod
    def _build_rot(theta: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )

    def plot_range(self, start_id: int, end_id: int, validation: bool = False) -> None:
        self._require_dataset()
        ids = self._ids_in_interval(int(start_id), int(end_id))
        if len(ids) == 0:
            raise ValueError("No dataset ids inside range.")
        if len(ids) == 1:
            raise ValueError("Need at least two ids for merged range plotting.")

        seq_data: Dict[int, Dict[str, object]] = {}
        for iid in ids:
            gray, _scale = self._load_preprocessed_gray(
                iid,
                max_side=self.cfg.image_max_side,
                mode=self.cfg.preprocess_mode,
            )
            pose_xy = self._pose_for_plot_optional(iid, validation=validation)
            seq_data[iid] = {"gray": gray, "pose": pose_xy}

        t_to_anchor: Dict[int, np.ndarray] = {int(ids[0]): np.eye(3, dtype=np.float64)}
        for prev_id, curr_id in zip(ids[:-1], ids[1:]):
            m = self._estimate_pair_motion(prev_id, curr_id)
            if (m.affine_M_1to0 is not None) and (m.affine_inliers >= int(self.cfg.seq_min_affine_inliers)):
                m_used = m.affine_M_1to0.astype(np.float64)
            else:
                m_used = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            h_curr_to_prev = self._affine2x3_to_hom(m_used)
            t_to_anchor[curr_id] = t_to_anchor[prev_id] @ h_curr_to_prev

        all_corners = []
        for iid in ids:
            gray = seq_data[iid]["gray"]
            h, w = gray.shape[:2]
            corners = np.array(
                [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                dtype=np.float32,
            )
            all_corners.append(self._transform_points_hom(t_to_anchor[iid], corners))

        all_corners_np = np.vstack(all_corners)
        padding = 30
        min_xy = np.floor(all_corners_np.min(axis=0) - padding).astype(np.int32)
        max_xy = np.ceil(all_corners_np.max(axis=0) + padding).astype(np.int32)

        canvas_w = int(max_xy[0] - min_xy[0] + 1)
        canvas_h = int(max_xy[1] - min_xy[1] + 1)
        t_shift = np.array(
            [[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        t_to_canvas = {iid: t_shift @ t_to_anchor[iid] for iid in ids}

        colors = np.array(
            [
                [245, 245, 245],
                [228, 26, 28],
                [55, 126, 184],
                [77, 175, 74],
                [255, 127, 0],
                [166, 86, 40],
            ],
            dtype=np.float32,
        )
        overlay = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

        for i, iid in enumerate(ids):
            gray = seq_data[iid]["gray"]
            m_aff = t_to_canvas[iid][:2, :]
            warped = cv2.warpAffine(gray, m_aff, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=0)
            tint = colors[i % len(colors)] / 255.0
            layer = (warped.astype(np.float32) / 255.0)[..., None] * tint[None, None, :]
            mask = warped > 0
            alpha = 0.50 if i == 0 else 0.36
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * layer[mask]
        overlay = np.clip(overlay, 0.0, 1.0)

        # Build center tracks in anchor/canvas frame and calibrate to global map space.
        anchor_id = int(ids[0])
        anchor_gray = seq_data[anchor_id]["gray"]
        ah, aw = anchor_gray.shape[:2]
        anchor_center = np.array([aw * 0.5, ah * 0.5], dtype=np.float32)

        center_canvas_list: List[np.ndarray] = []
        rel_anchor_list: List[np.ndarray] = []
        pose_used_list: List[Optional[np.ndarray]] = []

        for iid in ids:
            gray = seq_data[iid]["gray"]
            h, w = gray.shape[:2]
            center = np.array([[w * 0.5, h * 0.5]], dtype=np.float32)
            center_anchor = self._transform_points_hom(t_to_anchor[iid], center)[0]
            center_canvas = self._transform_points_hom(t_to_canvas[iid], center)[0]

            rel = (center_anchor - anchor_center).astype(np.float64)
            pose_used = seq_data[iid]["pose"]

            center_canvas_list.append(center_canvas.astype(np.float64))
            rel_anchor_list.append(rel)
            pose_used_list.append(None if pose_used is None else np.asarray(pose_used, dtype=np.float64))

        gt_pose_list: List[Optional[np.ndarray]] = [self._get_ground_truth(iid) for iid in ids]
        anchor_pose = gt_pose_list[0] if gt_pose_list[0] is not None else pose_used_list[0]
        if anchor_pose is None:
            raise AssertionError(
                f"id={anchor_id} has no pose for global calibration. "
                "Run forward_range/forward_test_set first or use validation mode with train ids."
            )

        cal_scale = 1.0
        cal_angle = 0.0
        ref_idx = None
        cal_source = "gt" if gt_pose_list[0] is not None else "pose"
        ref_pose_list = gt_pose_list if gt_pose_list[0] is not None else pose_used_list
        for i in range(1, len(ids)):
            ref_pose = ref_pose_list[i]
            rel = rel_anchor_list[i]
            if ref_pose is None:
                continue
            nr = float(np.linalg.norm(rel))
            ng = float(np.linalg.norm(ref_pose - anchor_pose))
            if nr > 1e-6 and ng > 1e-6:
                cal_scale = ng / nr
                cal_angle = float(np.arctan2((ref_pose - anchor_pose)[1], (ref_pose - anchor_pose)[0]) - np.arctan2(rel[1], rel[0]))
                ref_idx = i
                break

        r_cal = self._build_rot(cal_angle)
        overlay_global = np.array(
            [anchor_pose + (r_cal @ rel) * cal_scale for rel in rel_anchor_list],
            dtype=np.float64,
        )

        pose_used_xy = np.array(
            [p if p is not None else np.array([np.nan, np.nan], dtype=np.float64) for p in pose_used_list],
            dtype=np.float64,
        )
        gt_xy = np.array(
            [
                self._train_pos_map[iid] if iid in self._train_pos_map else np.array([np.nan, np.nan], dtype=np.float64)
                for iid in ids
            ],
            dtype=np.float64,
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(overlay)
        axes[0].set_title(f"Merged range {ids[0]} -> {ids[-1]} (center anchors)")
        axes[0].axis("off")

        center_canvas_np = np.vstack(center_canvas_list).astype(np.float64)
        axes[0].plot(center_canvas_np[:, 0], center_canvas_np[:, 1], "-", color="white", linewidth=1.1, alpha=0.85)
        for i, iid in enumerate(ids):
            c = colors[i % len(colors)] / 255.0
            p = center_canvas_np[i]
            axes[0].scatter(p[0], p[1], s=115, c=[c], edgecolors="black", linewidths=0.8, zorder=3)
            axes[0].text(
                p[0] + 8.0,
                p[1] + 8.0,
                f"{i}:{iid}",
                color="white",
                fontsize=8,
                zorder=4,
            )

        axes[1].plot(
            overlay_global[:, 0],
            overlay_global[:, 1],
            "-o",
            color="crimson",
            markersize=4,
            linewidth=1.7,
            label="overlay-calibrated",
        )
        valid_used = np.isfinite(pose_used_xy[:, 0]) & np.isfinite(pose_used_xy[:, 1])
        if np.any(valid_used):
            axes[1].plot(
                pose_used_xy[valid_used, 0],
                pose_used_xy[valid_used, 1],
                "--o",
                color="orange",
                markersize=3.5,
                linewidth=1.2,
                alpha=0.85,
                label="pose used (raw)",
            )
        valid_gt = np.isfinite(gt_xy[:, 0]) & np.isfinite(gt_xy[:, 1])
        if np.any(valid_gt):
            axes[1].scatter(gt_xy[valid_gt, 0], gt_xy[valid_gt, 1], s=22, c="deepskyblue", alpha=0.7, label="train GT")
        for i, iid in enumerate(ids):
            axes[1].text(overlay_global[i, 0] + 6.0, overlay_global[i, 1] + 6.0, str(iid), color="crimson", fontsize=7)
            if valid_used[i]:
                axes[1].text(
                    pose_used_xy[i, 0] + 6.0,
                    pose_used_xy[i, 1] - 7.0,
                    str(iid),
                    color="darkorange",
                    fontsize=7,
                    alpha=0.95,
                )
            if valid_gt[i]:
                axes[1].text(
                    gt_xy[i, 0] - 7.0,
                    gt_xy[i, 1] + 10.0,
                    str(iid),
                    color="deepskyblue",
                    fontsize=7,
                    alpha=0.95,
                )
        cal_note = "ref=none"
        if ref_idx is not None:
            cal_note = f"{cal_source} ref={ids[ref_idx]} scale={cal_scale:.3f} yaw={np.degrees(cal_angle):.2f}deg"
        axes[1].set_title(f"Map trajectory ({cal_note})")
        axes[1].set_xlabel("x_pixel")
        axes[1].set_ylabel("y_pixel")
        axes[1].invert_yaxis()
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
