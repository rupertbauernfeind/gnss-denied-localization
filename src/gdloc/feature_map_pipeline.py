from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .stream_dataloader import GNSSStreamDataLoader, StreamFrame


@dataclass
class MatchOutput:
    keypoints0: np.ndarray  # [N0,2]
    keypoints1: np.ndarray  # [N1,2]
    matches: np.ndarray  # [M,2] indices into keypoints0/keypoints1
    confidence: np.ndarray  # [M]


@dataclass
class PairMetrics:
    backend: str
    use_magsac: bool
    frame_id0: int
    frame_id1: int
    raw_matches: int
    inliers: int
    inlier_ratio: float
    match_time_ms: float
    geom_time_ms: float


@dataclass
class FrameMeta:
    frame_id: int
    split: str
    image_path: Path
    width: int
    height: int
    pose_xy: Optional[Tuple[float, float]]
    pose_source: str
    intrinsics_fx: float
    intrinsics_fy: float
    intrinsics_cx: float
    intrinsics_cy: float


class _UnionFind:
    def __init__(self) -> None:
        self.parent: List[int] = []
        self.rank: List[int] = []

    def make(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, a: int) -> int:
        p = self.parent[a]
        if p != a:
            self.parent[a] = self.find(p)
        return self.parent[a]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class InlierGlobalFeatureMap:
    """
    Incremental global feature map that only integrates inlier correspondences.
    Observations are keyed by quantized keypoint coordinates per frame.
    """

    def __init__(self, quantize_px: float = 2.0) -> None:
        if quantize_px <= 0.0:
            raise ValueError("quantize_px must be > 0")
        self.quantize_px = float(quantize_px)

        self.frames: Dict[int, FrameMeta] = {}
        self.pairs: List[PairMetrics] = []

        self._uf = _UnionFind()
        self._obs_to_node: Dict[Tuple[int, int, int], int] = {}
        self._node_to_obs: List[Tuple[int, float, float]] = []

    def add_frame(self, frame: StreamFrame) -> None:
        h, w = frame.image_rgb.shape[:2]
        pose_xy = frame.init_xy if frame.init_xy is not None else frame.gt_xy

        self.frames[frame.frame_id] = FrameMeta(
            frame_id=frame.frame_id,
            split=frame.split,
            image_path=Path(frame.image_path),
            width=int(w),
            height=int(h),
            pose_xy=pose_xy,
            pose_source=frame.pose_source,
            intrinsics_fx=float(frame.intrinsics.fx),
            intrinsics_fy=float(frame.intrinsics.fy),
            intrinsics_cx=float(frame.intrinsics.cx),
            intrinsics_cy=float(frame.intrinsics.cy),
        )

    def add_pair(
        self,
        backend: str,
        use_magsac: bool,
        frame_id0: int,
        frame_id1: int,
        match_output: MatchOutput,
        inlier_mask: np.ndarray,
        match_time_ms: float,
        geom_time_ms: float,
    ) -> None:
        raw_matches = int(match_output.matches.shape[0])
        inliers = int(np.count_nonzero(inlier_mask))
        ratio = float(inliers / raw_matches) if raw_matches > 0 else 0.0

        self.pairs.append(
            PairMetrics(
                backend=backend,
                use_magsac=bool(use_magsac),
                frame_id0=int(frame_id0),
                frame_id1=int(frame_id1),
                raw_matches=raw_matches,
                inliers=inliers,
                inlier_ratio=ratio,
                match_time_ms=float(match_time_ms),
                geom_time_ms=float(geom_time_ms),
            )
        )

        if raw_matches == 0 or inliers == 0:
            return

        idx = np.where(inlier_mask)[0]
        for midx in idx:
            m = match_output.matches[midx]
            p0 = match_output.keypoints0[int(m[0])]
            p1 = match_output.keypoints1[int(m[1])]

            n0 = self._node_for_obs(int(frame_id0), float(p0[0]), float(p0[1]))
            n1 = self._node_for_obs(int(frame_id1), float(p1[0]), float(p1[1]))
            self._uf.union(n0, n1)

    def _obs_key(self, frame_id: int, x: float, y: float) -> Tuple[int, int, int]:
        q = self.quantize_px
        qx = int(round(x / q))
        qy = int(round(y / q))
        return (int(frame_id), qx, qy)

    def _node_for_obs(self, frame_id: int, x: float, y: float) -> int:
        key = self._obs_key(frame_id, x, y)
        n = self._obs_to_node.get(key)
        if n is not None:
            return n

        n = self._uf.make()
        self._obs_to_node[key] = n
        self._node_to_obs.append((int(frame_id), float(x), float(y)))
        return n

    def iter_tracks(self, min_track_len: int = 2) -> Iterator[List[Tuple[int, float, float]]]:
        clusters: Dict[int, List[Tuple[int, float, float]]] = {}
        for n, obs in enumerate(self._node_to_obs):
            r = self._uf.find(n)
            clusters.setdefault(r, []).append(obs)

        for obs_list in clusters.values():
            # keep at most one observation per frame to stabilize downstream export
            seen: Dict[int, Tuple[int, float, float]] = {}
            for f_id, x, y in obs_list:
                if f_id not in seen:
                    seen[f_id] = (f_id, x, y)
            dedup = list(seen.values())
            if len(dedup) >= int(min_track_len):
                yield dedup

    def tracks_as_list(self, min_track_len: int = 2) -> List[List[Tuple[int, float, float]]]:
        return list(self.iter_tracks(min_track_len=min_track_len))

    def pair_metrics_df(self) -> pd.DataFrame:
        if not self.pairs:
            return pd.DataFrame(
                columns=[
                    "backend",
                    "use_magsac",
                    "frame_id0",
                    "frame_id1",
                    "raw_matches",
                    "inliers",
                    "inlier_ratio",
                    "match_time_ms",
                    "geom_time_ms",
                ]
            )
        return pd.DataFrame([p.__dict__ for p in self.pairs])

    def summary(self, min_track_len: int = 2) -> Dict[str, float]:
        df = self.pair_metrics_df()
        if df.empty:
            return {
                "num_frames": float(len(self.frames)),
                "num_pairs": 0.0,
                "mean_raw_matches": 0.0,
                "mean_inliers": 0.0,
                "mean_inlier_ratio": 0.0,
                "mean_match_time_ms": 0.0,
                "mean_geom_time_ms": 0.0,
                "num_tracks": 0.0,
            }

        return {
            "num_frames": float(len(self.frames)),
            "num_pairs": float(len(df)),
            "mean_raw_matches": float(df["raw_matches"].mean()),
            "mean_inliers": float(df["inliers"].mean()),
            "mean_inlier_ratio": float(df["inlier_ratio"].mean()),
            "mean_match_time_ms": float(df["match_time_ms"].mean()),
            "mean_geom_time_ms": float(df["geom_time_ms"].mean()),
            "num_tracks": float(len(self.tracks_as_list(min_track_len=min_track_len))),
        }


class MatchBackend(ABC):
    name: str

    @abstractmethod
    def available(self) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> MatchOutput:
        pass


class LoFTRBackend(MatchBackend):
    name = "loftr"

    def __init__(
        self,
        device: Optional[str] = None,
        max_size: int = 1280,
        min_conf: float = 0.2,
        pretrained: str = "outdoor",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = int(max_size)
        self.min_conf = float(min_conf)
        self.pretrained = str(pretrained)

        self._model = None

    def available(self) -> Tuple[bool, str]:
        try:
            from kornia.feature import LoFTR  # noqa: F401
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        from kornia.feature import LoFTR

        self._model = LoFTR(pretrained=self.pretrained).eval().to(self.device)

    def _to_tensor(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, float]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        scale = 1.0
        if max(h, w) > self.max_size:
            scale = self.max_size / float(max(h, w))
            nw = max(32, int(round(w * scale)))
            nh = max(32, int(round(h * scale)))
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

        t = torch.from_numpy(gray).float()[None, None] / 255.0
        return t.to(self.device), scale

    @torch.inference_mode()
    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> MatchOutput:
        self._lazy_init()
        t0, s0 = self._to_tensor(img0_rgb)
        t1, s1 = self._to_tensor(img1_rgb)

        out = self._model({"image0": t0, "image1": t1})

        if "keypoints0" not in out or out["keypoints0"].numel() == 0:
            return MatchOutput(
                keypoints0=np.zeros((0, 2), dtype=np.float32),
                keypoints1=np.zeros((0, 2), dtype=np.float32),
                matches=np.zeros((0, 2), dtype=np.int32),
                confidence=np.zeros((0,), dtype=np.float32),
            )

        k0 = out["keypoints0"].detach().cpu().numpy().astype(np.float32)
        k1 = out["keypoints1"].detach().cpu().numpy().astype(np.float32)
        conf = out.get("confidence", None)
        if conf is None:
            conf_np = np.ones((k0.shape[0],), dtype=np.float32)
        else:
            conf_np = conf.detach().cpu().numpy().astype(np.float32)

        keep = conf_np >= self.min_conf
        k0 = k0[keep]
        k1 = k1[keep]
        conf_np = conf_np[keep]

        if s0 != 1.0:
            k0 /= s0
        if s1 != 1.0:
            k1 /= s1

        n = k0.shape[0]
        matches = np.column_stack([np.arange(n), np.arange(n)]).astype(np.int32)
        return MatchOutput(keypoints0=k0, keypoints1=k1, matches=matches, confidence=conf_np)


class SuperPointLightGlueBackend(MatchBackend):
    name = "superpoint_lightglue"

    def __init__(
        self,
        device: Optional[str] = None,
        max_num_keypoints: int = 2048,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_num_keypoints = int(max_num_keypoints)

        self._extractor = None
        self._matcher = None
        self._rbd = None

    def available(self) -> Tuple[bool, str]:
        try:
            import lightglue  # noqa: F401
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def _lazy_init(self) -> None:
        if self._extractor is not None and self._matcher is not None:
            return

        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd

        self._extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        self._matcher = LightGlue(features="superpoint").eval().to(self.device)
        self._rbd = rbd

    def _to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        t = torch.from_numpy(gray).float()[None, None] / 255.0
        return t.to(self.device)

    @torch.inference_mode()
    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> MatchOutput:
        self._lazy_init()
        t0 = self._to_tensor(img0_rgb)
        t1 = self._to_tensor(img1_rgb)

        f0 = self._extractor.extract(t0)
        f1 = self._extractor.extract(t1)
        out = self._matcher({"image0": f0, "image1": f1})
        f0, f1, out = [self._rbd(x) for x in [f0, f1, out]]

        k0 = f0["keypoints"].detach().cpu().numpy().astype(np.float32)
        k1 = f1["keypoints"].detach().cpu().numpy().astype(np.float32)
        m = out["matches"].detach().cpu().numpy().astype(np.int32)

        if "scores" in out:
            conf = out["scores"].detach().cpu().numpy().astype(np.float32)
        else:
            conf = np.ones((m.shape[0],), dtype=np.float32)

        return MatchOutput(keypoints0=k0, keypoints1=k1, matches=m, confidence=conf)


class SuperPointSuperGlueBackend(MatchBackend):
    name = "superpoint_superglue"

    def __init__(
        self,
        device: Optional[str] = None,
        superglue_repo_path: Optional[Path] = None,
        superglue_weights: str = "outdoor",
        max_keypoints: int = 2048,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.superglue_repo_path = Path(superglue_repo_path) if superglue_repo_path else None
        self.superglue_weights = str(superglue_weights)
        self.max_keypoints = int(max_keypoints)

        self._matching = None

    def available(self) -> Tuple[bool, str]:
        import sys

        if self.superglue_repo_path is not None:
            p = str(self.superglue_repo_path.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            from models.matching import Matching  # noqa: F401
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def _lazy_init(self) -> None:
        if self._matching is not None:
            return

        import sys

        if self.superglue_repo_path is not None:
            p = str(self.superglue_repo_path.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        from models.matching import Matching

        cfg = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": self.max_keypoints,
            },
            "superglue": {
                "weights": self.superglue_weights,
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        }

        self._matching = Matching(cfg).eval().to(self.device)

    def _to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        t = torch.from_numpy(gray).float()[None, None] / 255.0
        return t.to(self.device)

    @torch.inference_mode()
    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> MatchOutput:
        self._lazy_init()

        t0 = self._to_tensor(img0_rgb)
        t1 = self._to_tensor(img1_rgb)

        pred = self._matching({"image0": t0, "image1": t1})

        k0 = pred["keypoints0"][0].detach().cpu().numpy().astype(np.float32)
        k1 = pred["keypoints1"][0].detach().cpu().numpy().astype(np.float32)
        matches0 = pred["matches0"][0].detach().cpu().numpy().astype(np.int32)
        scores0 = pred["matching_scores0"][0].detach().cpu().numpy().astype(np.float32)

        valid = matches0 > -1
        idx0 = np.where(valid)[0].astype(np.int32)
        idx1 = matches0[valid].astype(np.int32)
        matches = np.column_stack([idx0, idx1]).astype(np.int32)
        conf = scores0[valid].astype(np.float32)

        return MatchOutput(keypoints0=k0, keypoints1=k1, matches=matches, confidence=conf)


def magsac_homography_inliers(
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    matches: np.ndarray,
    reproj_thr: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 10000,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if matches.shape[0] < 8:
        return None, np.zeros((matches.shape[0],), dtype=bool)

    pts0 = keypoints0[matches[:, 0]].astype(np.float32)
    pts1 = keypoints1[matches[:, 1]].astype(np.float32)

    H, mask = cv2.findHomography(
        pts0,
        pts1,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=float(reproj_thr),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )

    if H is None or mask is None:
        return None, np.zeros((matches.shape[0],), dtype=bool)

    inlier_mask = mask.ravel().astype(bool)
    return H, inlier_mask


@dataclass
class BackendRunResult:
    backend: str
    use_magsac: bool
    available: bool
    reason: str
    summary: Dict[str, float]
    feature_map: Optional[InlierGlobalFeatureMap]


@dataclass
class ValidationPredictionResult:
    backend: str
    use_magsac: bool
    available: bool
    reason: str
    summary: Dict[str, float]
    predictions_df: pd.DataFrame


def _iter_frames_for_split(
    loader: GNSSStreamDataLoader,
    split: str,
    include_gt_in_val: bool = False,
) -> Iterator[StreamFrame]:
    if split == "train":
        return loader.stream_train(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
        )
    if split == "val":
        return loader.stream_val(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
            include_gt_in_output=include_gt_in_val,
        )
    if split == "test":
        return loader.stream_test(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
        )
    raise ValueError(f"Unknown split: {split}")


def build_feature_map_on_stream(
    loader: GNSSStreamDataLoader,
    backend: MatchBackend,
    split: str = "train",
    max_frames: Optional[int] = None,
    use_magsac: bool = True,
    magsac_reproj_thr: float = 3.0,
    quantize_px: float = 2.0,
    print_progress: bool = True,
) -> BackendRunResult:
    ok, reason = backend.available()
    if not ok:
        return BackendRunResult(
            backend=backend.name,
            use_magsac=use_magsac,
            available=False,
            reason=reason,
            summary={},
            feature_map=None,
        )

    fmap = InlierGlobalFeatureMap(quantize_px=quantize_px)

    prev: Optional[StreamFrame] = None
    n = 0
    for frame in _iter_frames_for_split(loader, split=split):
        fmap.add_frame(frame)

        if prev is not None:
            t0 = perf_counter()
            out = backend.match(prev.image_rgb, frame.image_rgb)
            t_match = (perf_counter() - t0) * 1000.0

            t1 = perf_counter()
            if use_magsac:
                _, inliers = magsac_homography_inliers(
                    out.keypoints0,
                    out.keypoints1,
                    out.matches,
                    reproj_thr=magsac_reproj_thr,
                )
            else:
                inliers = np.ones((out.matches.shape[0],), dtype=bool)
            t_geom = (perf_counter() - t1) * 1000.0

            fmap.add_pair(
                backend=backend.name,
                use_magsac=use_magsac,
                frame_id0=prev.frame_id,
                frame_id1=frame.frame_id,
                match_output=out,
                inlier_mask=inliers,
                match_time_ms=t_match,
                geom_time_ms=t_geom,
            )

        prev = frame
        n += 1
        if print_progress and n % 20 == 0:
            print(f"[{backend.name}] split={split} processed={n}")

        if max_frames is not None and n >= int(max_frames):
            break

    summary = fmap.summary(min_track_len=2)
    summary.update(
        {
            "backend": backend.name,
            "use_magsac": bool(use_magsac),
            "available": True,
        }
    )

    return BackendRunResult(
        backend=backend.name,
        use_magsac=use_magsac,
        available=True,
        reason="ok",
        summary=summary,
        feature_map=fmap,
    )


def run_train_val_feature_map(
    loader: GNSSStreamDataLoader,
    backend: MatchBackend,
    max_train_frames: Optional[int] = None,
    max_val_frames: Optional[int] = None,
    use_magsac: bool = True,
    magsac_reproj_thr: float = 3.0,
    quantize_px: float = 2.0,
    print_progress: bool = True,
) -> BackendRunResult:
    ok, reason = backend.available()
    if not ok:
        return BackendRunResult(
            backend=backend.name,
            use_magsac=use_magsac,
            available=False,
            reason=reason,
            summary={},
            feature_map=None,
        )

    loader.reset_runtime_state()
    fmap = InlierGlobalFeatureMap(quantize_px=quantize_px)

    prev: Optional[StreamFrame] = None

    def _consume(frames: Iterable[StreamFrame], max_frames: Optional[int], split: str) -> None:
        nonlocal prev
        for i, frame in enumerate(frames, 1):
            fmap.add_frame(frame)
            if prev is not None:
                t0 = perf_counter()
                out = backend.match(prev.image_rgb, frame.image_rgb)
                t_match = (perf_counter() - t0) * 1000.0

                t1 = perf_counter()
                if use_magsac:
                    _, inliers = magsac_homography_inliers(
                        out.keypoints0,
                        out.keypoints1,
                        out.matches,
                        reproj_thr=magsac_reproj_thr,
                    )
                else:
                    inliers = np.ones((out.matches.shape[0],), dtype=bool)
                t_geom = (perf_counter() - t1) * 1000.0

                fmap.add_pair(
                    backend=backend.name,
                    use_magsac=use_magsac,
                    frame_id0=prev.frame_id,
                    frame_id1=frame.frame_id,
                    match_output=out,
                    inlier_mask=inliers,
                    match_time_ms=t_match,
                    geom_time_ms=t_geom,
                )

            prev = frame
            if print_progress and i % 20 == 0:
                print(f"[{backend.name}] split={split} processed={i}")

            if max_frames is not None and i >= int(max_frames):
                break

    _consume(
        loader.stream_train(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
        ),
        max_frames=max_train_frames,
        split="train",
    )

    _consume(
        loader.stream_val(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
            include_gt_in_output=False,
        ),
        max_frames=max_val_frames,
        split="val",
    )

    summary = fmap.summary(min_track_len=2)
    summary.update(
        {
            "backend": backend.name,
            "use_magsac": bool(use_magsac),
            "available": True,
        }
    )

    return BackendRunResult(
        backend=backend.name,
        use_magsac=use_magsac,
        available=True,
        reason="ok",
        summary=summary,
        feature_map=fmap,
    )


def compare_backends(
    loader: GNSSStreamDataLoader,
    backends: Sequence[MatchBackend],
    split: str = "train",
    max_frames: Optional[int] = None,
    magsac_modes: Sequence[bool] = (True,),
    magsac_reproj_thr: float = 3.0,
    quantize_px: float = 2.0,
    print_progress: bool = True,
) -> Tuple[pd.DataFrame, List[BackendRunResult]]:
    results: List[BackendRunResult] = []
    rows: List[Dict[str, object]] = []

    for backend in backends:
        for use_magsac in magsac_modes:
            run = build_feature_map_on_stream(
                loader=loader,
                backend=backend,
                split=split,
                max_frames=max_frames,
                use_magsac=use_magsac,
                magsac_reproj_thr=magsac_reproj_thr,
                quantize_px=quantize_px,
                print_progress=print_progress,
            )
            results.append(run)

            if run.available:
                row: Dict[str, object] = {
                    "backend": run.backend,
                    "use_magsac": run.use_magsac,
                    "available": run.available,
                    "reason": run.reason,
                }
                row.update(run.summary)
                rows.append(row)
            else:
                rows.append(
                    {
                        "backend": run.backend,
                        "use_magsac": run.use_magsac,
                        "available": run.available,
                        "reason": run.reason,
                    }
                )

    return pd.DataFrame(rows), results


def predict_validation_from_train_retrieval(
    loader: GNSSStreamDataLoader,
    backend: MatchBackend,
    max_train_frames: Optional[int] = None,
    max_val_frames: Optional[int] = None,
    top_k_train: int = 5,
    use_magsac: bool = True,
    magsac_reproj_thr: float = 3.0,
    min_match_score: int = 8,
    print_progress: bool = True,
) -> ValidationPredictionResult:
    """
    Predicts validation positions by retrieval against train frames with known GT poses.
    For each val frame:
    - match against train frames
    - score by inliers (MAGSAC) or raw matches
    - estimate pose as weighted average of top-k train poses
    """
    ok, reason = backend.available()
    if not ok:
        return ValidationPredictionResult(
            backend=backend.name,
            use_magsac=bool(use_magsac),
            available=False,
            reason=reason,
            summary={},
            predictions_df=pd.DataFrame(),
        )

    train_frames: List[StreamFrame] = []
    for i, fr in enumerate(
        loader.stream_train(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
        ),
        1,
    ):
        if fr.gt_xy is not None:
            train_frames.append(fr)
        if max_train_frames is not None and len(train_frames) >= int(max_train_frames):
            break
        if print_progress and i % 50 == 0:
            print(f"[{backend.name}] cached train frames: {len(train_frames)}")

    if len(train_frames) == 0:
        return ValidationPredictionResult(
            backend=backend.name,
            use_magsac=bool(use_magsac),
            available=True,
            reason="no_train_frames",
            summary={"val_count": 0.0, "mean_err_px": 0.0},
            predictions_df=pd.DataFrame(),
        )

    pred_by_id: Dict[int, Tuple[float, float]] = {}
    last_pred: Optional[Tuple[float, float]] = None
    rows: List[Dict[str, object]] = []

    for i, val_fr in enumerate(
        loader.stream_val(
            feature_extractor=lambda img: None,
            update_global_map=False,
            print_timing=False,
            include_gt_in_output=True,
        ),
        1,
    ):
        if max_val_frames is not None and i > int(max_val_frames):
            break

        candidates: List[Tuple[int, int, float, float]] = []
        total_match_time_ms = 0.0
        total_geom_time_ms = 0.0

        for tr_fr in train_frames:
            t0 = perf_counter()
            out = backend.match(val_fr.image_rgb, tr_fr.image_rgb)
            total_match_time_ms += (perf_counter() - t0) * 1000.0

            t1 = perf_counter()
            if use_magsac:
                _, inliers = magsac_homography_inliers(
                    out.keypoints0,
                    out.keypoints1,
                    out.matches,
                    reproj_thr=magsac_reproj_thr,
                )
                score = int(np.count_nonzero(inliers))
            else:
                score = int(out.matches.shape[0])
            total_geom_time_ms += (perf_counter() - t1) * 1000.0

            if score <= 0:
                continue

            tr_xy = tr_fr.gt_xy
            if tr_xy is None:
                continue
            candidates.append((score, int(tr_fr.frame_id), float(tr_xy[0]), float(tr_xy[1])))

        candidates.sort(key=lambda x: x[0], reverse=True)

        pred_xy: Optional[Tuple[float, float]] = None
        pose_source = "none"
        best_train_id = -1
        best_score = 0

        strong = [c for c in candidates if c[0] >= int(min_match_score)]
        top = strong[: int(max(1, top_k_train))]
        if len(top) > 0:
            weights = np.asarray([c[0] for c in top], dtype=np.float64)
            xs = np.asarray([c[2] for c in top], dtype=np.float64)
            ys = np.asarray([c[3] for c in top], dtype=np.float64)
            sw = float(weights.sum())
            if sw > 0.0:
                pred_xy = (float(np.dot(weights, xs) / sw), float(np.dot(weights, ys) / sw))
                pose_source = "train_retrieval"
                best_score = int(top[0][0])
                best_train_id = int(top[0][1])

        if pred_xy is None:
            prev = pred_by_id.get(int(val_fr.frame_id) - 1)
            if prev is not None:
                pred_xy = prev
                pose_source = "prev_id_fallback"
            elif last_pred is not None:
                pred_xy = last_pred
                pose_source = "last_fallback"

        if pred_xy is not None:
            pred_by_id[int(val_fr.frame_id)] = pred_xy
            last_pred = pred_xy

        gt_xy = val_fr.gt_xy
        if pred_xy is not None and gt_xy is not None:
            err_px = float(np.hypot(pred_xy[0] - float(gt_xy[0]), pred_xy[1] - float(gt_xy[1])))
        else:
            err_px = np.nan

        rows.append(
            {
                "backend": backend.name,
                "use_magsac": bool(use_magsac),
                "frame_id": int(val_fr.frame_id),
                "pred_x": np.nan if pred_xy is None else float(pred_xy[0]),
                "pred_y": np.nan if pred_xy is None else float(pred_xy[1]),
                "gt_x": np.nan if gt_xy is None else float(gt_xy[0]),
                "gt_y": np.nan if gt_xy is None else float(gt_xy[1]),
                "err_px": err_px,
                "pose_source": pose_source,
                "best_train_id": best_train_id,
                "best_score": best_score,
                "num_candidates": int(len(candidates)),
                "match_time_ms": float(total_match_time_ms),
                "geom_time_ms": float(total_geom_time_ms),
            }
        )

        if print_progress and i % 5 == 0:
            print(f"[{backend.name}] val processed={i} mean_err_so_far={np.nanmean([r['err_px'] for r in rows]):.2f}px")

    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        summary = {"val_count": 0.0, "mean_err_px": 0.0, "median_err_px": 0.0}
    else:
        valid = pred_df.dropna(subset=["err_px"])
        summary = {
            "val_count": float(len(pred_df)),
            "pred_available_count": float(pred_df["pred_x"].notna().sum()),
            "mean_err_px": float(valid["err_px"].mean()) if len(valid) else float("nan"),
            "median_err_px": float(valid["err_px"].median()) if len(valid) else float("nan"),
            "mean_best_score": float(pred_df["best_score"].mean()),
            "mean_candidates": float(pred_df["num_candidates"].mean()),
            "mean_match_time_ms": float(pred_df["match_time_ms"].mean()),
            "mean_geom_time_ms": float(pred_df["geom_time_ms"].mean()),
        }

    return ValidationPredictionResult(
        backend=backend.name,
        use_magsac=bool(use_magsac),
        available=True,
        reason="ok",
        summary=summary,
        predictions_df=pred_df,
    )


def compare_backends_validation_predictions(
    loader: GNSSStreamDataLoader,
    backends: Sequence[MatchBackend],
    max_train_frames: Optional[int] = None,
    max_val_frames: Optional[int] = None,
    top_k_train: int = 5,
    magsac_modes: Sequence[bool] = (True,),
    magsac_reproj_thr: float = 3.0,
    min_match_score: int = 8,
    print_progress: bool = True,
) -> Tuple[pd.DataFrame, List[ValidationPredictionResult]]:
    rows: List[Dict[str, object]] = []
    results: List[ValidationPredictionResult] = []

    for backend in backends:
        for use_magsac in magsac_modes:
            res = predict_validation_from_train_retrieval(
                loader=loader,
                backend=backend,
                max_train_frames=max_train_frames,
                max_val_frames=max_val_frames,
                top_k_train=top_k_train,
                use_magsac=use_magsac,
                magsac_reproj_thr=magsac_reproj_thr,
                min_match_score=min_match_score,
                print_progress=print_progress,
            )
            results.append(res)

            row: Dict[str, object] = {
                "backend": res.backend,
                "use_magsac": res.use_magsac,
                "available": res.available,
                "reason": res.reason,
            }
            row.update(res.summary)
            rows.append(row)

    return pd.DataFrame(rows), results


def plot_validation_maps_for_results(
    map_rgb: np.ndarray,
    results: Sequence[ValidationPredictionResult],
    cols: int = 2,
    line_alpha: float = 0.5,
    line_width: float = 1.0,
    figsize_per_subplot: Tuple[float, float] = (8.0, 6.0),
) -> None:
    n = len(results)
    if n == 0:
        print("No results to plot.")
        return

    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_subplot[0] * cols, figsize_per_subplot[1] * rows),
    )

    if isinstance(axes, np.ndarray):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = [axes]

    for ax, res in zip(ax_list, results):
        ax.imshow(map_rgb)
        ax.axis("off")

        title = f"{res.backend} | magsac={res.use_magsac}"
        if not res.available:
            ax.set_title(title + " | unavailable")
            ax.text(0.5, 0.5, res.reason, transform=ax.transAxes, ha="center", va="center", color="red")
            continue

        df = res.predictions_df.copy()
        if len(df) == 0:
            ax.set_title(title + " | no predictions")
            continue

        valid = df.dropna(subset=["pred_x", "pred_y", "gt_x", "gt_y"])
        if len(valid) == 0:
            ax.set_title(title + " | no valid pred/gt pairs")
            continue

        for r in valid.itertuples(index=False):
            ax.plot([r.pred_x, r.gt_x], [r.pred_y, r.gt_y], color="yellow", linewidth=line_width, alpha=line_alpha)

        ax.scatter(valid["pred_x"], valid["pred_y"], s=16, c="red", label="Pred")
        ax.scatter(valid["gt_x"], valid["gt_y"], s=16, c="lime", label="GT")

        mean_err = float(valid["err_px"].mean()) if "err_px" in valid else float("nan")
        ax.set_title(f"{title} | n={len(valid)} | mean_err={mean_err:.1f}px")
        ax.legend(loc="upper right", fontsize=8)

    for ax in ax_list[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pose_graph(
    fmap: InlierGlobalFeatureMap,
    min_edge_inliers: int = 20,
    title: str = "Camera Pose Graph",
) -> None:
    if not fmap.frames:
        print("No frames to plot.")
        return

    plt.figure(figsize=(12, 8))

    # Draw camera poses
    train_xy = []
    val_xy = []
    for meta in fmap.frames.values():
        if meta.pose_xy is None:
            continue
        if meta.split == "train":
            train_xy.append(meta.pose_xy)
        else:
            val_xy.append(meta.pose_xy)

    if train_xy:
        arr = np.asarray(train_xy, dtype=np.float32)
        plt.scatter(arr[:, 0], arr[:, 1], s=20, label="train poses")
    if val_xy:
        arr = np.asarray(val_xy, dtype=np.float32)
        plt.scatter(arr[:, 0], arr[:, 1], s=20, label="val/test poses")

    # Draw edges weighted by inliers
    for p in fmap.pairs:
        if p.inliers < int(min_edge_inliers):
            continue

        m0 = fmap.frames.get(p.frame_id0)
        m1 = fmap.frames.get(p.frame_id1)
        if m0 is None or m1 is None or m0.pose_xy is None or m1.pose_xy is None:
            continue

        x = [m0.pose_xy[0], m1.pose_xy[0]]
        y = [m0.pose_xy[1], m1.pose_xy[1]]
        alpha = min(0.9, 0.2 + (p.inliers / max(1.0, 200.0)))
        plt.plot(x, y, color="tab:gray", alpha=alpha, linewidth=1.0)

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.legend()
    plt.xlabel("x_pixel")
    plt.ylabel("y_pixel")
    plt.show()


def export_colmap_text_model(
    fmap: InlierGlobalFeatureMap,
    out_dir: Path,
    min_track_len: int = 2,
) -> Path:
    """
    Exports a COLMAP-compatible text model from the inlier map.
    Note: 3D points are synthetic (pose-graph based), intended for visualization/debug only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames_sorted = sorted(fmap.frames.values(), key=lambda x: x.frame_id)

    camera_ids: Dict[int, int] = {}
    image_ids: Dict[int, int] = {}

    # cameras.txt
    cam_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for idx, fr in enumerate(frames_sorted, 1):
        cam_id = idx
        camera_ids[fr.frame_id] = cam_id
        cam_lines.append(
            f"{cam_id} PINHOLE {fr.width} {fr.height} {fr.intrinsics_fx:.8f} {fr.intrinsics_fy:.8f} {fr.intrinsics_cx:.8f} {fr.intrinsics_cy:.8f}"
        )
    (out / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    tracks = fmap.tracks_as_list(min_track_len=min_track_len)

    # per-image point lists and point-index maps
    image_points: Dict[int, List[Tuple[float, float, int]]] = {fr.frame_id: [] for fr in frames_sorted}

    points3d_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
    point3d_id = 1

    # Build synthetic points from tracks
    for tr in tracks:
        # one obs per frame already ensured by map class
        obs = tr

        # create synthetic 3D at average camera-center location (z=0)
        xs = []
        ys = []
        for f_id, _, _ in obs:
            fr = fmap.frames.get(f_id)
            if fr is not None and fr.pose_xy is not None:
                xs.append(float(fr.pose_xy[0]))
                ys.append(float(fr.pose_xy[1]))

        if xs and ys:
            x3 = float(np.mean(xs))
            y3 = float(np.mean(ys))
        else:
            x3, y3 = 0.0, 0.0
        z3 = 0.0

        track_elems: List[str] = []
        for f_id, x2, y2 in obs:
            pts = image_points[f_id]
            p2d_idx = len(pts)
            pts.append((float(x2), float(y2), int(point3d_id)))
            # COLMAP track stores IMAGE_ID POINT2D_IDX, not frame_id
            # map frame_id -> image_id below
            track_elems.append(f"{f_id}:{p2d_idx}")

        points3d_lines.append(
            f"{point3d_id} {x3:.8f} {y3:.8f} {z3:.8f} 255 255 255 1.0 " + " ".join(track_elems)
        )
        point3d_id += 1

    # images.txt
    img_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as X Y POINT3D_ID"]
    for idx, fr in enumerate(frames_sorted, 1):
        image_ids[fr.frame_id] = idx

    # rewrite track references in points3D with image IDs
    rewritten_points = [points3d_lines[0]]
    for line in points3d_lines[1:]:
        parts = line.split()
        prefix = parts[:8]
        track_tokens = parts[8:]
        rewritten_track: List[str] = []
        for tk in track_tokens:
            f_id_s, pidx_s = tk.split(":")
            f_id = int(f_id_s)
            pidx = int(pidx_s)
            rewritten_track.extend([str(image_ids[f_id]), str(pidx)])
        rewritten_points.append(" ".join(prefix + rewritten_track))

    for fr in frames_sorted:
        img_id = image_ids[fr.frame_id]
        cam_id = camera_ids[fr.frame_id]

        # identity rotation; t = -C for R=I
        if fr.pose_xy is not None:
            cx, cy = fr.pose_xy
            tx, ty, tz = -float(cx), -float(cy), 0.0
        else:
            tx, ty, tz = 0.0, 0.0, 0.0

        img_lines.append(
            f"{img_id} 1 0 0 0 {tx:.8f} {ty:.8f} {tz:.8f} {cam_id} {fr.image_path.name}"
        )

        pts = image_points.get(fr.frame_id, [])
        if pts:
            pts_line = " ".join([f"{x:.3f} {y:.3f} {pid}" for x, y, pid in pts])
        else:
            pts_line = ""
        img_lines.append(pts_line)

    (out / "images.txt").write_text("\n".join(img_lines) + "\n")
    (out / "points3D.txt").write_text("\n".join(rewritten_points) + "\n")

    return out


def try_load_pycolmap_reconstruction(model_dir: Path):
    """Best-effort loader for pycolmap reconstruction object."""
    import pycolmap  # type: ignore

    p = Path(model_dir)

    # variant 1: constructor with path
    try:
        return pycolmap.Reconstruction(str(p))
    except Exception:
        pass

    # variant 2: empty + read
    recon = pycolmap.Reconstruction()
    if hasattr(recon, "read"):
        recon.read(str(p))
        return recon

    if hasattr(recon, "read_text"):
        recon.read_text(str(p))
        return recon

    raise RuntimeError("Could not load reconstruction with pycolmap API in this environment.")


__all__ = [
    "BackendRunResult",
    "InlierGlobalFeatureMap",
    "LoFTRBackend",
    "MatchBackend",
    "MatchOutput",
    "PairMetrics",
    "SuperPointLightGlueBackend",
    "SuperPointSuperGlueBackend",
    "ValidationPredictionResult",
    "build_feature_map_on_stream",
    "compare_backends_validation_predictions",
    "compare_backends",
    "export_colmap_text_model",
    "magsac_homography_inliers",
    "plot_validation_maps_for_results",
    "plot_pose_graph",
    "predict_validation_from_train_retrieval",
    "run_train_val_feature_map",
    "try_load_pycolmap_reconstruction",
]
