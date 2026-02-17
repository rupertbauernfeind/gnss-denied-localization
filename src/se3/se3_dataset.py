from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from se3.utils.se3_dataset_paths import SE3DatasetPaths


@dataclass(frozen=True)
class SE3FrameMeta:
    idx: int
    frame_id: int
    split: str  # "train" | "test"
    image_path: Path
    fx: float
    fy: float
    cx: float
    cy: float
    gt_xy: Optional[Tuple[float, float]]  # nur train



class SE3Dataset:
    """
    Einfacher Zugriff auf SE3-Bilder + gespeicherte Kamera-Parameter + optionale GT-Position.
    
    Directory Structure:
        📁 data/
        ├── 📁 test_data/
        │   ├── 📁 test_images/      - Test image files
        │   └── 📄 test_cam.csv      - Camera parameters for test set
        ├── 📁 train_data/
        │   ├── 📁 train_images/     - Training image files
        │   ├── 📄 train_cam.csv     - Camera parameters for training set
        │   └── 📄 train_pos.csv     - Ground truth positions for training set
        └── 🗺️ map.png              - Reference map image
    """

    def __init__(self, data_root: Path):
        self.paths = SE3DatasetPaths.from_data_root(data_root)
        self.frames_by_idx: List[SE3FrameMeta] = [] # List of SE3FrameMeta, indexed by idx
        self.frames_by_frame_id: Dict[int, SE3FrameMeta] = {} # Dict of SE3FrameMeta, indexed by frame_id

        self.load_metadata()

    def __len__(self):
        return len(self.frames_by_idx)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, SE3FrameMeta]:
        meta = self.frames_by_idx[idx]
        return self.load_image_rgb(meta.frame_id), meta
    
    def get_by_frame_id(self, frame_id: int) -> Tuple[np.ndarray, SE3FrameMeta]:
        meta = self.frames_by_frame_id[frame_id]
        return self.load_image_rgb(meta.frame_id), meta
    
    def get_map_image(self) -> np.ndarray:
        img = cv2.imread(str(self.paths.map_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Map-Bild konnte nicht geladen werden: {self.paths.map_image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_image_rgb(self, frame_id: int) -> np.ndarray:
        meta = self.frames_by_frame_id[frame_id]
        bgr = cv2.imread(str(meta.image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Bild konnte nicht geladen werden: {meta.image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    def load_metadata(self):
        train_cam = pd.read_csv(self.paths.train_cam_csv)
        train_pos = pd.read_csv(self.paths.train_pos_csv)
        test_cam = pd.read_csv(self.paths.test_cam_csv)

        # check if frame_ids are unique and consistent across files
        for df in (train_cam, train_pos, test_cam):
            df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)

        train = train_cam.merge(train_pos[["id", "x_pixel", "y_pixel"]], on="id")
        test = test_cam.copy()
        
        print(f"Train frames: {len(train)}, Test frames: {len(test)}")

        rows: List[SE3FrameMeta] = []
        
        for r in train.itertuples(index=False):
            img = self._resolve_image_path(r.id, split="train")
            rows.append(
                SE3FrameMeta(
                    idx=len(rows),
                    frame_id=int(r.id),
                    split="train",
                    image_path=img,
                    fx=float(r.fx),
                    fy=float(r.fy),
                    cx=float(r.cx),
                    cy=float(r.cy),
                    gt_xy=(float(r.x_pixel), float(r.y_pixel)),
                )
            )

        for r in test.itertuples(index=False):
            img = self._resolve_image_path(r.id, split="test")
            rows.append(
                SE3FrameMeta(
                    idx=len(rows),
                    frame_id=int(r.id),
                    split="test",
                    image_path=img,
                    fx=float(r.fx),
                    fy=float(r.fy),
                    cx=float(r.cx),
                    cy=float(r.cy),
                    gt_xy=None,
                )
            )
        
        rows.sort(key=lambda x: x.frame_id)

        # idx neu setzen entsprechend der chronologischen Reihenfolge
        rows = [
            SE3FrameMeta(
                idx=i,
                frame_id=m.frame_id,
                split=m.split,
                image_path=m.image_path,
                fx=m.fx,
                fy=m.fy,
                cx=m.cx,
                cy=m.cy,
                gt_xy=m.gt_xy,
            )
            for i, m in enumerate(rows)
        ]

        # check for duplicate frame_ids
        duplicate_ids = [rid for rid, c in pd.Series([x.frame_id for x in rows]).value_counts().items() if c > 1]
        if duplicate_ids:
            raise ValueError(f"Doppelte IDs gefunden: {duplicate_ids[:20]}")

        self.frames_by_idx = rows
        self.frames_by_frame_id = {frame.frame_id: frame for frame in rows}


    def _resolve_image_path(self, frame_id: int, split: str) -> Path:
        base = self.paths.train_images if split == "train" else self.paths.test_images
        stems = (f"{int(frame_id):04d}", str(int(frame_id)))

        for stem in stems:
            for ext in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
                p = base / f"{stem}{ext}"
                if p.exists():
                    return p

        raise FileNotFoundError(f"Kein Bild für id={frame_id} in {base}")
    
    def plot_frame(self, idx: int):
        img, meta = self[idx]
        plt.imshow(img)
        plt.title(f"Frame ID: {meta.frame_id}, Split: {meta.split}, x: {meta.gt_xy[0]:.2f} | y: {meta.gt_xy[1]:.2f}" if meta.gt_xy else "No Ground Truth")
        plt.axis("off")
        plt.show()

    def plot_map(self):
        img = self.get_map_image()
        plt.imshow(img)
        plt.title("Map Image")
        plt.axis("off")
        plt.show()