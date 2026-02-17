from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SE3DatasetPaths:
    root: Path
    train_images: Path
    test_images: Path
    train_cam_csv: Path
    test_cam_csv: Path
    train_pos_csv: Path
    map_path: Path

    @classmethod
    def from_data_root(cls, data_root: Path) -> "SE3DatasetPaths":
        root = Path(data_root)
        return cls(
            root=root,
            train_images=root / "train_data" / "train_images",
            test_images=root / "test_data" / "test_images",
            train_cam_csv=root / "train_data" / "train_cam.csv",
            test_cam_csv=root / "test_data" / "test_cam.csv",
            train_pos_csv=root / "train_data" / "train_pos.csv",
            map_path=root / "map.png",
        )

