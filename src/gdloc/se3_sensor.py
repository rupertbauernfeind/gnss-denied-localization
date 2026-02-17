from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from gdloc.se3_dataset import SE3Dataset, SE3FrameMeta


@dataclass(frozen=True)
class SensorSample:
    """Single streamed camera sample produced by :class:`SE3Sensor`.

    Attributes:
        recording_idx: Zero-based index of the current recording segment.
        step_idx: Zero-based timestep inside the current recording playback.
        frame_id: Global frame identifier from the dataset CSV files.
        split: Dataset split label, usually ``"train"`` or ``"test"``.
        image_rgb: Loaded RGB image as a NumPy array.
        meta: Original immutable frame metadata from :class:`SE3Dataset`.
    """

    recording_idx: int
    step_idx: int
    frame_id: int
    split: str
    image_rgb: np.ndarray
    meta: SE3FrameMeta


class SE3Sensor:
    """Virtual camera stream simulator for GNSS-denied localization experiments.

    The sensor groups frames into contiguous recording segments (based on frame-id
    gaps) and can optionally augment each recording so playback starts and/or ends
    with training frames.

    A new recording is created whenever the frame-id difference between two
    consecutive frames is greater than one.

    Args:
        data_root: Root directory of the SE3 dataset.
        always_start_recording_with_train: If ``True``, prepend a reversed warm-up
            sequence so each recording starts with train frames when possible.
        always_end_recording_with_train: If ``True``, append a reversed tail
            sequence so each recording ends with train frames when possible.
    """

    def __init__(
        self,
        data_root: Path | str,
        always_start_recording_with_train: bool = True,
        always_end_recording_with_train: bool = True,
    ) -> None:
        self.dataset = SE3Dataset(data_root)
        self.always_start_recording_with_train = always_start_recording_with_train
        self.always_end_recording_with_train = always_end_recording_with_train

        # Teile die Frames in Recordings auf, basierend auf ID-Gaps
        self.recordings: List[List[SE3FrameMeta]] = self._split_into_recordings()
        for rec_idx, rec in enumerate(self.recordings):
            print(f"Raw Recording {rec_idx}: {len(rec)} frames, frame_id range [{rec[0].frame_id}, {rec[-1].frame_id}]")
        
        if self.always_start_recording_with_train:
            self._augment_recordings_start_with_train()
        if self.always_end_recording_with_train:
            self._augment_recordings_end_with_train()

    def _split_into_recordings(self) -> List[List[SE3FrameMeta]]:
        """Split sorted dataset frames into contiguous recording blocks.

        Returns:
            A list of recording segments, where each segment is a list of
            :class:`SE3FrameMeta` with gapless frame ids.
        """

        recordings: List[List[SE3FrameMeta]] = []
        recordings.append([])
    
        for id in range(len(self.dataset)):
            # Get the frame metadata for the current ID
            metaframe = self.dataset.frames_by_idx[id]

            # Check if this is the first frame
            if id == 0:
                recordings[-1].append(metaframe)
                continue

            # Check if the current frame ID is more than 1 greater than the last frame ID in the current recording
            if (metaframe.frame_id - recordings[-1][-1].frame_id) > 1:
                recordings.append([])
            recordings[-1].append(metaframe)

        return recordings
    
    def _augment_recordings_start_with_train(self, n_train_frames: int = 2):
        """Prepend a reversed prefix so recordings can start from train context.

        For recordings that begin with test frames, this method finds the first
        train frame and adds a reversed prefix ending at that region. This emulates
        a short backward motion before normal forward playback.

        Args:
            n_train_frames: Number of train frames after the first train index to
                include in the reversed prefix.

        Raises:
            AssertionError: If a recording is unexpectedly small or has no train
                frames.
        """

        for rec_idx, rec in enumerate(self.recordings):
            assert len(rec) > 4, f"Kleiner Recording gefunden, was nicht erwartet wird. Recording {rec_idx} hat nur {len(rec)} Frames."
            # one liner to find the index of the first train frame in the recording
            first_train_idx = next((i for i, meta in enumerate(rec) if meta.split == "train"), None)

            assert first_train_idx is not None, "Recording ohne Train-Frames gefunden, was nicht erwartet wird."
            if first_train_idx == 0:
                print(f"Already starts with train, skipping augmentation for recording {rec_idx}.")
                continue  # already starts with train

            # Augmentiere den Recording-Start mit den vorherigen Frames bis zum ersten Train-Frame
            self.recordings[rec_idx] = list(reversed(rec[1:first_train_idx+n_train_frames])) + rec  # füge den ersten Train-Frame am Anfang hinzu
            print(f"Augmented Recording {rec_idx} with {first_train_idx+n_train_frames} frames at start. New length: {len(self.recordings[rec_idx])}. Frame ID range: [{self.recordings[rec_idx][0].frame_id}, {self.recordings[rec_idx][-1].frame_id}]")

            # for r in rec:
            #     print(r.frame_id)
    
    def _augment_recordings_end_with_train(self, n_train_frames: int = 1):
        """Append a reversed suffix so recordings can end with train context.

        For recordings ending with test frames, this method mirrors a short sequence
        around the last train frame and appends it to the recording tail.

        Args:
            n_train_frames: Number of frames before the last train frame to mirror
                into the appended suffix.

        Raises:
            AssertionError: If a recording is unexpectedly small or has no train
                frames.
        """

        for rec_idx, rec in enumerate(self.recordings):
            assert len(rec) > 4, f"Kleiner Recording gefunden, was nicht erwartet wird. Recording {rec_idx} hat nur {len(rec)} Frames."
            # one liner to find the index of the last train frame in the recording
            last_train_idx = next((i for i, meta in reversed(list(enumerate(rec))) if meta.split == "train"), None)
            print(f"Last local train idx: {last_train_idx} for recording with frame_id {rec[last_train_idx].frame_id}")

            assert last_train_idx is not None, f"Recording {rec_idx} ohne Train-Frames gefunden, was nicht erwartet wird."
            if last_train_idx == len(rec) - 1:
                print(f"Already ends with train, skipping augmentation for recording {rec_idx}.")
                continue  # already ends with train
            # Augmentiere den Recording-Ende mit den folgenden Frames ab dem letzten Train-Frame
            self.recordings[rec_idx] = rec + list(reversed(rec[last_train_idx-n_train_frames:-1]))  # füge den letzten Train-Frame am Ende hinzu
            print(f"Augmented Recording {rec_idx} with {n_train_frames} frames at end. New length: {len(self.recordings[rec_idx])}. Frame ID range: [{self.recordings[rec_idx][0].frame_id}, {self.recordings[rec_idx][-1].frame_id}]")
            # for r in rec:
            #     print(r.frame_id)
            
            
    def __iter__(self) -> Iterator[SensorSample]:
        """Iterate over all recordings and yield streamed sensor samples.

        Yields:
            ``SensorSample`` objects in playback order across all recordings.
        """

        for recording_idx in range(len(self.recordings)):
            print(f"Starting recording {recording_idx} with {len(self.recordings[recording_idx])} frames.")
            seq = self.recordings[recording_idx]
            for step_idx, meta in enumerate(seq):
                img, _ = self.dataset.get_by_frame_id(meta.frame_id)
                yield SensorSample(
                    recording_idx=recording_idx,
                    step_idx=step_idx,
                    frame_id=meta.frame_id,
                    split=meta.split,
                    image_rgb=img,
                    meta=meta,
                )
            print(f"Finished recording {recording_idx} with {len(seq)} frames.")
    
