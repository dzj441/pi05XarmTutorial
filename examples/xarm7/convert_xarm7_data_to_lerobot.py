"""
Convert xArm7 demonstrations (dataset/data1/demonstration_*) to LeRobot format.

Expected per-demonstration files (based on dataset/data1/demonstration_1):
  - cam_0_rgb_video.mp4
  - cam_0_rgb_video.metadata  (python pickle dict with timestamps)
  - cam_1_rgb_video.mp4
  - cam_1_rgb_video.metadata  (python pickle dict with timestamps)
  - xarm_gripper_state.h5     (positions: [T], timestamps: [T])
  - xarm_cartesian_state.h5   (positions: [T,3], rotations: [T,3], timestamps: [T])
  - raw_lang.json             (e.g. ["Move the lemon into the red cup."])

Usage:
  uv run examples/xarm7/convert_xarm7_data_to_lerobot.py \
    --data-dir path/to/your/repo/dataset/data1 \
    --repo-id local/xarm7_data1
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def _to_seconds(ts: np.ndarray) -> np.ndarray:
    """Normalize timestamps to seconds."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size == 0:
        return ts
    # Camera metadata timestamps are usually in ms (~1e12), robot in s (~1e9).
    if np.nanmedian(ts) > 1e11:
        return ts / 1000.0
    return ts


def _nearest_indices(query_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp, return nearest target index."""
    q = np.asarray(query_ts, dtype=np.float64)
    t = np.asarray(target_ts, dtype=np.float64)
    if len(t) == 0:
        raise ValueError("target_ts is empty")
    idx = np.searchsorted(t, q, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    left = np.clip(idx - 1, 0, len(t) - 1)
    choose_left = np.abs(q - t[left]) <= np.abs(q - t[idx])
    idx = np.where(choose_left, left, idx)
    return idx.astype(np.int64)


def _read_pickled_metadata(path: Path) -> dict:
    data = pickle.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metadata type: {type(data)} ({path})")
    return data


def _read_selected_frames_rgb(video_path: Path, indices: np.ndarray) -> list[np.ndarray]:
    """Read selected frame indices from video (RGB, HWC uint8)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        cap.release()
        return []

    order = np.argsort(indices)
    sorted_idx = indices[order]
    max_needed = int(sorted_idx[-1])

    by_index: dict[int, np.ndarray] = {}
    frame_i = 0
    ptr = 0
    while frame_i <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        while ptr < len(sorted_idx) and sorted_idx[ptr] == frame_i:
            by_index[frame_i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ptr += 1
        frame_i += 1
        if ptr >= len(sorted_idx):
            break
    cap.release()

    # Reconstruct original order.
    out = []
    for idx in indices:
        if int(idx) not in by_index:
            raise RuntimeError(f"Could not read requested frame {idx} from {video_path}")
        out.append(by_index[int(idx)])
    return out


def _load_language(demo_dir: Path, default_task: str) -> str:
    p = demo_dir / "raw_lang.json"
    if not p.exists():
        return default_task
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]
        if isinstance(v, str):
            return v
    except Exception:
        pass
    return default_task


def _load_robot_series(
    demo_dir: Path,
    *,
    action_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      robot_ts_sec: [T]
      state:        [T,7] (xyz + rpy + gripper)
      actions:      [T,7]
                    - cartesian_pose: strict savedata absolute mode
                      state_t = qpos[t], action_t = qpos[t+1]
                    - delta_cartesian: action_t[:6] = pose_{t+1} - pose_t, action_t[6] = gripper_{t+1}
    """
    with h5py.File(demo_dir / "xarm_cartesian_state.h5", "r") as f_cart:
        cart_pos = np.asarray(f_cart["positions"][:], dtype=np.float32)  # [T,3]
        cart_rot = np.asarray(f_cart["rotations"][:], dtype=np.float32)  # [T,3]
        cart_ts = _to_seconds(np.asarray(f_cart["timestamps"][:], dtype=np.float64))
    with h5py.File(demo_dir / "xarm_gripper_state.h5", "r") as f_grip:
        grip = np.asarray(f_grip["positions"][:], dtype=np.float32)
        grip_ts = _to_seconds(np.asarray(f_grip["timestamps"][:], dtype=np.float64))

    # truncate cartesian and gripper streams to the same length by index (not timestamp NN alignment).
    min_len = min(len(cart_pos), len(cart_rot), len(grip), len(cart_ts), len(grip_ts))
    cart_pos = cart_pos[:min_len]
    cart_rot = cart_rot[:min_len]
    gripper_pos = grip[:min_len]
    robot_ts = cart_ts[:min_len]

    pose = np.concatenate([cart_pos.astype(np.float32), cart_rot.astype(np.float32)], axis=1)  # [min_len, 6]
    qpos = np.concatenate([pose, gripper_pos[:, None]], axis=1).astype(np.float32)  # [min_len, 7]

    if action_source == "cartesian_pose":
        # strict savedata absolute semantics:
        # qpos <- qpos[:-1], action <- qpos[1:], timestamps <- timestamps[:-1]
        state = qpos[:-1]
        actions = qpos[1:]
        robot_ts = robot_ts[:-1]
    elif action_source == "delta_cartesian":
        # Keep same sequence length/offset convention as the absolute mode.
        state = qpos[:-1]
        next_pose = pose[1:]
        curr_pose = pose[:-1]
        delta_pose = (next_pose - curr_pose).astype(np.float32)
        next_gripper = gripper_pos[1:]
        actions = np.concatenate([delta_pose, next_gripper[:, None]], axis=1).astype(np.float32)
        robot_ts = robot_ts[:-1]
    else:
        raise ValueError(f"Unknown action_source: {action_source}")

    return robot_ts, state, actions


def _validate_demo_files(demo_dir: Path) -> None:
    required = [
        "cam_0_rgb_video.mp4",
        "cam_0_rgb_video.metadata",
        "cam_1_rgb_video.mp4",
        "cam_1_rgb_video.metadata",
        "xarm_gripper_state.h5",
        "xarm_cartesian_state.h5",
    ]
    missing = [x for x in required if not (demo_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"{demo_dir}: missing {missing}")


def main(
    data_dir: str,
    *,
    repo_id: str = "local/xarm7_data1",
    robot_type: str = "xarm7",
    fps: int = 30,
    image_writer_processes: int = 0,
    image_writer_threads: int = 1,
    default_task: str = "perform the demonstration task",
    push_to_hub: bool = False,
    action_source: str = "cartesian_pose",  # cartesian_pose | delta_cartesian
) -> None:
    root = Path(data_dir)
    demo_dirs = sorted([p for p in root.glob("demonstration_*") if p.is_dir()])
    if not demo_dirs:
        raise ValueError(f"No demonstration_* directories found under: {root}")

    # Clean existing output dataset.
    output_path = HF_LEROBOT_HOME / repo_id
    HF_LEROBOT_HOME.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Use demonstration_1 to infer image shape.
    _validate_demo_files(demo_dirs[0])
    probe_cap = cv2.VideoCapture(str(demo_dirs[0] / "cam_0_rgb_video.mp4"))
    ok, probe_frame = probe_cap.read()
    probe_cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {demo_dirs[0] / 'cam_0_rgb_video.mp4'}")
    h, w = probe_frame.shape[:2]

    features = {
        # Use LIBERO-style key names so downstream config/transforms are easier to adapt.
        "image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    converted = 0
    skipped = 0
    for demo_dir in demo_dirs:
        try:
            _validate_demo_files(demo_dir)

            task = _load_language(demo_dir, default_task)
            robot_ts, state, actions = _load_robot_series(demo_dir, action_source=action_source)

            md0 = _read_pickled_metadata(demo_dir / "cam_0_rgb_video.metadata")
            md1 = _read_pickled_metadata(demo_dir / "cam_1_rgb_video.metadata")
            cam0_ts = _to_seconds(np.asarray(md0["timestamps"], dtype=np.float64))
            cam1_ts = _to_seconds(np.asarray(md1["timestamps"], dtype=np.float64))

            idx0 = _nearest_indices(robot_ts, cam0_ts)
            idx1 = _nearest_indices(robot_ts, cam1_ts)

            frames0 = _read_selected_frames_rgb(demo_dir / "cam_0_rgb_video.mp4", idx0)
            frames1 = _read_selected_frames_rgb(demo_dir / "cam_1_rgb_video.mp4", idx1)

            T = min(len(robot_ts), len(frames0), len(frames1), len(state), len(actions))
            if T <= 1:
                raise RuntimeError(f"{demo_dir}: too few aligned frames ({T})")

            for i in range(T):
                frame = {
                    "image": frames0[i],
                    "wrist_image": frames1[i],
                    "state": state[i],
                    "actions": actions[i],
                    "task": task,
                }
                dataset.add_frame(frame)

            dataset.save_episode()
            converted += 1
            print(f"[OK] {demo_dir.name}: T={T}, task={task}")
        except Exception as e:  # noqa: PERF203
            skipped += 1
            print(f"[SKIP] {demo_dir.name}: {e}")

    print(f"converted={converted}, skipped={skipped}, output={HF_LEROBOT_HOME / repo_id}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["xarm7", "real-robot", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
