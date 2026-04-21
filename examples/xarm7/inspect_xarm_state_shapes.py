"""Inspect dataset shapes for xArm state HDF5 files in one demonstration directory.

Usage:
  uv run examples/xarm7/inspect_xarm_state_shapes.py
  uv run examples/xarm7/inspect_xarm_state_shapes.py --demo-dir dataset/data1/demonstration_1
"""

from __future__ import annotations

from pathlib import Path

import h5py
import tyro


def _print_h5_dataset_shapes(path: Path) -> None:
    print(f"\n=== {path} ===")
    if not path.exists():
        print("MISSING")
        return

    with h5py.File(path, "r") as f:
        found_any_dataset = False

        def _visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            nonlocal found_any_dataset
            if isinstance(obj, h5py.Dataset):
                found_any_dataset = True
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(_visitor)
        if not found_any_dataset:
            print("(no datasets found)")


def main(demo_dir: str = "dataset/data1/demonstration_1") -> None:
    demo_path = Path(demo_dir)
    targets = [
        "xarm_joint_state.h5",
        "xarm_gripper_state.h5",
        "xarm_cartesian_state.h5",
    ]

    print(f"Demo dir: {demo_path.resolve()}")
    for name in targets:
        _print_h5_dataset_shapes(demo_path / name)


if __name__ == "__main__":
    tyro.cli(main)
