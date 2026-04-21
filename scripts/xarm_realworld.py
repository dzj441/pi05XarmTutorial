#!/usr/bin/env python3
"""Run OpenPI policy directly on xArm real-world setup (no policy server needed)."""

from __future__ import annotations

import argparse
import pathlib

import time
from collections import deque

import cv2
import numpy as np

import sys
import os
sys.path.append(os.getcwd())


import openpi.transforms as _transforms
from examples.xarm7.video_streamer import VideoStreamer
from examples.xarm7.xarm import Xarm
from openpi.policies import policy_config
from openpi.training import config as training_config


def prepare_input(obs: dict, state: np.ndarray, prompt: str | list[str]) -> dict:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    return {
        "observation/state": np.asarray(state, dtype=np.float32),
        "observation/image": np.asarray(obs["third"], dtype=np.uint8),
        "observation/wrist_image": np.asarray(obs["wrist"], dtype=np.uint8),
        "prompt": prompt,
    }


class XarmEnv:
    def __init__(
        self,
        *,
        robot_ip: str,
        third_cam_ip: str,
        third_cam_port: int,
        wrist_cam_ip: str,
        wrist_cam_port: int,
    ):
        self.robot = Xarm(robot_ip)
        self.third_cam_receiver = VideoStreamer(third_cam_ip, third_cam_port)
        self.wrist_cam_receiver = VideoStreamer(wrist_cam_ip, wrist_cam_port)

    def step(self, action: list[float] | np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got shape={action.shape}")

        self.robot.move_coords(action[:6].tolist())
        gripper = 1 if float(action[6]) >= 0.5 else 0
        self.robot.move_gripper_percentage(gripper)

    def reset(self) -> None:
        self.robot.home()

    def get_obs(self) -> tuple[dict, np.ndarray]:
        wrist_img = self.wrist_cam_receiver.get_image_tensor()
        third_img = self.third_cam_receiver.get_image_tensor()

        # Keep the same resize convention as your previous script.
        wrist_img = cv2.resize(wrist_img, (320, 240))
        third_img = cv2.resize(third_img, (320, 240))

        configuration = np.asarray(self.robot.get_state(), dtype=np.float32)  # [x, y, z, roll, pitch, yaw]
        gripper_value = float(self.robot.get_gripper_state())  # 0 or 1
        state = np.concatenate([configuration, np.array([gripper_value], dtype=np.float32)], axis=0)

        obs = {
            "wrist": wrist_img,
            "third": third_img,
        }
        return obs, state


def run_eval(
    *,
    policy,
    deploy_env: XarmEnv,
    prompt: str,
    max_timesteps: int,
    query_frequency: int,
    rollouts: int,
    warmup_iters: int,
) -> None:
    if not prompt:
        raise ValueError("prompt must be non-empty")

    for rollout_id in range(rollouts):
        print(f"[Rollout {rollout_id + 1}/{rollouts}] reset robot")
        deploy_env.reset()
        action_queue: deque[np.ndarray] = deque()

        for t in range(max_timesteps):
            if t % query_frequency == 0:
                obs, state = deploy_env.get_obs()
                policy_input = prepare_input(obs, state, prompt)

                if t == 0 and rollout_id == 0 and warmup_iters > 0:
                    for _ in range(warmup_iters):
                        _ = policy.infer(policy_input)
                    print("network warmup done")

                outputs = policy.infer(policy_input)
                pred_actions = np.asarray(outputs["actions"])
                if pred_actions.ndim == 1:
                    pred_actions = pred_actions[None, :]

                n = min(query_frequency, pred_actions.shape[0])
                action_queue.extend(pred_actions[:n])
                print(f"[t={t}] infer {pred_actions.shape[0]} actions, queue+={n}")

            if not action_queue:
                continue

            action = action_queue.popleft()
            deploy_env.step(action.tolist())
            print(f"[t={t}] action={np.asarray(action).round(4).tolist()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_xarm7_finetune")
    p.add_argument("--checkpoint-dir", type=str ,default="/home/fvl/dzj/pi05_libero_jax_8gpu_realworld/4999", help="Path to one checkpoint step dir, e.g. .../1000")
    p.add_argument("--prompt", type=str ,default="put the lemon into the red cup.")
    p.add_argument("--robot-ip", default="192.168.1.200")
    p.add_argument("--third-cam-ip", default="192.168.1.201")
    p.add_argument("--third-cam-port", type=int, default=10005)
    p.add_argument("--wrist-cam-ip", default="192.168.1.201")
    p.add_argument("--wrist-cam-port", type=int, default=10006)
    p.add_argument("--max-timesteps", type=int, default=1000)
    p.add_argument("--query-frequency", type=int, default=10)
    p.add_argument("--rollouts", type=int, default=3)
    p.add_argument("--warmup-iters", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = training_config.get_config(args.config_name)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    # Keep this repack block to match your original call style.
    repack_transform = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/image": "observation/image",
                    "observation/wrist_image": "observation/wrist_image",
                    "observation/state": "observation/state",
                    "prompt": "prompt",
                }
            )
        ]
    )

    policy = policy_config.create_trained_policy(
        train_config=cfg,
        checkpoint_dir=checkpoint_dir,
        repack_transforms=repack_transform,
    )

    env = XarmEnv(
        robot_ip=args.robot_ip,
        third_cam_ip=args.third_cam_ip,
        third_cam_port=args.third_cam_port,
        wrist_cam_ip=args.wrist_cam_ip,
        wrist_cam_port=args.wrist_cam_port,
    )

    t0 = time.time()
    run_eval(
        policy=policy,
        deploy_env=env,
        prompt=args.prompt,
        max_timesteps=args.max_timesteps,
        query_frequency=args.query_frequency,
        rollouts=args.rollouts,
        warmup_iters=args.warmup_iters,
    )
    print(f"done, elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
