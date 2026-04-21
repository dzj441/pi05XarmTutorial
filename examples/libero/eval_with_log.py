import collections
import contextlib
import csv
import dataclasses
import datetime
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    max_success_videos_per_task: int = 10  # Save at most this many success videos per task. Use -1 for no limit.
    txt_log_path: str = "data/libero/eval_summary.txt"  # Text summary log path
    fail_log_path: str = "data/libero/fail_episodes.txt"  # Failed episodes log path
    csv_log_path: str = None  # Optional per-episode CSV log path

    seed: int = 7  # Random seed (for reproducibility)


def _log_line(txt_file, message: str) -> None:
    logging.info(message)
    txt_file.write(message + "\n")
    txt_file.flush()


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    video_out_dir = pathlib.Path(args.video_out_path)
    success_video_dir = video_out_dir / "success"
    failure_video_dir = video_out_dir / "failure"
    txt_log_file = pathlib.Path(args.txt_log_path)
    fail_log_file = pathlib.Path(args.fail_log_path)
    csv_log_file = pathlib.Path(args.csv_log_path) if args.csv_log_path else None
    success_video_dir.mkdir(parents=True, exist_ok=True)
    failure_video_dir.mkdir(parents=True, exist_ok=True)
    txt_log_file.parent.mkdir(parents=True, exist_ok=True)
    fail_log_file.parent.mkdir(parents=True, exist_ok=True)
    if csv_log_file is not None:
        csv_log_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    csv_fields = [
        "timestamp",
        "task_suite",
        "task_id",
        "task_description",
        "episode_idx",
        "success",
        "steps_executed",
        "task_success_rate_so_far",
        "total_success_rate_so_far",
        "error",
    ]

    csv_context = (
        csv_log_file.open("w", encoding="utf-8", newline="")
        if csv_log_file is not None
        else contextlib.nullcontext(None)
    )
    with txt_log_file.open("w", encoding="utf-8") as txt_f, fail_log_file.open(
        "w", encoding="utf-8"
    ) as fail_f, csv_context as csv_f:
        writer = None
        if csv_f is not None:
            writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
            writer.writeheader()
            csv_f.flush()

        run_start = datetime.datetime.now().isoformat(timespec="seconds")
        _log_line(txt_f, f"=== LIBERO Eval Start: {run_start} ===")
        _log_line(
            txt_f,
            (
                f"task_suite={args.task_suite_name}, num_tasks={num_tasks_in_suite}, "
                f"trials_per_task={args.num_trials_per_task}, seed={args.seed}"
            ),
        )
        _log_line(txt_f, f"video_out_path={video_out_dir}")
        _log_line(txt_f, f"success_video_dir={success_video_dir}")
        _log_line(txt_f, f"failure_video_dir={failure_video_dir}")
        _log_line(txt_f, f"max_success_videos_per_task={args.max_success_videos_per_task}")
        _log_line(txt_f, f"txt_log_path={txt_log_file}")
        _log_line(txt_f, f"fail_log_path={fail_log_file}")
        if csv_log_file is not None:
            _log_line(txt_f, f"csv_log_path={csv_log_file}")
        else:
            _log_line(txt_f, "csv_log_path=disabled")

        # Start evaluation
        total_episodes, total_successes = 0, 0
        failed_episodes_by_task: dict[int, list[int]] = collections.defaultdict(list)
        failed_episode_details: list[dict[str, str]] = []
        saved_success_videos_by_task: dict[int, int] = collections.defaultdict(int)
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                episode_success = False
                episode_error = ""

                logging.info(f"Starting episode {task_episodes + 1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), (
                                f"We want to replan every {args.replan_steps} steps, but policy only predicts "
                                f"{len(action_chunk)} steps."
                            )
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            episode_success = True
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:  # noqa: PERF203
                        episode_error = str(e)
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                steps_executed = max(0, t - args.num_steps_wait)

                # Save a replay video of the episode
                suffix = "success" if episode_success else "failure"
                video_path = None
                should_save_video = False
                if episode_success:
                    if args.max_success_videos_per_task < 0:
                        should_save_video = True
                    elif saved_success_videos_by_task[task_id] < args.max_success_videos_per_task:
                        should_save_video = True
                    if should_save_video:
                        video_path = success_video_dir / f"rollout_task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"
                else:
                    # Save all failure videos.
                    should_save_video = True
                    video_path = failure_video_dir / f"rollout_task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"

                if should_save_video:
                    if replay_images:
                        imageio.mimwrite(
                            video_path,
                            [np.asarray(x) for x in replay_images],
                            fps=10,
                        )
                        if episode_success:
                            saved_success_videos_by_task[task_id] += 1
                    else:
                        logging.warning(
                            "No replay frames for task_id=%s episode_idx=%s, skip video write.",
                            task_id,
                            episode_idx,
                        )

                task_success_rate = _safe_rate(task_successes, task_episodes)
                total_success_rate = _safe_rate(total_successes, total_episodes)

                row = {
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "task_suite": args.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "episode_idx": episode_idx,
                    "success": int(episode_success),
                    "steps_executed": steps_executed,
                    "task_success_rate_so_far": f"{task_success_rate:.6f}",
                    "total_success_rate_so_far": f"{total_success_rate:.6f}",
                    "error": episode_error,
                }
                if writer is not None:
                    writer.writerow(row)
                    csv_f.flush()

                if not episode_success:
                    failed_episodes_by_task[task_id].append(episode_idx)
                    failed_episode_details.append(
                        {
                            "task_id": str(task_id),
                            "episode_idx": str(episode_idx),
                            "steps_executed": str(steps_executed),
                            "error": (episode_error or "-").replace("\n", " ").replace("\t", " ").strip(),
                            "task_description": task_description.replace("\n", " ").replace("\t", " ").strip(),
                            "video_path": str(video_path) if video_path is not None else "-",
                        }
                    )

                # Log current results
                _log_line(txt_f, f"Success: {episode_success}")
                _log_line(txt_f, f"# episodes completed so far: {total_episodes}")
                _log_line(txt_f, f"# successes: {total_successes} ({total_success_rate * 100:.1f}%)")

            task_success_rate = _safe_rate(task_successes, task_episodes)
            total_success_rate = _safe_rate(total_successes, total_episodes)
            _log_line(txt_f, f"Current task success rate: {task_success_rate}")
            _log_line(txt_f, f"Current total success rate: {total_success_rate}")

        final_success_rate = _safe_rate(total_successes, total_episodes)
        _log_line(txt_f, f"Total success rate: {final_success_rate}")
        _log_line(txt_f, f"Total episodes: {total_episodes}")
        _log_line(txt_f, f"=== LIBERO Eval End: {datetime.datetime.now().isoformat(timespec='seconds')} ===")

        fail_f.write(f"suite={args.task_suite_name}\n")
        fail_f.write(f"total_episodes={total_episodes}\n")
        fail_f.write(f"total_successes={total_successes}\n")
        fail_f.write(f"total_failures={total_episodes - total_successes}\n")
        fail_f.write(f"total_success_rate={final_success_rate:.6f}\n\n")
        fail_f.write("fail episodes by task:\n")
        if failed_episodes_by_task:
            for task_id in sorted(failed_episodes_by_task):
                fail_f.write(f"task_id={task_id}\tfailed_episodes={sorted(failed_episodes_by_task[task_id])}\n")
        else:
            fail_f.write("none\n")

        fail_f.write("\nfail episode details:\n")
        if failed_episode_details:
            for item in failed_episode_details:
                fail_f.write(
                    "task_id={task_id}\tepisode={episode_idx}\tsteps={steps_executed}\terror={error}\tvideo={video_path}\tdesc={task_description}\n".format(
                        **item
                    )
                )
        else:
            fail_f.write("none\n")
        fail_f.flush()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
