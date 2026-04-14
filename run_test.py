# -*- coding: utf-8 -*-
"""
Evaluation entry point for the pretrained DDPG agent in CARLA.

Loads model weights from the ``model_weights/weights/`` directory, runs a configurable
number of test episodes, and outputs per-episode and aggregate statistics.

Results are saved to:
    <output_dir>/<run_tag>/eval_seed_<seed>.csv       — per-episode metrics
    <output_dir>/<run_tag>/eval_raw_seed_<seed>.npz   — per-step raw data
"""

import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

import argparse
import csv
import sys
import random
from datetime import datetime

import numpy as np

_np_major = int(np.__version__.split('.')[0])
if _np_major >= 2:
    print(
        f"[ERROR] NumPy {np.__version__} (>= 2.0) detected. "
        f"This is incompatible with the current OpenCV / PyTorch binaries.\n"
        f"Please run: pip install 'numpy<2'  (recommended: numpy==1.26.4)",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
import gc

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from env_carla.carla_env import CarlaEnv
from model import DDPGAgent


# =========================================================
# Utilities
# =========================================================
def set_seed(seed):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dirs(base, run_tag):
    """Return data_dir = base/run_tag/ and create it if needed."""
    data_dir = os.path.join(base, run_tag)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def generate_run_tag(seed):
    """Generate a unified identifier: seed_{k}_{MMDD_HHMM}."""
    ts = datetime.now().strftime('%m%d_%H%M')
    return f'seed_{seed}_{ts}'


# =========================================================
# Evaluation CSV header (matches training code exactly)
# =========================================================
EVAL_CSV_HEADER = [
    'episode',
    'success', 'collision', 'episode_reward',
    'average_speed', 'average_deviation', 'average_angle',
    'steer_variance', 'accel_variance',
    'route_completion', 'steps', 'lane_change_num',
    'steer_jerk', 'accel_jerk', 'steer_reversal_count',
]


# =========================================================
# Test
# =========================================================
def test(param, carla_env, agent):
    """Run evaluation episodes, save CSV / NPZ, and print results."""
    agent.load()
    seed = param.seed
    run_tag = param.run_tag
    data_dir = make_output_dirs(param.output_dir, run_tag)

    # ---- CSV writer ----
    eval_csv_path = os.path.join(data_dir, f'eval_seed_{seed}.csv')
    csv_file = open(eval_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=EVAL_CSV_HEADER)
    csv_writer.writeheader()

    # ---- Raw per-step collectors (flat concat + episode_id index) ----
    raw_ep_ids = []
    raw_steer = []
    raw_accel = []
    raw_speed = []
    raw_deviation = []
    raw_angle = []
    raw_reward = []

    results = []

    try:
        for i in range(param.test_iteration):
            agent.start_episode()

            all_speed, all_dev, all_angle = [], [], []
            all_steer, all_accel = [], []
            all_rewards = []

            state = carla_env.reset()
            step_count = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done = carla_env.step(action)

                all_speed.append(carla_env.forward_speed)
                all_dev.append(abs(carla_env.ego_deviation))
                all_angle.append(abs(carla_env.angle))
                all_steer.append(float(action[1]))
                all_accel.append(float(action[0]))
                all_rewards.append(reward)
                step_count += 1

                if done:
                    route_completion = min(carla_env._route_progress_ratio(), 1.0)
                    collision = 1 if carla_env.collision else 0
                    success = 0
                    if carla_env.destinations is not None:
                        if carla_env._check_waypoint(carla_env.destinations):
                            success = 1
                            route_completion = 1.0
                    break
                state = next_state

            # ---- Compute statistics ----
            steer_arr = np.array(all_steer, dtype=np.float32)
            accel_arr = np.array(all_accel, dtype=np.float32)

            steer_var = float(np.var(steer_arr))
            accel_var = float(np.var(accel_arr))
            steer_jerk = float(np.mean(np.abs(np.diff(steer_arr)))) if len(steer_arr) > 1 else 0.0
            accel_jerk = float(np.mean(np.abs(np.diff(accel_arr)))) if len(accel_arr) > 1 else 0.0
            steer_sign = np.sign(np.diff(steer_arr))
            steer_reversal = int(np.sum(np.abs(np.diff(steer_sign)) > 0)) if len(steer_sign) > 1 else 0

            # ---- Write CSV row ----
            row = {
                'episode': i,
                'success': success,
                'collision': collision,
                'episode_reward': float(np.sum(all_rewards)),
                'average_speed': float(np.mean(all_speed)),
                'average_deviation': float(np.mean(all_dev)),
                'average_angle': float(np.mean(all_angle)),
                'steer_variance': steer_var,
                'accel_variance': accel_var,
                'route_completion': route_completion,
                'steps': step_count,
                'lane_change_num': carla_env.lane_change_num,
                'steer_jerk': steer_jerk,
                'accel_jerk': accel_jerk,
                'steer_reversal_count': steer_reversal,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            results.append(row)

            # ---- Append to flat raw collectors ----
            n = len(all_steer)
            raw_ep_ids.extend([i] * n)
            raw_steer.extend(all_steer)
            raw_accel.extend(all_accel)
            raw_speed.extend(all_speed)
            raw_deviation.extend(all_dev)
            raw_angle.extend(all_angle)
            raw_reward.extend(all_rewards)

            # ---- Per-episode summary line ----
            print("── Test {:>3d}/{} done | {:>4d} steps | succ {} col {} | R {:>+7.2f} | spd {:.2f} | dev {:.3f} | route {:.1f}%".format(
                i + 1, param.test_iteration, step_count, success, collision,
                row['episode_reward'], row['average_speed'],
                row['average_deviation'], route_completion * 100))

    finally:
        csv_file.close()
        # ---- Save raw per-step NPZ (flat structure) ----
        eval_npz_path = os.path.join(data_dir, f'eval_raw_seed_{seed}.npz')
        np.savez_compressed(
            eval_npz_path,
            episode_id=np.array(raw_ep_ids, dtype=np.int32),
            steer=np.array(raw_steer, dtype=np.float32),
            accel=np.array(raw_accel, dtype=np.float32),
            speed=np.array(raw_speed, dtype=np.float32),
            deviation=np.array(raw_deviation, dtype=np.float32),
            angle=np.array(raw_angle, dtype=np.float32),
            reward=np.array(raw_reward, dtype=np.float32),
        )
        print(f"Results saved to {data_dir}")

    # ---- Aggregate statistics ----
    if results:
        n = len(results)
        total_success = sum(r['success'] for r in results)
        total_collision = sum(r['collision'] for r in results)

        print("\n========== Aggregate Results ({} episodes) ==========".format(n))
        print(f"  Success rate:       {total_success}/{n} ({100 * total_success / n:.1f}%)")
        print(f"  Collision rate:     {total_collision}/{n} ({100 * total_collision / n:.1f}%)")
        print(f"  Avg reward:         {np.mean([r['episode_reward'] for r in results]):.2f}")
        print(f"  Avg speed:          {np.mean([r['average_speed'] for r in results]):.2f} m/s")
        print(f"  Avg deviation:      {np.mean([r['average_deviation'] for r in results]):.3f} m")
        print(f"  Avg heading angle:  {np.mean([r['average_angle'] for r in results]):.2f} deg")
        print(f"  Avg route compl:    {np.mean([r['route_completion'] for r in results]) * 100:.1f}%")
        print(f"  Avg steer jerk:     {np.mean([r['steer_jerk'] for r in results]):.4f}")
        print(f"  Avg accel jerk:     {np.mean([r['accel_jerk'] for r in results]):.4f}")
        print(f"  Avg steer variance: {np.mean([r['steer_variance'] for r in results]):.4f}")
        print(f"  Avg accel variance: {np.mean([r['accel_variance'] for r in results]):.4f}")
        print(f"  Avg steer reversal: {np.mean([r['steer_reversal_count'] for r in results]):.1f}")
        print(f"  Avg lane changes:   {np.mean([r['lane_change_num'] for r in results]):.1f}")
        print("=====================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pretrained DDPG agent in CARLA.")

    # random seed & output
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output root directory. Data is saved to output_dir/seed_{k}_{MMDD_HHMM}/')

    # CARLA environment
    parser.add_argument('--map', type=str, default='Town04')
    parser.add_argument('--task_mode', type=str, default='random')
    parser.add_argument('--carla_port', type=int, default=2000)
    parser.add_argument('--synchronous_mode', type=bool, default=True)
    parser.add_argument('--no_rendering_mode', type=bool, default=False)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.05)

    # environment control
    parser.add_argument('--max_time_episode', type=int, default=2000)
    parser.add_argument('--number_of_vehicles', type=int, default=100)
    parser.add_argument('--max_speed', type=float, default=9.0)
    parser.add_argument('--min_speed', type=float, default=4.0)
    parser.add_argument('--max_angle', type=float, default=60.0)
    parser.add_argument('--min_angle', type=float, default=10.0)
    parser.add_argument('--out_lane', type=float, default=5.0)
    parser.add_argument('--max_lane', type=float, default=1.0)
    parser.add_argument('--max_past_step', type=int, default=1)
    parser.add_argument('--acc_range', type=float, default=0.3)
    parser.add_argument('--steer_range', type=float, default=0.4)
    parser.add_argument('--discrete', type=bool, default=False)
    parser.add_argument('--discrete_acc', type=list, default=[-0.2, 0.6])
    parser.add_argument('--discrete_steer', type=list, default=[-0.1, 0.0, 0.1])
    parser.add_argument('--area', type=list, default=[60.0, 20.0, 40, 40])

    # observation
    parser.add_argument('--img_size', nargs=2, type=int, default=[112, 112])
    parser.add_argument('--bev_size', nargs=2, type=int, default=[112, 112])
    parser.add_argument('--pixels_per_meter', type=int, default=3)
    parser.add_argument('--preview_distances', nargs='+', type=float, default=[1.0, 3.0, 5.0])

    # evaluation
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--test_iteration', type=int, default=50)
    parser.add_argument('--residual_alpha', type=float, default=0.8)

    # model weights
    parser.add_argument('--load_dir', type=str, default='./model_weights/weights/')
    parser.add_argument('--device', type=str, default='cuda:0')

    params = parser.parse_args()

    set_seed(params.seed)

    # ---- Generate unified run_tag (same format as training code) ----
    params.run_tag = generate_run_tag(params.seed)
    print(f"[Run Tag] {params.run_tag}")

    env = CarlaEnv(params)
    agent = DDPGAgent(params)
    try:
        test(params, env, agent)
    finally:
        env.reset_carla()
        env = None
        agent = None
        gc.collect()
        print("Done. CARLA client restored.")
