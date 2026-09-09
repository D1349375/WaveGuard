"""
train.py
Training pipeline for WaveGuard CoLight MARL Agent.
Trains on Taiwan Xinyi Corridor (or Benchmark network), logging:
- Loss and Multi-objective Reward curves
- Average delay, throughput, TTC safety events
- Periodic evaluation and model checkpointing
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from envs.waveguard_env import WaveGuardEnv
from agents.colight_agent import WaveGuardCoLightAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train WaveGuard CoLight MARL Agent")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    parser.add_argument("--steps-per-episode", type=int, default=300, help="Simulation steps per episode")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N episodes")
    parser.add_argument("--net", type=str, default="taiwan_xinyi", choices=["taiwan_xinyi", "benchmark"], help="Network to train on")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for logs and checkpoints")
    parser.add_argument("--seed-base", type=int, default=42,
                         help="Base seed for per-episode env resets (episode i uses seed-base+i). Vary this "
                              "across independent training runs to get a variance estimate across training "
                              "seeds -- see docs/rl_training_pitfalls_checklist.md item 2 for why a single "
                              "training seed's result shouldn't be treated as a stable conclusion.")
    return parser.parse_args()

def train():
    args = parse_args()
    out_dir = BASE_DIR / args.output_dir
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = BASE_DIR / "configs" / "default_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.net == "taiwan_xinyi":
        sumocfg = BASE_DIR / "sumo_nets" / "taiwan_xinyi" / "xinyi_corridor.sumocfg"
    else:
        sumocfg = BASE_DIR / "sumo_nets" / "benchmark" / "benchmark_grid.sumocfg"

    print(f"==================================================")
    print(f"Starting WaveGuard MARL Training")
    print(f"Network: {args.net} ({sumocfg})")
    print(f"Episodes: {args.episodes}, Steps/Episode: {args.steps_per_episode}")
    print(f"==================================================")

    env = WaveGuardEnv(config_path=config_path, sumocfg_path=sumocfg, gui=False)
    env.max_steps = args.steps_per_episode

    # Initialize agent
    obs, info = env.reset(seed=args.seed_base)
    tls_ids = env.tls_ids
    agent = WaveGuardCoLightAgent(
        tls_ids=tls_ids,
        grid_size=cfg["environment"]["grid_size"],
        scalar_dim=3,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["gat_heads"],
        action_dim=cfg["model"]["action_dim"],
        lr=cfg["model"]["learning_rate"],
        gamma=cfg["model"]["gamma"],
        epsilon_start=cfg["model"]["epsilon_start"],
        epsilon_min=cfg["model"]["epsilon_min"],
        batch_size=cfg["model"]["batch_size"],
        buffer_size=cfg["model"]["buffer_size"]
    )

    # Warm-up: prime the environment's reward normalizers (RunningMeanStd for
    # R_flow/R_safety/R_equity) with a random-action rollout before real
    # training starts. See docs/rl_training_pitfalls_checklist.md item 5: with
    # no warm-up, the normalizers' running mean/std start from an uninformed
    # prior and swing wildly during exactly the phase of training where the
    # agent is most sensitive to reward scale, making the effective reward
    # signal non-stationary on top of the usual RL non-stationarity.
    warmup_steps = cfg.get("reward_normalization", {}).get("warmup_steps", 300)
    print(f"Warm-up: priming reward normalizers with {warmup_steps} random-action steps...")
    warmup_obs, _ = env.reset(seed=args.seed_base)
    for _ in range(warmup_steps):
        random_actions = {tid: random.randint(0, cfg["model"]["action_dim"] - 1) for tid in tls_ids}
        warmup_obs, _, terminated, truncated, _ = env.step(random_actions)
        if terminated or truncated:
            warmup_obs, _ = env.reset(seed=args.seed_base)

    training_records = []
    best_eval_wait = float("inf")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed_base + ep)
        ep_rewards = []
        ep_losses = []

        for step in range(args.steps_per_episode):
            actions = agent.act(obs, explore=True)
            next_obs, rewards, terminated, truncated, step_info = env.step(actions)

            # Store experience
            agent.store_transition(obs, actions, rewards, next_obs, terminated or truncated)

            # Train optimization step
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            mean_r = float(np.mean(list(rewards.values())))
            ep_rewards.append(mean_r)

            obs = next_obs
            if terminated or truncated:
                break

        final_metrics = env.wrapper.get_aggregate_metrics()
        avg_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        avg_ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        record = {
            "episode": ep,
            "seed_base": args.seed_base,
            "mean_reward": round(avg_ep_reward, 4),
            "mean_loss": round(avg_ep_loss, 4),
            "epsilon": round(agent.epsilon, 4),
            "overall_avg_wait_sec": final_metrics["overall_avg_wait_sec"],
            "car_avg_wait_sec": final_metrics["car_avg_wait_sec"],
            "moto_avg_wait_sec": final_metrics["moto_avg_wait_sec"],
            "equity_wait_delta": final_metrics["equity_wait_delta"],
            "throughput_vehicles": final_metrics["throughput_vehicles"],
            "ttc_conflicts": final_metrics["ttc_conflicts"]
        }
        training_records.append(record)

        print(f"Episode {ep:02d}/{args.episodes:02d} | Reward: {avg_ep_reward:7.2f} | Loss: {avg_ep_loss:6.4f} | Eps: {agent.epsilon:5.3f} | Wait: {final_metrics['overall_avg_wait_sec']:5.1f}s | Throughput: {final_metrics['throughput_vehicles']:4d} | TTC: {final_metrics['ttc_conflicts']:3d}")

        # Periodic evaluation & Checkpoint
        if ep % args.eval_interval == 0 or ep == args.episodes:
            if final_metrics["overall_avg_wait_sec"] < best_eval_wait or ep == args.episodes:
                best_eval_wait = final_metrics["overall_avg_wait_sec"]
                best_ckpt = ckpt_dir / "best_colight_model.pt"
                agent.save(best_ckpt)

        # Decay epsilon once per completed episode for the *next* episode
        # (see WaveGuardCoLightAgent.decay_epsilon_for_episode / checklist item 1).
        agent.decay_epsilon_for_episode(ep, args.episodes)

    env.close()

    # Save training log to CSV
    df = pd.DataFrame(training_records)
    log_csv = out_dir / "training_log.csv"
    df.to_csv(log_csv, index=False)
    print(f"Training log saved to {log_csv}")

    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df["episode"], df["mean_reward"], marker='o', color='royalblue')
    plt.title("Mean Multi-Objective Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(df["episode"], df["mean_loss"], marker='s', color='crimson')
    plt.title("Q-Learning Bellman Loss")
    plt.xlabel("Episode")
    plt.ylabel("Huber Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(df["episode"], df["overall_avg_wait_sec"], marker='^', color='forestgreen', label='Overall Wait')
    plt.plot(df["episode"], df["moto_avg_wait_sec"], linestyle='--', color='darkorange', label='Moto Wait')
    plt.title("Average Vehicle Waiting Time (s)")
    plt.xlabel("Episode")
    plt.ylabel("Seconds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(df["episode"], df["throughput_vehicles"], marker='d', color='purple', label='Throughput')
    plt.title("Throughput (Completed Trips)")
    plt.xlabel("Episode")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_png = out_dir / "training_curves.png"
    plt.savefig(curve_png, dpi=300)
    plt.close()
    print(f"Training curve figure saved to {curve_png}")
    print("Training finished successfully!")

if __name__ == "__main__":
    train()
