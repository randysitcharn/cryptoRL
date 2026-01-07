#!/usr/bin/env python3
"""Quick evaluation script for best model on test dataset."""

from sb3_contrib import TQC
from src.training.env import CryptoTradingEnv

# Load best model
print("Loading best model...")
model = TQC.load("weights/checkpoints/best_model.zip")

# Create eval env on test dataset
print("Creating eval environment...")
env = CryptoTradingEnv(
    parquet_path="data/processed_data_eval.parquet",
    window_size=64,
    episode_length=None,
    random_start=False,
    reward_scaling=30.0,
    downside_coef=0,
    upside_coef=0,
)

# Run evaluation
print("Running evaluation...")
obs, info = env.reset()
total_reward = 0
step = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1

    if step % 500 == 0:
        nav = info["nav"]
        pos = info["position_pct"]
        print(f"Step {step}: NAV=${nav:,.2f}, Pos={pos:+.2f}")

    if terminated or truncated:
        break

# Final metrics
initial_nav = 10000
final_nav = info["nav"]
total_return = (final_nav / initial_nav - 1) * 100
trades = info["total_trades"]
commission = info["total_commission"]

print()
print("=" * 50)
print("EVALUATION RESULTS (Test Dataset)")
print("=" * 50)
print(f"  Steps:         {step:,}")
print(f"  Initial NAV:   $10,000.00")
print(f"  Final NAV:     ${final_nav:,.2f}")
print(f"  Total Return:  {total_return:+.2f}%")
print(f"  Total Trades:  {trades:,}")
print(f"  Commission:    ${commission:,.2f}")
print(f"  Total Reward:  {total_reward:.2f}")
print("=" * 50)
