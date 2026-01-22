#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_model_actions.py - Diagnostic script for small action values.

Investigates why the trained model outputs very small actions (~0.02).
"""

import torch
import numpy as np
from sb3_contrib import TQC
from src.training.batch_env import BatchCryptoEnv

def main():
    model_path = "weights/wfo/segment_0/tqc.zip"
    data_path = "data/processed_data.parquet"

    print("=" * 60)
    print("MODEL ACTION DIAGNOSTIC")
    print("=" * 60)

    # Load model
    model = TQC.load(model_path, device="cpu")

    # 1. Entropy coefficient
    print("\n=== Entropy Coefficient ===")
    if hasattr(model, "ent_coef_tensor"):
        print(f"ent_coef_tensor: {model.ent_coef_tensor.item():.6f}")
    if hasattr(model, "log_ent_coef"):
        log_ent = model.log_ent_coef.item()
        ent_coef = np.exp(log_ent)
        print(f"log_ent_coef: {log_ent:.6f}")
        print(f"ent_coef (exp): {ent_coef:.6f}")

    # 2. Action space
    print("\n=== Action Space ===")
    print(f"Low: {model.action_space.low}")
    print(f"High: {model.action_space.high}")

    # 3. Actor network analysis
    print("\n=== Actor Network Parameters ===")
    actor = model.actor
    for name, param in actor.named_parameters():
        if any(k in name.lower() for k in ["mu", "log_std", "action", "last"]):
            print(f"{name}:")
            print(f"  shape: {param.shape}")
            print(f"  mean: {param.data.mean().item():.6f}")
            print(f"  std: {param.data.std().item():.6f}")
            print(f"  min: {param.data.min().item():.6f}")
            print(f"  max: {param.data.max().item():.6f}")

    # 4. Load environment and test predictions
    print("\n=== Environment & Predictions ===")
    model_window = model.observation_space['market'].shape[0]
    model_features = model.observation_space['market'].shape[-1]
    print(f"Model expects: window={model_window}, features={model_features}")

    env = BatchCryptoEnv(
        parquet_path=data_path,
        n_envs=1,
        device='cpu',
        window_size=model_window,
        episode_length=100,
        price_column='BTC_Close',
    )

    obs, _ = env.gym_reset()
    env_features = obs['market'].shape[-1]
    print(f"Env provides: features={env_features}")

    if model_features != env_features:
        print(f"ERROR: Feature mismatch!")
        env.close()
        return

    # 5. Test deterministic vs stochastic
    print("\n=== Deterministic vs Stochastic Actions ===")
    det_actions = []
    stoch_actions = []

    for _ in range(10):
        action_det, _ = model.predict(obs, deterministic=True)
        action_stoch, _ = model.predict(obs, deterministic=False)
        det_actions.append(action_det[0])
        stoch_actions.append(action_stoch[0])

    det_actions = np.array(det_actions)
    stoch_actions = np.array(stoch_actions)

    print(f"Deterministic: mean={det_actions.mean():.4f}, std={det_actions.std():.6f}")
    print(f"Stochastic:    mean={stoch_actions.mean():.4f}, std={stoch_actions.std():.4f}")
    print(f"Stochastic range: [{stoch_actions.min():.4f}, {stoch_actions.max():.4f}]")

    # 6. Check volatility scaling in env
    print("\n=== Environment Volatility Scaling ===")
    if hasattr(env, 'target_volatility'):
        print(f"target_volatility: {env.target_volatility}")
    if hasattr(env, 'vol_scalar'):
        print(f"vol_scalar: {env.vol_scalar}")
    if hasattr(env, 'max_leverage'):
        print(f"max_leverage: {env.max_leverage}")

    # 7. Raw actor output (before tanh squashing)
    print("\n=== Raw Actor Output (before squashing) ===")
    with torch.no_grad():
        # Get observation as tensor
        market_obs = torch.FloatTensor(obs['market'])
        position_obs = torch.FloatTensor(obs['position'])

        obs_dict = {
            'market': market_obs,
            'position': position_obs
        }

        # Get actor distribution
        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs_dict)
        print(f"Mean actions (pre-tanh): {mean_actions.numpy()}")
        print(f"Log std: {log_std.numpy()}")
        print(f"Std (exp): {torch.exp(log_std).numpy()}")

    # 8. Check over multiple different observations
    print("\n=== Actions Over Multiple Steps ===")
    actions_over_time = []
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        actions_over_time.append(action[0])
        obs, _, _, _, _ = env.gym_step(action)

    actions_over_time = np.array(actions_over_time)
    print(f"20 steps - mean: {actions_over_time.mean():.4f}, std: {actions_over_time.std():.4f}")
    print(f"Range: [{actions_over_time.min():.4f}, {actions_over_time.max():.4f}]")
    print(f"All actions: {[f'{a:.4f}' for a in actions_over_time]}")

    env.close()

    # 9. Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    if abs(det_actions.mean()) < 0.1:
        print("- Actions are very small (< 0.1)")
        print("- Possible causes:")
        print("  1. High entropy coefficient -> model explores too much")
        print("  2. Actor learned to output near-zero (conservative policy)")
        print("  3. Training collapsed to safe 'hold' strategy")
        print("  4. Volatility scaling issue in environment")

if __name__ == "__main__":
    main()
