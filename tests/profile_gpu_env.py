#!/usr/bin/env python3
"""
profile_gpu_env.py - Benchmark BatchCryptoEnv GPU performance.

Measures:
- Throughput (steps/sec)
- Latency per step
- VRAM usage
- CPU/GPU sync bottlenecks
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.batch_env import BatchCryptoEnv


def profile_env(n_envs: int = 1024, n_steps: int = 1000, warmup: int = 100):
    print(f"{'='*60}")
    print(f"Profiling BatchCryptoEnv (N={n_envs}, Steps={n_steps})")
    print(f"{'='*60}")

    if not torch.cuda.is_available():
        print("CUDA not available! Running on CPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Memory Baseline
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1e6
    else:
        start_mem = 0

    # 2. Create Environment
    print(f"\n[1/4] Creating environment...")
    t0 = time.time()

    env = BatchCryptoEnv(
        parquet_path="data/processed_data.parquet",
        price_column="BTC_Close",
        n_envs=n_envs,
        device=device,
        window_size=64,
        episode_length=2048,
    )

    if device == "cuda":
        torch.cuda.synchronize()
    init_time = time.time() - t0

    if device == "cuda":
        init_mem = torch.cuda.memory_allocated() / 1e6
        print(f"  Init time: {init_time:.2f}s")
        print(f"  Memory (init): {init_mem - start_mem:.2f} MB")

    # 3. Reset
    print(f"\n[2/4] Resetting environment...")
    obs = env.reset()

    # Verify observation device
    print(f"  Obs market shape: {obs['market'].shape}")
    print(f"  Obs market dtype: {obs['market'].dtype}")

    # 4. Warmup
    print(f"\n[3/4] Warming up ({warmup} steps)...")
    actions = torch.zeros((n_envs, 1), device=device).numpy()

    for _ in range(warmup):
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()

    if device == "cuda":
        torch.cuda.synchronize()
        warmup_mem = torch.cuda.memory_allocated() / 1e6
        print(f"  Memory (after warmup): {warmup_mem:.2f} MB")

    # 5. Benchmark
    print(f"\n[4/4] Benchmarking ({n_steps} steps)...")

    # Preallocate random actions on GPU
    if device == "cuda":
        gpu_actions = torch.rand((n_steps, n_envs, 1), device='cuda') * 2 - 1

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    for i in range(n_steps):
        # Use GPU-generated actions (converted to numpy for SB3 interface)
        if device == "cuda":
            actions_np = gpu_actions[i].cpu().numpy()
        else:
            actions_np = (torch.rand((n_envs, 1)) * 2 - 1).numpy()

        env.step_async(actions_np)
        obs, rewards, dones, infos = env.step_wait()

        # First step validation
        if i == 0:
            print(f"\n  [Validation Step 0]")
            print(f"    rewards type: {type(rewards)}, shape: {rewards.shape}")
            print(f"    dones type: {type(dones)}, shape: {dones.shape}")
            print(f"    infos length: {len(infos)}")
            if infos:
                print(f"    info[0] keys: {list(infos[0].keys())}")

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # 6. Results
    total_time = end_time - start_time
    total_steps = n_steps * n_envs
    sps = total_steps / total_time
    latency_ms = (total_time / n_steps) * 1000

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_mem = 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total Time:     {total_time:.4f} s")
    print(f"  Throughput:     {sps:,.0f} steps/sec (global)")
    print(f"  Latency:        {latency_ms:.3f} ms/step")
    print(f"  Peak VRAM:      {peak_mem:.2f} MB")
    print(f"{'='*60}")

    # 7. Bottleneck Analysis
    print(f"\n[Bottleneck Analysis]")

    # Test pure GPU step (no observation transfer)
    if device == "cuda":
        print("  Testing internal GPU operations...")

        # Measure just the GPU tensor operations
        actions_tensor = torch.rand((n_envs,), device='cuda') * 2 - 1
        env._actions = actions_tensor

        torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(100):
            # Simulate step_wait without CPU transfers
            raw_actions = torch.clamp(env._actions, -1.0, 1.0)
            old_navs = env._get_navs()
            vol = torch.clamp(torch.sqrt(env.ema_vars), min=1e-6)
            env.vol_scalars = torch.clamp(env.target_volatility / vol, min=0.1, max=env.max_leverage)
            effective_actions = torch.clamp(raw_actions * env.vol_scalars, -1.0, 1.0)

            if env.action_discretization > 0:
                target_positions = torch.round(effective_actions / env.action_discretization) * env.action_discretization
            else:
                target_positions = effective_actions

            position_deltas = torch.abs(target_positions - env.position_pcts)
            env.current_steps += 1
            new_navs = env._get_navs()
            env.current_steps -= 1  # Revert for next iteration

        torch.cuda.synchronize()
        gpu_only_time = time.time() - t0
        gpu_only_sps = (100 * n_envs) / gpu_only_time

        print(f"  Pure GPU ops: {gpu_only_sps:,.0f} steps/sec")
        print(f"  Overhead ratio: {sps / gpu_only_sps:.2%} of theoretical max")

    env.close()
    return sps, latency_ms, peak_mem


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=1000)
    args = parser.parse_args()

    try:
        profile_env(n_envs=args.n_envs, n_steps=args.n_steps)
    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {e}")
        import traceback
        traceback.print_exc()
