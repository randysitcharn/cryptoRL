# WFO Hyperparameters

> **Generated:** 2026-01-15 | **Branch:** `feat/training-speed-optimization`

---

## Source 1: `scripts/run_full_wfo.py` - `WFOConfig` (L50-109)

### WFO Window Parameters

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| train_months | 12 | Training window in months |
| test_months | 3 | Test/validation window in months |
| step_months | 3 | Rolling step size |
| hours_per_month | 720 | 30 days * 24 hours |

### Training Steps

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| mae_epochs | 90 | Foundation model (MAE) training epochs |
| tqc_timesteps | 30,000,000 | TQC training steps per segment |

### TQC Core Hyperparameters

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| learning_rate | 1e-4 | Reduced to avoid memorization |
| buffer_size | 2,500,000 | 2.5M replay buffer |
| n_envs | 1024 | GPU-optimized (power of 2 for BatchCryptoEnv) |
| batch_size | 2048 | Large batch for GPU efficiency |
| gamma | 0.95 | Discount factor (horizon ~20h) |
| ent_coef | "auto" | Auto entropy tuning |

### Curriculum Learning (3-Phase)

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| churn_coef | 0.5 | Max target after curriculum ramp |
| smooth_coef | 1e-5 | Very low base (curriculum raises to 0.00005 max) |

### Regularization (Anti-Overfitting)

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| observation_noise | 0.01 | 1% Gaussian noise on market observations |

### Volatility Scaling

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| target_volatility | 0.05 | 5% target vol |
| vol_window | 24 | 24h rolling window |
| max_leverage | 2.0 | Conservative max scaling |

### GPU Acceleration

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| use_batch_env | True | Use BatchCryptoEnv for GPU-accelerated training |

---

## Source 2: `src/config/training.py` - `TQCTrainingConfig` (L18-109)

### Environment

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| window_size | 64 | Observation window size |
| commission | 0.0015 | 0.15% - Higher cost during training (penalty) |
| train_ratio | 0.8 | Train/val split ratio |
| episode_length | 2048 | Steps per episode |
| eval_episode_length | 720 | 1 month eval (30 days * 24h) |

### Reward Function

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| reward_scaling | 1.0 | Keep at 1.0 (SCALE=100 in env) |
| downside_coef | 10.0 | Downside risk penalty |
| upside_coef | 0.0 | Upside bonus (disabled) |
| action_discretization | 0.1 | Action step size |

### Foundation Model

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| d_model | 128 | Transformer hidden dimension |
| n_heads | 4 | Number of attention heads |
| n_layers | 2 | Number of transformer layers |
| freeze_encoder | True | Freeze pretrained encoder weights |

### TQC Policy Network

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| tau | 0.005 | Soft update coefficient |
| train_freq | 1 | Training frequency |
| gradient_steps | 1 | Gradient steps per update |
| top_quantiles_to_drop | 2 | Conservative Q-value estimation |
| n_critics | 2 | Number of critic networks |
| n_quantiles | 25 | Quantiles for distributional RL |

### gSDE (State-Dependent Exploration)

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| use_sde | True | Enable gSDE |
| sde_sample_freq | -1 | Resample once per episode |
| use_sde_at_warmup | True | Use gSDE during warmup |

### Callbacks

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| eval_freq | 5,000 | Evaluation frequency |
| checkpoint_freq | 50,000 | Checkpoint save frequency |
| log_freq | 100 | Logging frequency |

### Curriculum Learning

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| use_curriculum | True | Enable 3-phase curriculum |
| curriculum_warmup_steps | 50,000 | Initial warmup phase |

### Risk Management

| Parametre | Valeur | Description |
|-----------|--------|-------------|
| use_risk_management | True | Enable risk wrapper |
| risk_vol_window | 24 | Volatility estimation window |
| risk_vol_threshold | 3.0 | Position reduction threshold |
| risk_max_drawdown | 0.10 | 10% max drawdown |
| risk_cooldown | 12 | Hours before resuming after drawdown |

---

## CLI Arguments (argparse)

```bash
python scripts/run_full_wfo.py \
    --raw-data "data/raw_training_data.parquet" \
    --segments <max_segments> \
    --segment <specific_segment> \
    --timesteps 30000000 \
    --mae-epochs 90 \
    --train-months 12 \
    --test-months 3 \
    --step-months 3 \
    --eval-only \
    --eval-segments "0,1,2" \
    --no-batch-env  # Disable GPU acceleration
```

---

## Computed Properties

| Property | Formula | Example |
|----------|---------|---------|
| train_rows | train_months * hours_per_month | 12 * 720 = 8,640 rows |
| test_rows | test_months * hours_per_month | 3 * 720 = 2,160 rows |
| step_rows | step_months * hours_per_month | 3 * 720 = 2,160 rows |

---

*End of Hyperparameters Documentation*
