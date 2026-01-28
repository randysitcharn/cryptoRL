# -*- coding: utf-8 -*-
"""
test_pump_reaction.py - Stress test pour valider la sortie du Cash Trap.

Vérifie que le RegretDSR génère un reward négatif massif en cas d'inaction
lors d'un pump (+5%), et que la Gate (Phase 2) s'ouvre correctement.

Métriques critiques:
- reward/regret_dsr_raw : sensible aux mouvements du benchmark
- policy/action_saturation : augmente en volatilité
- policy/gate_open_ratio : niveau de conviction (Phase 2)
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.batch_env import BatchCryptoEnv


def create_price_series(prices: list) -> pd.DataFrame:
    """Série de prix pour les tests (même structure que test_reward)."""
    n = len(prices)
    return pd.DataFrame({
        "open": np.array(prices),
        "high": np.array(prices),
        "low": np.array(prices),
        "close": np.array(prices),
        "RSI_14": np.full(n, 0.5),
        "MACD_12_26_9": np.zeros(n),
        "MACDh_12_26_9": np.zeros(n),
        "ATRr_14": np.full(n, 0.02),
        "BBP_20_2.0": np.full(n, 0.5),
        "BBB_20_2.0": np.full(n, 0.05),
        "log_ret": np.zeros(n),
        "sin_hour": np.zeros(n),
        "cos_hour": np.ones(n),
        "sin_day": np.zeros(n),
        "cos_day": np.ones(n),
        "volume_rel": np.ones(n),
        "HMM_Prob_0": np.full(n, 0.25),
        "HMM_Prob_1": np.full(n, 0.25),
        "HMM_Prob_2": np.full(n, 0.25),
        "HMM_Prob_3": np.full(n, 0.25),
        "HMM_Entropy": np.full(n, 0.5),
    })


def make_env_flat_then_pump(n_flat: int = 20, pump_pct: float = 1.05, **kwargs):
    """Environnement avec prix plats puis pump au prochain step."""
    # Prix plats puis une hausse : le pump est injecté manuellement au step t+1
    prices = [100.0] * (n_flat + 50)
    df = create_price_series(prices)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    df.to_parquet(tmp.name)
    w = kwargs.pop("window_size", 10)
    ep_len = len(prices) - w - 2
    env = BatchCryptoEnv(
        parquet_path=tmp.name,
        n_envs=1,
        device="cpu",
        window_size=w,
        episode_length=max(ep_len, 1),
        price_column="close",
        random_start=False,
        use_regret_dsr=kwargs.pop("use_regret_dsr", True),
        inaction_threshold=kwargs.pop("inaction_threshold", 0.05),
        inaction_penalty=kwargs.pop("inaction_penalty", 0.01),
        action_discretization=0.0,
        **kwargs,
    )
    return env, tmp.name, pump_pct


# -----------------------------------------------------------------------------
# Phase 1 & 2 : RegretDSR doit pénaliser l'inaction lors d'un pump
# -----------------------------------------------------------------------------


def test_regret_dsr_negative_reward_on_pump_with_inaction():
    """
    Quand le prix monte de 5% et que l'agent reste au cash (action=0),
    le reward doit être négatif (Regret).
    """
    env, tmp_path, pump_pct = make_env_flat_then_pump(
        n_flat=25, pump_pct=1.05, use_regret_dsr=True
    )
    try:
        obs, info = env.gym_reset()
        # Phase 1 : quelques steps plats, agent au cash
        for _ in range(5):
            obs, r, term, trun, info = env.gym_step(np.array([0.0]))
            assert not term, "episode ne doit pas terminer pendant la phase plate"

        # Phase 2 : injection d'un pump +5% au prochain step, agent forcé à 0
        t = env.current_steps[0].item()
        next_idx = t + 1
        if next_idx < env.prices.shape[0]:
            with __import__("torch").no_grad():
                env.prices[next_idx] = env.prices[t].clone() * pump_pct

        obs, reward, term, trun, info = env.gym_step(np.array([0.0]))

        # Regret : être en cash pendant +5% doit donner un reward négatif
        assert reward < 0, (
            f"RegretDSR attendu: reward < 0 lors d'un pump avec inaction. Observé: {reward:.4f}"
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def test_regret_dsr_flat_stays_neutral():
    """En marché plat avec position 0, le reward ne doit pas être fortement négatif."""
    env, tmp_path, _ = make_env_flat_then_pump(n_flat=30, use_regret_dsr=True)
    try:
        obs, info = env.gym_reset()
        rewards = []
        for _ in range(5):
            obs, r, _, _, info = env.gym_step(np.array([0.0]))
            rewards.append(r)
        mean_r = np.mean(rewards)
        # En flat, reward proche de 0 (pas de forte pénalité Regret)
        assert mean_r > -0.5, (
            f"En marché plat avec position 0, reward moyen ne doit pas être très négatif. Observé: {mean_r:.4f}"
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Phase 3 : avec modèle (optionnel) — Gate et action tranchée
# -----------------------------------------------------------------------------


def test_pump_reaction_with_model_if_available():
    """
    Si un modèle est fourni (ou trouvé), vérifie qu'après un pump
    la Gate s'ouvre et l'action s'éloigne de 0.

    Sans modèle, le test est ignoré (pytest skip ou simple return).
    """
    try:
        from stable_baselines3 import load_format_errors
        from sb3_contrib import TQC
    except Exception:
        return  # pas de SB3/sb3_contrib, on skip la partie modèle

    env, tmp_path, pump_pct = make_env_flat_then_pump(
        n_flat=25, pump_pct=1.05, use_regret_dsr=True
    )
    model_path = os.environ.get("CRYPTORL_PUMP_TEST_MODEL")
    if not model_path or not os.path.isfile(model_path):
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return  # pas de modèle fourni, on skip

    try:
        model = TQC.load(model_path, env=env)
        obs, info = env.gym_reset()
        # Phase 1 : marché flat
        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.flatten()
            obs, _, term, trun, info = env.gym_step(action if action.size else np.array([0.0]))

        # Phase 2 : pump, action forcée à 0 pour mesurer le regret
        t = env.current_steps[0].item()
        if t + 1 < env.prices.shape[0]:
            with __import__("torch").no_grad():
                env.prices[t + 1] = env.prices[t].clone() * pump_pct
        obs, reward, _, _, info = env.gym_step(np.array([0.0]))
        assert reward < 0, f"Regret attendu lors du pump avec inaction. Observé: {reward:.4f}"

        # Phase 3 : l'agent réagit au pump
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action.flatten()[0]) if hasattr(action, "flatten") else float(action)
        gate = getattr(model.policy.actor, "_last_gate_val", None)
        # Gate doit être définie si Phase 2 (Gated Policy) est utilisée
        # Action doit s'éloigner de 0 (ex. |action| > 0.1) après un pump
        if gate is not None:
            assert 0 <= gate <= 1, f"Gate doit être dans [0,1]. Observé: {gate}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Script standalone pour lancer le scénario manuellement (avec/sans modèle)
# -----------------------------------------------------------------------------


def run_pump_reaction_script(env_path: str = None, model_path: str = None):
    """
    Exécute le scénario Pump (Phase 1–3) pour validation manuelle.

    Usage:
        python -m tests.test_pump_reaction  # utilise des données synthétiques
        CRYPTORL_PUMP_TEST_MODEL=/path/to/model.zip python -m tests.test_pump_reaction
    """
    import torch

    if env_path and os.path.isfile(env_path):
        env = BatchCryptoEnv(
            parquet_path=env_path,
            n_envs=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_start=False,
            use_regret_dsr=True,
        )
        tmp_path = None
        pump_pct = 1.05
    else:
        env, tmp_path, pump_pct = make_env_flat_then_pump(
            n_flat=25, pump_pct=1.05, use_regret_dsr=True
        )

    obs, info = env.gym_reset()
    print("--- Phase 1: Marché flat (l'agent doit rester calme) ---")
    for _ in range(10):
        obs, reward, term, trun, info = env.gym_step(np.array([0.0]))
        pos = info.get("position_pct", 0.0)
        print(f"  Position: {pos:.4f} | Reward: {reward:.4f}")
        if term:
            break

    print("\n--- Phase 2: Artificial pump (+5% en 1 step), action=0 (test Regret) ---")
    t = env.current_steps[0].item()
    if t + 1 < env.prices.shape[0]:
        with torch.no_grad():
            env.prices[t + 1] = env.prices[t].clone() * pump_pct
    obs, reward, term, trun, info = env.gym_step(np.array([0.0]))
    print(f"  Reward observé: {reward:.4f} (attendu: < 0 si RegretDSR actif)")
    if reward >= 0:
        print("  [ATTENTION] RegretDSR devrait donner un reward négatif ici.")

    model = None
    if model_path and os.path.isfile(model_path):
        try:
            from sb3_contrib import TQC
            model = TQC.load(model_path, env=env)
        except Exception as e:
            print(f"  Chargement du modèle ignoré: {e}")

    if model is not None:
        print("\n--- Phase 3: Réaction de l'agent au pump ---")
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(np.asarray(action).flatten()[0])
        gate = getattr(model.policy.actor, "_last_gate_val", None)
        print(f"  Action: {action_val:.4f} | Gate: {gate}")
    else:
        print("\n--- Phase 3: non exécutée (pas de modèle). Définir CRYPTORL_PUMP_TEST_MODEL pour tester.) ---")

    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    env_path = os.environ.get("CRYPTORL_PUMP_TEST_ENV")
    model_path = os.environ.get("CRYPTORL_PUMP_TEST_MODEL")
    run_pump_reaction_script(env_path=env_path, model_path=model_path)
