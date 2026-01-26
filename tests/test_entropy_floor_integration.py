# -*- coding: utf-8 -*-
"""
test_entropy_floor_integration.py - Integration test for EntropyFloorCallback.

Validates that EntropyFloorCallback is automatically added to callbacks
in train_agent.create_callbacks().
"""

def test_entropy_floor_in_callbacks():
    """Test que EntropyFloorCallback est présent dans create_callbacks()."""
    from src.training.train_agent import create_callbacks
    from src.training.callbacks import EntropyFloorCallback
    from src.config import TQCTrainingConfig

    config = TQCTrainingConfig()
    # eval_env=None pour simuler le mode WFO
    callbacks, _ = create_callbacks(config, eval_env=None)

    # Vérifier qu'EntropyFloorCallback est présent
    has_entropy_floor = any(isinstance(cb, EntropyFloorCallback) for cb in callbacks)
    assert has_entropy_floor, 'EntropyFloorCallback manquant dans create_callbacks()!'
    print('✅ Integration test: EntropyFloorCallback présent dans create_callbacks()')


def test_config_ent_coef():
    """Test que TQCTrainingConfig et WFOTrainingConfig ont ent_coef = 'auto_0.5'."""
    from src.config import TQCTrainingConfig, WFOTrainingConfig

    # Test TQCTrainingConfig
    tqc_config = TQCTrainingConfig()
    assert tqc_config.ent_coef == "auto_0.5", f"TQCTrainingConfig.ent_coef devrait être 'auto_0.5', got {tqc_config.ent_coef}"
    print(f"✅ TQCTrainingConfig.ent_coef: {tqc_config.ent_coef}")

    # Test WFOTrainingConfig
    wfo_config = WFOTrainingConfig()
    assert wfo_config.ent_coef == "auto_0.5", f"WFOTrainingConfig.ent_coef devrait être 'auto_0.5', got {wfo_config.ent_coef}"
    print(f"✅ WFOTrainingConfig.ent_coef: {wfo_config.ent_coef}")


if __name__ == "__main__":
    test_entropy_floor_in_callbacks()
    test_config_ent_coef()
    print("\n✅ ALL INTEGRATION TESTS PASSED")
