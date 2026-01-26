# -*- coding: utf-8 -*-
"""
test_entropy_floor.py - Tests for EntropyFloorCallback.

Validates that the callback prevents entropy collapse by enforcing a minimum
entropy coefficient threshold.
"""

import numpy as np
import torch
from unittest.mock import MagicMock
from src.training.callbacks import EntropyFloorCallback


def test_entropy_floor_applies():
    """Test que le floor est appliqué quand ent_coef trop bas."""
    callback = EntropyFloorCallback(min_ent_coef=0.01, check_freq=1)
    
    # Mock model avec log_ent_coef trop bas
    callback.model = MagicMock()
    callback.model.log_ent_coef = torch.tensor(np.log(0.001))  # 0.001 < 0.01
    callback.logger = None
    callback.n_calls = 1
    
    callback._on_step()
    
    # Vérifier que le floor a été appliqué
    expected_log = np.log(0.01)
    actual_log = callback.model.log_ent_coef.item()
    assert abs(actual_log - expected_log) < 1e-6, f"Expected {expected_log}, got {actual_log}"
    assert callback.floor_count == 1
    print("✅ Test entropy floor applies: PASSED")


def test_entropy_floor_no_apply():
    """Test que le floor n'est PAS appliqué quand ent_coef OK."""
    callback = EntropyFloorCallback(min_ent_coef=0.01, check_freq=1)
    
    # Mock model avec log_ent_coef OK
    callback.model = MagicMock()
    original_log = np.log(0.05)  # 0.05 > 0.01
    callback.model.log_ent_coef = torch.tensor(original_log)
    callback.logger = None
    callback.n_calls = 1
    
    callback._on_step()
    
    # Vérifier que rien n'a changé
    actual_log = callback.model.log_ent_coef.item()
    assert abs(actual_log - original_log) < 1e-6
    assert callback.floor_count == 0
    print("✅ Test entropy floor no apply: PASSED")


def test_entropy_floor_check_freq():
    """Test que le callback ne vérifie que toutes les check_freq steps."""
    callback = EntropyFloorCallback(min_ent_coef=0.01, check_freq=5)
    
    callback.model = MagicMock()
    callback.model.log_ent_coef = torch.tensor(np.log(0.001))  # Trop bas
    callback.logger = None
    
    # Appeler 4 fois (pas de vérification)
    for i in range(1, 5):
        callback.n_calls = i
        callback._on_step()
        assert callback.floor_count == 0, f"Floor should not apply at step {i}"
    
    # 5ème appel (check_freq) - floor doit être appliqué
    callback.n_calls = 5
    callback._on_step()
    assert callback.floor_count == 1, "Floor should apply at check_freq"
    print("✅ Test entropy floor check_freq: PASSED")


if __name__ == "__main__":
    test_entropy_floor_applies()
    test_entropy_floor_no_apply()
    test_entropy_floor_check_freq()
    print("\n✅ ALL TESTS PASSED")
