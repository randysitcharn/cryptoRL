"""
Hardware Auto-Detection and Adaptive Configuration Module.

This module provides automatic hardware detection (CPU, GPU, RAM) and
computes optimal training hyperparameters based on available resources.

Based on Gemini collaboration (2026-01-13):
- CPU: Use physical cores - 1 for n_envs (cap at 32)
- GPU: Scale batch_size based on VRAM tiers
- RAM: Allocate 40% of available RAM for buffer_size

Usage:
    from src.utils.hardware import HardwareManager

    hw = HardwareManager()
    config = hw.get_adaptive_config()
    hw.apply_optimizations(config)
    hw.log_summary(config)
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil
import torch

# Try importing pynvml for accurate NVIDIA GPU stats
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class HardwareSpecs:
    """Detected hardware specifications."""
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_count: int
    gpu_name: str
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float
    cuda_version: str
    device: str


@dataclass
class OptimalConfig:
    """Computed optimal training configuration."""
    n_envs: int
    batch_size: int
    buffer_size: int
    device: str
    torch_compile: bool
    num_threads: int


class HardwareManager:
    """
    Manages hardware detection and adaptive configuration.

    This class detects available hardware resources and computes
    optimal hyperparameters for training. Supports user overrides.

    Attributes:
        specs: Detected hardware specifications
        logger: Logger instance for output
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize HardwareManager.

        Args:
            logger: Optional logger. If None, creates a default logger.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.specs = self._detect_hardware()

    def _detect_hardware(self) -> HardwareSpecs:
        """
        Detect available hardware resources.

        Returns:
            HardwareSpecs with detected CPU, RAM, and GPU info.
        """
        # 1. CPU Detection
        phys_cores = psutil.cpu_count(logical=False) or 1
        log_cores = psutil.cpu_count(logical=True) or 1

        # 2. RAM Detection
        vm = psutil.virtual_memory()
        ram_total = vm.total / (1024**3)
        ram_avail = vm.available / (1024**3)

        # 3. GPU Detection
        gpu_count = 0
        gpu_name = "CPU Only"
        gpu_vram_total = 0.0
        gpu_vram_free = 0.0
        device = "cpu"
        cuda_ver = "N/A"

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device = "cuda"
            cuda_ver = torch.version.cuda or "Unknown"

            # Get VRAM stats - prefer pynvml for accurate free memory
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    name = pynvml.nvmlDeviceGetName(handle)
                    gpu_name = name.decode('utf-8') if isinstance(name, bytes) else name
                    gpu_vram_total = info.total / (1024**3)
                    gpu_vram_free = info.free / (1024**3)
                    pynvml.nvmlShutdown()
                except Exception as e:
                    self.logger.warning(f"pynvml failed, using torch fallback: {e}")
                    props = torch.cuda.get_device_properties(0)
                    gpu_name = props.name
                    gpu_vram_total = props.total_memory / (1024**3)
                    gpu_vram_free = gpu_vram_total * 0.85  # Conservative estimate
            else:
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                gpu_vram_total = props.total_memory / (1024**3)
                gpu_vram_free = gpu_vram_total * 0.85  # Conservative estimate

        return HardwareSpecs(
            cpu_physical_cores=phys_cores,
            cpu_logical_cores=log_cores,
            ram_total_gb=ram_total,
            ram_available_gb=ram_avail,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_vram_total_gb=gpu_vram_total,
            gpu_vram_free_gb=gpu_vram_free,
            cuda_version=cuda_ver,
            device=device
        )

    def get_adaptive_config(
        self,
        user_overrides: Optional[Dict[str, Any]] = None,
        obs_size_bytes: int = 11_000
    ) -> OptimalConfig:
        """
        Compute optimal configuration based on hardware.

        Args:
            user_overrides: Dict of user-specified values to override auto-config.
                           Keys: n_envs, batch_size, buffer_size, torch_compile
            obs_size_bytes: Size of one observation in bytes.
                           Default: ~11KB (43 features * 64 steps * 4 bytes)

        Returns:
            OptimalConfig with computed hyperparameters.
        """
        specs = self.specs
        overrides = user_overrides or {}

        # --- Rule 1: n_envs (CPU Bound) ---
        # Use physical cores - 1 (leave 1 for OS/learner)
        # Cap at 32 to avoid IPC overhead
        rec_envs = max(1, specs.cpu_physical_cores - 1)
        rec_envs = min(rec_envs, 32)

        # --- Rule 2: batch_size (VRAM Bound) ---
        # Tiered based on FREE VRAM
        vram_ref = specs.gpu_vram_free_gb
        if vram_ref < 4.0:
            rec_batch = 256
        elif vram_ref < 8.0:
            rec_batch = 512
        elif vram_ref < 16.0:
            rec_batch = 1024
        else:
            rec_batch = 2048

        # --- Rule 3: buffer_size (RAM Bound) ---
        # Transition size: obs + action + reward + next_obs + done
        # Approximate: 3x observation size
        transition_size = obs_size_bytes * 3

        # Allocate max 40% of AVAILABLE RAM for buffer
        ram_budget_bytes = specs.ram_available_gb * 0.4 * (1024**3)
        rec_buffer = int(ram_budget_bytes / transition_size)

        # Safety bounds
        rec_buffer = min(rec_buffer, 1_000_000)  # Standard SAC/TQC max
        rec_buffer = max(rec_buffer, 50_000)     # Minimum viable for learning

        # --- Rule 4: torch.compile ---
        # Only enable on Linux + CUDA + PyTorch 2.0+
        can_compile = (
            specs.device == "cuda" and
            hasattr(torch, "compile") and
            os.name == 'posix'  # Linux/Mac only, buggy on Windows
        )

        # --- Rule 5: num_threads ---
        # Limit intra-op parallelism when running many envs
        rec_threads = 2 if rec_envs < 8 else 1

        # --- Apply User Overrides ---
        final_config = OptimalConfig(
            n_envs=overrides.get("n_envs", rec_envs),
            batch_size=overrides.get("batch_size", rec_batch),
            buffer_size=overrides.get("buffer_size", rec_buffer),
            device=specs.device,
            torch_compile=overrides.get("torch_compile", can_compile),
            num_threads=overrides.get("num_threads", rec_threads)
        )

        return final_config

    def apply_optimizations(self, config: OptimalConfig) -> None:
        """
        Apply global PyTorch optimizations based on config.

        This sets:
        - TF32 precision for Ampere+ GPUs (RTX 3000/4000, A100)
        - cuDNN benchmark mode for consistent input sizes
        - Number of threads for intra-op parallelism

        Args:
            config: OptimalConfig from get_adaptive_config()
        """
        if config.device == "cuda":
            # TF32 for Ampere+ GPUs - significant speedup, minimal precision loss
            torch.set_float32_matmul_precision('high')
            # cuDNN benchmark - auto-tune convolutions for fixed input sizes
            torch.backends.cudnn.benchmark = True
            self.logger.info("[Optim] TF32 + cuDNN benchmark enabled")

        # Set thread count for intra-op parallelism
        torch.set_num_threads(config.num_threads)
        self.logger.info(f"[Optim] PyTorch threads set to {config.num_threads}")

    def log_summary(self, config: OptimalConfig) -> None:
        """
        Log hardware specs and adaptive configuration.

        Args:
            config: OptimalConfig to display
        """
        s = self.specs

        self.logger.info("=" * 55)
        self.logger.info("HARDWARE AUTO-DETECTION")
        self.logger.info("-" * 55)
        self.logger.info(f"CPU  : {s.cpu_physical_cores} Physical / {s.cpu_logical_cores} Logical Cores")
        self.logger.info(f"RAM  : {s.ram_available_gb:.1f} GB Available / {s.ram_total_gb:.1f} GB Total")

        if s.device == "cuda":
            self.logger.info(f"GPU  : {s.gpu_name}")
            self.logger.info(f"VRAM : {s.gpu_vram_free_gb:.1f} GB Free / {s.gpu_vram_total_gb:.1f} GB Total")
            self.logger.info(f"CUDA : {s.cuda_version} ({s.gpu_count} GPU(s))")
        else:
            self.logger.info("GPU  : Not available (using CPU)")

        self.logger.info("-" * 55)
        self.logger.info("ADAPTIVE CONFIGURATION")
        self.logger.info("-" * 55)
        self.logger.info(f"n_envs      : {config.n_envs:<6} (optimized for CPU cores)")
        self.logger.info(f"batch_size  : {config.batch_size:<6} (optimized for VRAM)")
        self.logger.info(f"buffer_size : {config.buffer_size:<6} (optimized for RAM)")
        self.logger.info(f"device      : {config.device}")
        self.logger.info(f"torch.compile: {'Enabled' if config.torch_compile else 'Disabled'}")
        self.logger.info("=" * 55)


def get_hardware_summary() -> str:
    """
    Quick utility to get a one-line hardware summary.

    Returns:
        String like "8 cores | 32GB RAM | RTX 4090 (24GB)"
    """
    hw = HardwareManager(logger=logging.getLogger("hardware"))
    s = hw.specs

    if s.device == "cuda":
        return f"{s.cpu_physical_cores} cores | {s.ram_total_gb:.0f}GB RAM | {s.gpu_name} ({s.gpu_vram_total_gb:.0f}GB)"
    else:
        return f"{s.cpu_physical_cores} cores | {s.ram_total_gb:.0f}GB RAM | CPU only"
