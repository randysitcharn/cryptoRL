import sys
import torch
import gymnasium
import stable_baselines3

def setup_env():
    print("Checking environment setup...")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Using CPU.")

    # Print versions
    print(f"Gymnasium version: {gymnasium.__version__}")
    print(f"Stable Baselines3 version: {stable_baselines3.__version__}")

    print("-" * 30)
    print("Environment Setup Complete. System Ready.")

if __name__ == "__main__":
    setup_env()
