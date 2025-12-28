import platform
import torch

def get_device(verbose=True):
    system = platform.system()

    if system == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Windows 및 기타 OS는 무조건 CPU 사용
        device = torch.device("cpu")

    if verbose:
        print(f"[torch_helper] 사용 디바이스: {device}")
    return device