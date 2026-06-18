import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {_device}")
print(f"torch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")