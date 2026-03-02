"""
Test whether or not your PC has a CUDA-enabled GPU; if not, the model will train on the CPU.
If you have a CUDA-enabled but it still runs on the CPU, it's a config issue; first check that you pip-installed the right version of pytorch.
"""

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())