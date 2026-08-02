import torch
import gpytorch
import pykeops
import linear_operator

print("imports OK")
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("linear_operator", linear_operator.__version__)
print("gpytorch", gpytorch.__version__)
