import torch
import torch.nn.functional as F
import numpy as np

def boxcox(x, lambda_):
    if lambda_ == 0:
        return torch.log(x)
    else:
        return (torch.pow(x, lambda_) - 1) / lambda_

lambda_best_fit = 0.0

data = torch.tensor(np.random.exponential(scale=2, size=(1, 22400, 320)), dtype=torch.float)
transformed_data = boxcox(data, lambda_best_fit)
if lambda_best_fit == 0:
    original_data = torch.exp(transformed_data)
else:
    original_data = (transformed_data * lambda_best_fit + 1).pow(1 / lambda_best_fit)