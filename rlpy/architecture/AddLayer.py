import torch
import torch.nn as nn

class AddLayer(torch.nn.Module):
    def __init__(self, alpha = 1):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        x = torch.add(x, self.alpha)
        return x