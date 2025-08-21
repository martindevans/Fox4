import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class Fox4Network(nn.Module):
    """Neural network for Fox4. (Awoo, sorry I had to :<)"""
    
    def __init__(self, input_size: int = 36, hidden_sizes: Tuple[int, ...] = (256, 128, 128)):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = 6  # trigger, throttle, afterburner, yaw, pitch, roll
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.network = nn.Sequential(*layers)

        # yesterday (20/08/2025) testing created huge values, added output activation
        self.output_activation = nn.Tanh()  # Constrains outputs to [-1, 1]
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.network(x)
        return self.output_activation(x)  # Apply activation to constrain outputs, see above