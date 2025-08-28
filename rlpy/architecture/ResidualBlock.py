import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):

    def __init__(self, layers, input_size = None, output_size = None):
        super().__init__()
        self.module = torch.nn.Sequential(*layers)

        # If not specified try to infer input and output size from first and last layers with in/out features attribute
        input_size = input_size or next((l.in_features for l in layers if hasattr(l, "in_features")), None)
        output_size = output_size or next((l.out_features for l in reversed(layers) if hasattr(l, "out_features")), None)
        if input_size is None or output_size is None:
            raise ValueError("Cannot infer input/output sizes from non-linear layers")
        
        # Add a linear layer to adapt from input size to output size
        if input_size != output_size:
            self.adapter = nn.Linear(input_size, output_size) 
        else:
            self.adapter = nn.Identity()

    def forward(self, inputs):
        return self.module(inputs) + self.adapter(inputs)