import numpy as np
import torch
import torch.nn as nn
import onnx
from typing import Tuple
from torch.distributions import Normal
from architecture.ResidualBlock import ResidualBlock

class Fox4RLNetwork(nn.Module):
    """Neural network for Fox4."""

    def __init__(self, hidden_sizes: Tuple[int, ...] = (512, 512, 512, 256)):
        super().__init__()
        
        self.input_size = 29  # v3 input tensor
        self.action_size = 5  # v2 output tensor
        
        # Build the network layers
        shared_layers  = []
        prev_size = self.input_size
        last_shared_size = hidden_sizes[-1]
        
        for hidden_size in hidden_sizes:
            shared_layers.append(
                ResidualBlock([
                    nn.Linear(prev_size, hidden_size),
                    nn.CELU()
                ])
            )
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*shared_layers)

        # Actor Head
        # Takes the features from the shared body and decides on an action.
        # It outputs the MEAN of a probability distribution for the actions.
        self.actor_head = nn.Sequential(
            nn.Linear(last_shared_size, self.action_size),
            nn.Tanh()
        )

        # Critic Head
        # Takes the features from the shared body and estimates the value of the state.
        # It outputs a SINGLE number.
        self.critic_head = nn.Sequential(
            ResidualBlock([
                nn.Linear(last_shared_size, last_shared_size),
                nn.CELU()
            ]),
            ResidualBlock([
                nn.Linear(last_shared_size, last_shared_size),
                nn.CELU()
            ]),
            nn.Linear(last_shared_size, 1)
        )

        # Action Standard Deviation
        # PPO needs to explore by sampling actions from a distribution. We need
        # to define the standard deviation of that distribution. This is a
        # learnable parameter, so the agent can learn how much to explore.
        # Initialised to a small value, so initial stddev is small
        self.action_log_std = nn.Parameter(torch.full((1, self.action_size), -0.25))

        # Init model
        self.apply(self.init_weights)
        
        # Override init for actor head to have a very small gain
        torch.nn.init.orthogonal_(self.actor_head[0].weight, gain=0.01)
    
    def init_weights(self, module):
        """Initialize network weights. Orthogonal initialization is common for PPO."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=1.41)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass used for training.
        
        It doesn't return an action directly. Instead, it returns the
        value estimate and the action distribution. The PPO training algorithm
        will use these to calculate the loss.
        """
        # Pass the input through the shared body
        shared_features = self.shared_net(x)
        
        # --- Get outputs from the two heads ---
        value = self.critic_head(shared_features)
        action_mean = self.actor_head(shared_features)
        
        # Create the action probability distribution
        action_std = torch.exp(self.action_log_std)
        distribution = Normal(action_mean, action_std)

        return (value.flatten(), distribution)
    
    def save(self, path):
        path = path.removesuffix(".onnx")

        onnx_path = path + ".onnx"
        pth_path = path + ".pth"

        # Save as pth
        torch.save(self.state_dict(), pth_path)

        # Convert to onnx
        model = OnnxExportableModel(self.cpu())
        model.eval()
    
        # Save onnx
        torch.onnx.export(
            model,
            torch.randn(1, self.input_size),
            onnx_path,
            opset_version=18,
            input_names=['input'],
            output_names=['output','output_deviation']
        )
    
        # Add version metadata
        onnx_model = onnx.load(onnx_path)
        onnx_model.metadata_props.append(onnx.StringStringEntryProto(key="version", value="0.3-rl"))
        onnx_model.metadata_props.append(onnx.StringStringEntryProto(key="input_tensor_version", value="v3"))
        onnx_model.metadata_props.append(onnx.StringStringEntryProto(key="output_tensor_version", value="v2"))
        onnx.save(onnx_model, onnx_path)
    
class OnnxExportableModel(nn.Module):
    """
    A wrapper for the actor model that returns the distribution parameters
    (mean and std) as separate, concrete tensor outputs for ONNX.
    """
    def __init__(self, main_model: Fox4RLNetwork):
        super().__init__()

        # We need the parts of the model that produce the action mean
        self.shared_net = main_model.shared_net
        self.actor_head = main_model.actor_head
        
        # We also need the learnable log_std parameter
        # We register it here so it becomes part of this module's state
        self.register_parameter("action_log_std", main_model.action_log_std)

    def forward(self, x: torch.Tensor):
        # Calculate the mean
        shared_features = self.shared_net(x)
        action_mean = self.actor_head(shared_features)
        
        # Calculate the standard deviation
        # torch.exp is the inverse of log, recovering the std
        action_std = torch.exp(self.action_log_std)
        
        # The standard deviation is currently shape (1, 6). We need to
        # expand it to match the batch size of the input for the ONNX graph.
        action_std_expanded = action_std.expand_as(action_mean)
        
        # Return both tensors as outputs
        return action_mean, action_std_expanded