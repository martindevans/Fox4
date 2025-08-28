import torch
import pandas as pd
from pathlib import Path
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class PPOParameters():
    def __init__(self, gamma = 0.99, gae_lambda = 0.95, n_epochs = 8, batch_size = 128, learning_rate = 5e-5, clip_range = 0.1, value_coeff = 0.4, entropy_coeff = 0.01, large_entropy_coeff = 0.01):
        """Hyperparameters for PPO training.

        @param: gamma Discount factor for future rewards
        @param: gae_lambda Lambda for Generalized Advantage Estimation
        @param: n_epochs How many times to loop over the data per update
        @param: batch_size Size of minibatches for training
        @param: learning_rate Learning rate for the optimizer
        @param: clip_range The PPO clipping parameter
        @param: value_coeff How much to weight the value loss
        @param: entropy_coeff slightly reward a larger entropy for distribution, keeping a bit of uncertainty is good
        @param: large_entropy_coeff Penalty for stddev over max limit
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.large_entropy_coeff = large_entropy_coeff

def load_datasets(generation_path, sim_count):
    """Load CSVs from sim folders into one big dataset"""

    all_inputs = []
    all_outputs = []
    all_extras = []
    root = Path(generation_path)

    for i in range(sim_count):
        sim_folder = Path(os.path.join(root, f"sim_{i}"))
        if not sim_folder.is_dir():
            print(f"Sim directory is not a directory: {sim_folder}")
            continue

        # Find all input CSVs
        input_files = sorted(sim_folder.glob("input_tensors-*.csv"))
        output_files = sorted(sim_folder.glob("output_tensors-*.csv"))
        extra_files = sorted(sim_folder.glob("extra_tensors-*.csv"))

        # Make sure counts match
        if len(input_files) != len(output_files):
            print(f"Mismatch in number of input/output files in sim {i}")
            continue
            
        # Load each triple
        for in_file, out_file, extra_file in zip(input_files, output_files, extra_files):
            df_in = pd.read_csv(in_file)
            df_out = pd.read_csv(out_file)
            df_extra = pd.read_csv(extra_file)

            if len(df_in) != len(df_out):
                print(f"Row count mismatch in {in_file} / {out_file}")
                continue
            if len(df_in) != len(df_extra):
                print(f"Row count mismatch in {in_file} / {extra_file}")
                continue

            all_inputs.append(df_in)
            all_outputs.append(df_out)
            all_extras.append(df_extra)
        
        # Concatenate all data
        df_inputs = pd.concat(all_inputs, ignore_index=True)
        df_outputs = pd.concat(all_outputs, ignore_index=True)
        df_extras = pd.concat(all_extras, ignore_index=True)

        # Replace all NaN with 0
        df_inputs.fillna(0, inplace=True)
        df_outputs.fillna(0, inplace=True)
        df_extras.fillna(0, inplace=True)

        return (df_inputs, df_outputs, df_extras)
    

def compute_gae_and_returns(rewards, values, dones, gamma, gae_lambda, device):
    """Calculates the advantages and returns for the rollout data."""
    advantages = torch.zeros_like(rewards).to(device)
    last_advantage = 0
    
    # We add a final 'value' for the last state, which is 0 if the episode ended
    # This simplifies the calculation loop.
    next_values = torch.cat([values[1:], torch.tensor([0.0]).to(device)]).to(device)

    # Iterate backwards through the data
    for t in reversed(range(len(rewards))):
        # The 'done' flag acts as a mask. If the episode ended, the next state's value is 0.
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        
        last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        advantages[t] = last_advantage
        
    returns = advantages + values

    return advantages, returns

def train(model, dataset_in: pd.DataFrame, dataset_out: pd.DataFrame, dataset_extra: pd.DataFrame, parameters: PPOParameters, device):

    # Set model into training mode
    model.train()

    # Load tensors
    states = torch.from_numpy(dataset_in.to_numpy(dtype="float32")).to(device)
    actions = torch.from_numpy(dataset_out.to_numpy(dtype="float32")).to(device)
    old_log_probs = torch.tensor(dataset_extra["log_prob"], dtype=torch.float32, device=device)
    values = torch.tensor(dataset_extra["value"], dtype=torch.float32, device=device)
    rewards = torch.tensor(dataset_extra["score"], dtype=torch.float32, device=device)
    dones = torch.tensor(dataset_extra["done"], dtype=torch.float32, device=device)

    # Calculate advantage and returns
    advantages, returns = compute_gae_and_returns(rewards, values, dones, parameters.gamma, parameters.gae_lambda, device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.learning_rate)

    # Create a PyTorch Dataset and DataLoader
    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    dataloader = DataLoader(dataset, batch_size=parameters.batch_size, shuffle=True)

    losses = []
    entropies = []

    for epoch in range(parameters.n_epochs):
        for batch in dataloader:

            # Unpack the minibatch of data
            batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

            # Re-evaluate the states with the CURRENT model
            # We need the new values and log_probs from the model as it's being updated
            new_values, distribution = model.forward(batch_states)
            new_log_probs = distribution.log_prob(batch_actions).sum(axis=-1)

            # Calculate the Policy (Actor) Loss
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - parameters.clip_range, 1.0 + parameters.clip_range) * batch_advantages
            loss_policy = -torch.min(surr1, surr2).mean()

            # Calculate the Value (Critic) Loss
            # This is a simple Mean Squared Error
            loss_value = ((new_values - batch_returns) ** 2).mean()

            # Penalty for large stddev (over 0.75)
            max_std = 0.75
            std_penalty = torch.relu(distribution.stddev - max_std).pow(2).mean()

            # Combine and Optimize
            loss = (loss_policy
                 + parameters.value_coeff * loss_value
                 - parameters.entropy_coeff * distribution.entropy().mean()
                 + parameters.large_entropy_coeff * std_penalty
            )

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping can also help with stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

        entropy = distribution.entropy().mean().item()
        entropies.append(entropy)
        policy_loss = loss_policy.item()
        value_loss = loss_value.item()
        losses.append((value_loss, policy_loss))
        print(f"{epoch}/{parameters.n_epochs} Policy Loss: {policy_loss:.3f}, Value Loss: {value_loss:.3f}")

    total_episodes = dataset_extra["done"].sum()
    avg_reward = dataset_extra["score"].sum() / total_episodes
    return ((model, losses, entropies), avg_reward)