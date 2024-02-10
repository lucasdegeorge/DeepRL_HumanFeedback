from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import optim


class Reward_predictor(nn.Module):
    """
    Args:
        n_features: The number of features of the input state.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        lr: The learning rate
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        device: torch.device,
        lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the reward predictor network and its optimizer."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # outputs \hat{r} 
        ]

        self.reward = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.RMSprop(self.reward.parameters(), lr=lr)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the networks.
        Args:
            x: A batched vector of states.
        Returns:
            estimated_reward: A tensor with the estimated reward, with shape [n_envs,].
        """
        x = torch.Tensor(x).to(self.device)
        estimated_reward = self.reward(x)  # shape: [n_envs,]
        return estimated_reward
    
    def estimate_preferences(self, traj_1: dict, traj_2: dict):
        """
        Predict probability of human preferring one segment over another. 
        Args: 
            traj_1: dict containing the states and the actions of trajectory 1
            traj_2: dict containing the states and the actions of trajectory 2
        Returns:
            A tensor with the estamed preferences.
        """

        raise NotImplementedError

    def update_parameters(
        self, reward_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.
        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.optimizer.zero_grad()
        reward_loss.backward()
        self.critic_optim.step()