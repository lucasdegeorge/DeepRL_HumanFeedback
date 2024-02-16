from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from human_feedback import Human
from typing import Union


class Reward_predictor(nn.Module):
    """
    Args:
        n_features_state: The number of features of the input state.
        n_features_action: The number of features of the input action (=|A|)
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        lr: The learning rate
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features_state: int,
        n_features_action: int,
        hidden_size: int, 
        device: torch.device,
        lr: float,
    ) -> None:
        """Initializes the reward predictor network and its optimizer."""
        super().__init__()
        self.device = device
        self.n_features_state = n_features_state
        self.n_features_action = n_features_action

        layers = [
            nn.Linear(n_features_state + n_features_action, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # outputs \hat{r} 
        ]

        self.reward = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.RMSprop(self.reward.parameters(), lr=lr)

    def forward(self, s: Union[torch.Tensor, np.ndarray], 
        a: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Forward pass of the networks.
        Args:
            s: A state vector of shape [n_features_state] or [n_step_per_update, n_features_state]
            a: A action vector of shape [n_features_action] or [n_step_per_update, n_features_action]
        Returns:
            estimated_reward: A tensor with the estimated reward, with shape [].
        """
        if not(isinstance(s, torch.Tensor)):
            s = torch.Tensor(s).to(self.device)
        if not(isinstance(a, torch.Tensor)):
            a = torch.Tensor(a).to(self.device)
        x = torch.cat([s, a], dim=-1)
        estimated_reward = self.reward(x)  # shape: []
        return estimated_reward
    
    def estimate_preferences(self, traj_1: torch.Tensor, traj_2: torch.Tensor):
        """
        Predict probability of human preferring one segment over another. 
        Args: 
            traj_1: tensors containing the states and the actions of trajectory 1 
            traj_2: tensors containing the states and the actions tensors of trajectory 2
            (tensor ofshape [n_steps_per_update, n_features_state+n_features_action])
        Returns:
            A tensor with the estamed preferences.
        """
        states_1, actions_1 = traj_1[:,:self.n_features_state], traj_1[:,self.n_features_state:]
        states_2, actions_2 = traj_2[:,:self.n_features_state], traj_2[:,self.n_features_state:]
        rewards_1 = self.forward(states_1, actions_1).sum(dim=0)  # shape [1]  Mean of rewards on traj_1
        rewards_2 = self.forward(states_2, actions_2).sum(dim=0)  # shape [1]  Mean of rewards on traj_2
        rewards = torch.cat([rewards_1, rewards_2], dim=0)  
        preferences = torch.softmax(rewards, dim=0)  # shape [2]
        return preferences
    
    def compute_loss(self, all_trajectories: torch.Tensor, n_comp: int, human: Human):
        """
        Computes the loss from Section 2.2.3 of [1]. This loss is used to update the parameters of the reward predictor. 
        Select randomly nb_comp couples of trajectories. Ask the human agent to compare the trajectories. 
        Predict the preferences and compute the loss (Cross-Entropy)
        Args: 
            all_trajectories: tensor of size [n_trajectory, n_steps_per_update, n_features_state + n_features_action]
        Returns:
            reward_loss: the loss for the batch
        """
        estimated_pref = torch.zeros(n_comp, 2)
        human_pref = torch.zeros(n_comp, 2)
        couples = np.random.randint(0, all_trajectories.shape[0], n_comp*2).reshape((2,-1))
        for idx, (traj_1_idx, traj_2_idx) in enumerate(zip(couples[0], couples[1])):
            estimated_pref[idx] = self.estimate_preferences(all_trajectories[traj_1_idx], all_trajectories[traj_2_idx])  # each has shape [2]
            human_pref[idx] = human.compare(all_trajectories[traj_1_idx], all_trajectories[traj_2_idx])  # each has shape [2]
        loss = nn.CrossEntropyLoss(reduction='sum')(estimated_pref, human_pref)
        print("loss", loss.item())
        return loss

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
        self.optimizer.step()