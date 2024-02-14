from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from human_feedback import Human


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
        n_envs: int,
    ) -> None:
        """Initializes the reward predictor network and its optimizer."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
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

    def forward(self, s: np.ndarray, a: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the networks.
        Args:
            s: A batched vector of states.
            a: A batched vector of actions
        Returns:
            estimated_reward: A tensor with the estimated reward, with shape [n_envs,].
        """
        s = torch.Tensor(s).to(self.device)
        a = torch.Tensor(a).to(self.device)
        x = torch.cat([s, a], dim=1)
        estimated_reward = self.reward(x)  # shape: [n_envs,]
        return estimated_reward
    
    def estimate_preferences(self, traj_1: torch.Tensor, traj_2: torch.Tensor):
        """
        Predict probability of human preferring one segment over another. 
        Args: 
            traj_1: tensors containing the states and the actions of trajectory 1
            traj_2: tensors containing the states and the actions tensors of trajectory 2
            (tensors of size [n_envs, n_features_state + n_features_action])
        Returns:
            A tensor with the estamed preferences.
        """
        states_1, actions_1 = traj_1[:,:self.n_features_action], traj_1[:,self.n_features_action:]
        states_2, actions_2 = traj_2[:,:self.n_features_action], traj_2[:,self.n_features_action:]
        rewards_1 = self.forward(states_1, actions_1).unsqueeze(-1)  # [n_envs, 1]
        rewards_2 = self.forward(states_2, actions_2).unsqueeze(-1)  # [n_envs, 1]
        rewards = torch.cat(rewards_1, rewards_2)  
        preferences = torch.softmax(rewards, dim=1)  # [n_envs, 2]
        return preferences
    
    def compute_loss(self, all_trajectories: torch.Tensor, nb_comp: int, human: Human):
        """
        Computes the loss from Section 2.2.3 of [1]. This loss is used to update the parameters of the reward predictor. 
        Select randomly nb_comp couples of trajectories. Ask the human agent to compare the trajectories. 
        Predict the preferences and compute the loss (Cross-Entropy)
        Args: 
            all_trajectories: tensor of size [n_trajectory, n_envs, n_features_state + n_features_action]
        Returns:
            reward_loss: the loss for the batch
        """
        estimated_preferences = torch.zeros(nb_comp, self.n_envs, 2)
        human_preferences = torch.zeros(nb_comp, self.n_envs, 2)
        couples = np.random.randint(0, all_trajectories.shape[0], nb_comp*2).reshape((2,-1))
        for idx, (traj_1_idx, traj_2_idx) in enumerate(couples):
            estimated_preferences[idx] = self.estimated_preferences(all_trajectories[traj_1_idx], all_trajectories[traj_2_idx])  # [n_envs, 2]
            human_preferences[idx] = human.compare(all_trajectories[traj_1_idx], all_trajectories[traj_2_idx])  # [n_envs, 2]
        loss = nn.CrossEntropyLoss()(estimated_preferences, human_preferences)
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