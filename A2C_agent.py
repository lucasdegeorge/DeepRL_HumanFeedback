from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch import optim


class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class
    Args:
        n_features_state: The number of features of the input state.
        n_features_action: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
    Adaptated from https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/
    """

    def __init__(
        self,
        n_features_state: int,
        n_features_action: int,
        hidden_size: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device

        critic_layers = [
            nn.Linear(n_features_state, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features_state, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size, n_features_action
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.
        Args:
            x: A state vector.
        Returns:
            state_value: A tensor with the state value, with shape [1,].
            action_logits_vec: A tensor with the action logits, with shape [n_features_action].
        """
        x = torch.Tensor(x).to(self.device)
        state_value = self.critic(x)  # shape: [1,] value given by the critic at the state
        action_logits_vec = self.actor(x)  # shape: [n_features_action]
        return (state_value, action_logits_vec)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.
        Args:
            x: A state vector.
        Returns:
            action: A tensor with the action, with shape [].
            action_log_probs: A tensor with the log-probs of the actions, with shape [].
            state_value: A tensor with the state value given by the critic, with shape [].
        """
        state_value, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        action = action_pd.sample()
        action_log_probs = action_pd.log_prob(action)
        entropy = action_pd.entropy()
        return (action, action_log_probs, state_value, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).
        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).
        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.
        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
