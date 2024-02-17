#%% 
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import csv

from A2C_agent import A2C
from reward_predictor import Reward_predictor
from human_feedback import Human

class Trainer:
    def __init__(self, env_name: str, 
        exp_number: int,
        n_updates: int, 
        n_steps_per_update: int,
        n_trajectories: int, 
        segment_length: int,
        hidden_size: int,
        n_comp: int
    ):
        # Environment setup 
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset()

        self.n_features_state = self.env.observation_space.shape[0]
        self.n_features_action = self.env.action_space.n

        self.dict_actions = {
            0: "do nothing",
            1: "fire left orientation engine",
            2: "fire main engine",
            3: "fire right orientation engine"
        }

        self.n_updates = n_updates
        self.n_steps_per_update = n_steps_per_update
        self.n_trajectories = n_trajectories
        self.segment_length = segment_length

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        # A2C hyperparams
        self.gamma = 0.999
        self.lam = 0.95  # hyperparameter for GAE
        self.ent_coef = 0.1  # coefficient for the entropy bonus (to encourage exploration)
        actor_lr = 0.001
        critic_lr = 0.005

        # reward predictor hyperparams
        reward_lr = 0.005
        hidden_size = 32
        self.n_comp = n_comp

        # Init the agent
        self.agent = A2C(self.n_features_state, self.n_features_action, hidden_size, self.device, critic_lr, actor_lr)
        # Init the reward
        self.reward_predictor = Reward_predictor(self.n_features_state, self.n_features_action, hidden_size, self.n_steps_per_update, self.segment_length, self.device, reward_lr)
        # Init the human
        self.human = Human(self.n_features_state, self.n_features_action, n_steps_per_update, self.segment_length, self.dict_actions)

        self.critic_losses = []
        self.actor_losses = []
        self.entropies = []
        self.estimated_rewards = []
        self.real_rewards = []

        self.exp_number = exp_number

    def train(self):
        for sample_phase in tqdm(range(self.n_updates)):
            all_trajectories = torch.zeros((self.n_trajectories, self.n_steps_per_update, self.n_features_state + self.n_features_action), device=self.device)

            for traj_idx in range(self.n_trajectories):
                ep_value_preds = torch.zeros(self.n_steps_per_update, device=self.device)
                ep_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
                ep_real_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
                ep_action_log_probs = torch.zeros(self.n_steps_per_update, device=self.device)
                masks = torch.zeros(self.n_steps_per_update, device=self.device)

                ep_states = torch.zeros((self.n_steps_per_update, self.n_features_state), device=self.device)
                ep_actions = torch.zeros((self.n_steps_per_update, self.n_features_action), device=self.device)

                if sample_phase == 0:
                    state, _ = self.env.reset()

                for step in range(self.n_steps_per_update):
                    ep_states[step] = torch.from_numpy(state)
                    action, action_log_prob, state_value_pred, entropy = self.agent.select_action(state)
                    ep_actions[step] = F.one_hot(action, num_classes=4)

                    state, real_reward, terminated, _, _ = self.env.step(action.item())
                    reward = self.reward_predictor(state, F.one_hot(action, num_classes=4)).item()

                    ep_value_preds[step] = state_value_pred
                    ep_rewards[step] = torch.tensor(reward, device=self.device)
                    ep_action_log_probs[step] = action_log_prob
                    ep_real_rewards[step] = torch.tensor(real_reward, device=self.device)

                    masks[step] = torch.tensor([not terminated])

                critic_loss, actor_loss = self.agent.get_losses(
                    ep_rewards,  # replace ep_rewards by ep_real_rewards to get a classical A2C training
                    ep_action_log_probs,
                    ep_value_preds,
                    entropy,
                    masks, 
                    self.gamma,
                    self.lam,
                    self.ent_coef,
                    self.device,
                )

                self.agent.update_parameters(critic_loss, actor_loss)

                self.critic_losses.append(critic_loss.detach().cpu().numpy().item())
                self.actor_losses.append(actor_loss.detach().cpu().numpy().item())
                self.entropies.append(entropy.detach().mean().cpu().numpy().item())
                self.estimated_rewards = np.concatenate([self.estimated_rewards, ep_rewards.detach().cpu().numpy()])
                self.real_rewards = np.concatenate([self.real_rewards, ep_real_rewards.detach().cpu().numpy()])

                all_trajectories[traj_idx] = torch.cat([ep_states, ep_actions], dim=1)

            reward_loss = self.reward_predictor.compute_loss(all_trajectories, self.n_comp, self.human)
            self.reward_predictor.update_parameters(reward_loss)
            self.save_results(sample_phase)
    
    
    def save_results(self, sample):
        data = zip(self.critic_losses, 
            self.actor_losses, 
            self.entropies, 
            self.estimated_rewards.tolist(), 
            self.real_rewards.tolist()
        )

        with open(f"logs/exp{self.exp_number}_sample{sample}.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['critic loss', 'actor loss', 'entropy', 'estimated rewards', 'real rewards'])
            csv_writer.writerows(data)


    def plot_results(self, print_real=False):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(f"Training plots for {self.agent.__class__.__name__} in the {self.env_name} environment")

        axs[0][0].set_title("Rewards")
        axs[0][0].plot(
            np.arange(len(self.estimated_rewards)),
            self.estimated_rewards,
            label="Estimated rewards"
        )
        if print_real:
            axs[0][0].plot(
                np.arange(len(self.real_rewards)),
                self.real_rewards, 
                label="Real rewards"
            )
        axs[0][0].legend()
        axs[0][0].set_xlabel("Rewards (estimated and real)")

        axs[1][0].set_title("Entropy")
        axs[1][0].plot(
            np.arange(len(self.entropies)),
            self.entropies)
        axs[1][0].set_xlabel("Number of updates")

        axs[0][1].set_title("Critic Loss")
        axs[0][1].plot(
            np.arange(len(self.critic_losses)),
            self.critic_losses)
        axs[0][1].set_xlabel("Number of updates")

        axs[1][1].set_title("Actor Loss")
        axs[1][1].plot(
            np.arange(len(self.actor_losses)),
            self.actor_losses)
        axs[1][1].set_xlabel("Number of updates")

        plt.tight_layout()
        plt.show()



trainer = Trainer(
    env_name="LunarLander-v2",
    exp_number=0,
    n_updates=100,
    n_steps_per_update=100,
    n_trajectories=15,
    segment_length=32,
    hidden_size=32,
    n_comp=1
)

trainer.train()
trainer.plot_results()