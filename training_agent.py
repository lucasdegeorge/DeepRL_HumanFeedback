#%% 
import gymnasium as gym
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from A2C_agent import A2C
from reward_predictor import Reward_predictor
from human_feedback import Human


# environment hyperparams
n_updates = 8 # 100
n_steps_per_update = 16
n_trajectories = 2

# A2C agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 0.001
critic_lr = 0.005

# reward predictor hyperparams
reward_lr = 0.005
hidden_size = 32
n_comp = 2

# environment setup
env = gym.make("LunarLander-v2")
env.reset()  # Is it mandatory ? 
max_episode_steps = env.spec.max_episode_steps

n_features_state = env.observation_space.shape[0]  # state_shape
n_features_action = env.action_space.n  # action_shape

dict_actions = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine"
}


# set the device
use_cuda = False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print("device:", device)

# init the agent
agent = A2C(n_features_state, n_features_action, hidden_size, device, critic_lr, actor_lr)
# init the reward
reward_predictor = Reward_predictor(n_features_state, n_features_action, hidden_size, device, reward_lr)
# init the human 
human = Human(n_features_state, n_features_action, n_steps_per_update, dict_actions)

critic_losses = []
actor_losses = []
entropies = []


for sample_phase in tqdm(range(n_updates)):
    
    all_trajectories = torch.zeros((n_trajectories, n_steps_per_update, n_features_state+n_features_action), device=device)

    for traj_idx in range(n_trajectories):

        ### Process 1 ###

        ep_value_preds = torch.zeros(n_steps_per_update, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, device=device)
        masks = torch.zeros(n_steps_per_update, device=device)

        ep_states = torch.zeros((n_steps_per_update, n_features_state), device=device)  # Store states
        ep_actions = torch.zeros((n_steps_per_update, n_features_action), device=device)  # Store actions

        if sample_phase == 0:
            state, infos = env.reset()

        # play n steps in the environment to collect data
        for step in range(n_steps_per_update):

            # select an action A_{t} using S_{t} as input for the agent
            ep_states[step] = torch.from_numpy(state)
            action, action_log_prob, state_value_pred, entropy = agent.select_action(state)
            ep_actions[step] = F.one_hot(action, num_classes=4)

            # perform the action A_{t} in the environment to get S_{t+1}. 
            state, reward, terminated, truncated, _ = env.step(action.item())
            # get \hat{R}_{t+1} from the reward predictor
            reward = reward_predictor(state, F.one_hot(action, num_classes=4)).item()

            ep_value_preds[step] = state_value_pred
            ep_rewards[step] = torch.tensor(reward, device=device)
            ep_action_log_probs[step] = action_log_prob

            masks[step] = torch.tensor([not terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

        ### Process 2
        all_trajectories[traj_idx] = torch.cat([ep_states, ep_actions], dim=1)
    
    ### Process 3
    reward_loss = reward_predictor.compute_loss(all_trajectories, n_comp, human)
    reward_predictor.update_parameters(reward_loss)


#%% 

## Plotting 
rolling_length = 20
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
fig.suptitle(
    f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment \n \
             (n_steps_per_update={n_steps_per_update})"
)

# episode return
# axs[0][0].set_title("Episode Returns")
# episode_returns_moving_average = (
#     np.convolve(
#         np.array(envs_wrapper.return_queue).flatten(),
#         np.ones(rolling_length),
#         mode="valid",
#     )
#     / rolling_length
# )
# axs[0][0].plot(
#     np.arange(len(episode_returns_moving_average)) / n_envs,
#     episode_returns_moving_average,
# )
# axs[0][0].set_xlabel("Number of episodes")

# entropy
axs[1][0].set_title("Entropy")
entropy_moving_average = (
    np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][0].plot(entropy_moving_average)
axs[1][0].set_xlabel("Number of updates")


# critic loss
axs[0][1].set_title("Critic Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0][1].plot(critic_losses_moving_average)
axs[0][1].set_xlabel("Number of updates")


# actor loss
axs[1][1].set_title("Actor Loss")
actor_losses_moving_average = (
    np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][1].plot(actor_losses_moving_average)
axs[1][1].set_xlabel("Number of updates")

plt.tight_layout()
plt.show()