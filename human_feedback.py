import numpy as np
import torch
import torch.nn as nn
from torch import optim
import gymnasium as gym
import time

class Human():
    def __init__(self, 
        n_features_state: int,
        n_features_action: int,
        n_steps_per_update: int,
        dict_actions: dict
    ) -> None:
        self.n_features_state = n_features_state
        self.n_features_action = n_features_action
        self.n_steps_per_update = n_steps_per_update
        self.dict_actions = dict_actions

    def compare(self, traj_1: torch.Tensor, traj_2: torch.Tensor) -> torch.Tensor:
        """
        Shows the two trajectories to the human annotator using Gymanasium render
        Asks for the best trajectory
        Returns the human preferences
        Args: 
            traj_1: tensor of size [n_steps_per_update, n_features_state + n_features_state]
            traj_2: tensor of size [n_steps_per_update, n_features_state + n_features_action]
        Returns:
            tensor of shape [2] containing the human preferences. 
        """
        actions_1 = traj_1[:,self.n_features_state:]
        actions_2 = traj_2[:,self.n_features_state:]

        _ = input("Ready for trajectory 1? Press any key.")
        print("--- Trajectory 1 ---")
        review_env = gym.make("LunarLander-v2", render_mode="human").env
        review_env.reset()
        for step in range(self.n_steps_per_update):
            review_env.render()
            time.sleep(0.1)
            action = torch.argmax(actions_1[step]).item()
            # print("action played by the agent:", action, self.dict_actions[action])
            _ = review_env.step(action)
        review_env.close()
        print("--- End of trajectory 1 ---", '\n')

        time.sleep(1)
        _ = input("Ready for trajectory 2? Press any key.")

        print("--- Trajectory 2 ---")
        review_env = gym.make("LunarLander-v2", render_mode="human").env
        review_env.reset()
        for step in range(self.n_steps_per_update):
            review_env.render()
            time.sleep(0.1)
            action = torch.argmax(actions_2[step]).item()
            # print("action played by the agent:", action, self.dict_actions[action])
            _ = review_env.step(action)
        review_env.close()
        print("--- End of trajectory 2 ---", '\n')

        time.sleep(1)
        
        # Ask the human annotator for her feedback
        human_input = input("Which trajectory was the best? \nEnter 1 or 2 or 'tie'. \n")

        while human_input not in ['1', '2', 'tie']:
            human_input = input("You must choose between '1', '2' and 'tie'. Try again \n")

        if human_input == '1':  return torch.tensor([1.0, 0.0])
        if human_input == '2':  return torch.tensor([0.0, 1.0])
        if human_input == 'tie':  return torch.tensor([0.5, 0.5])
        else:  raise RuntimeError  
