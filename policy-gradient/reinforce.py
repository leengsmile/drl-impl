"""
reinforce: https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb

"""
import argparse
from collections import deque
import itertools
import logging
import math
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):

    def __init__(self, num_states: int, num_actions: int, hidden_dim: int):
        super().__init__()
       
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x: torch.Tensor):
        out = self.net(x)
        out = F.softmax(out, dim=-1)
        return out 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Reinforce in RL')
    parser.add_argument("--train", action="store_true", help="Train the agent")

    args, _ = parser.parse_known_args()
    return args


def initialize(log_file):
    base_dir = os.path.dirname(log_file)
    os.makedirs(base_dir, exist_ok=True)
    
    FORMAT = '[%(asctime)s]-[%(filename)s:%(funcName)s:%(lineno)d]-[%(levelname)s]: %(message)s'
    formmater = logging.Formatter(FORMAT)
    logging.basicConfig(format=FORMAT,
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


class Agent:

    def __init__(self, 
                 env_id: str = 'CartPole-v1',
                 discount_factor: float = 0.98,
                 hidden_dim: int = 128,
                 learning_rate: float = 0.0002,
                 stop_on_reward: float = 500,
                 num_episodes: int = 1000,
                 model_path: str = './model/reinforce/runs.pt',
                 gradient_clip: float = 1.0,
                 device: str = 'cpu',
                 use_entropy_regularization: bool = False,
                 log_interval: int = 100,
                 ):
        self.env_id = env_id
        self.discount_factor = discount_factor

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.stop_on_reward = stop_on_reward

        self.num_episodes = num_episodes

        self.model_path = model_path

        if not os.path.exists(model_path_path := os.path.dirname(self.model_path)):
            os.makedirs(model_path_path)

        self.gradient_clip = gradient_clip
        self.recent_rewards = deque(maxlen=100)
        self.device = device
        self.use_entropy_regularization = use_entropy_regularization
        self.log_interval = log_interval

    def compute_loss(self, rewards: torch.Tensor, log_probs: torch.Tensor):
        
        n = len(rewards)
        discounts = self.discount_factor ** torch.arange(n, device=rewards.device)
        returns = torch.flip(
            torch.cumsum(torch.flip(rewards * discounts, dims=[0]), dim=0),
            dims=[0]
        ) / discounts
        # todo(haleng): verify the \gamma^{t-1} in the summation on page 115 in drl
        loss = (-log_probs * returns).sum()
        return returns, loss


    def run(self, is_training: bool = False, render: bool = False):
        
        env = gym.make(self.env_id, render_mode='human' if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_net = Policy(num_states=num_states, num_actions=num_actions, hidden_dim=self.hidden_dim)
        policy_net.to(self.device)

        log_interval = self.log_interval
        
        if is_training:
            logging.info(f'start training env: {self.env_id}')
            optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
            best_reward = -math.inf
        else:
            logging.info(f'start evaluating env: {self.env_id}')
            policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            policy_net.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            done = False
            episode_reward = 0
            log_probs = []
            rewards = []
            entropy_loss = []
            while (not done):
                action_probs = policy_net(state.unsqueeze(0))

                if is_training:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                    entropy = dist.entropy().sum()
                    entropy_loss.append(entropy)
                else:
                    action = torch.argmax(action_probs, dim=-1)
                
                new_state, reward, terminated, truncated, info = env.step(action=action.item())

                episode_reward += reward
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=self.device)

                rewards.append(reward)

                state = new_state
                done = terminated or truncated
            self.recent_rewards.append(episode_reward)

            if is_training:

                if episode_reward > best_reward:
                    
                    increase_rate = (episode_reward - best_reward) / best_reward
                    logging.info(f'Episode {episode}, New best reward: {episode_reward:0.1f}, ({increase_rate:.2%})')
                    best_reward = episode_reward
                    torch.save(policy_net.state_dict(), self.model_path)
                entropy_loss = torch.stack(entropy_loss, dim=-1).sum()
                loss = self.optimize(optimizer, rewards, log_probs, entropy_loss)
                if (episode + 1) % log_interval == 0:
                    avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else episode_reward
                    logging.info(f'Episode {episode}, best reward: {best_reward}, reward: {episode_reward}, avg({log_interval}): {avg_reward}, loss: {loss:.3f}')
                    self.recent_rewards = []
            else:
                logging.info(f'Episode {episode}, reward: {episode_reward}')
        env.close()

    def optimize(self, optimizer: optim.Optimizer, rewards: torch.Tensor, log_probs: torch.Tensor, entropy_loss: torch.Tensor):
        rewards = torch.stack(rewards)
        log_probs = torch.stack(log_probs)
        
        _, loss = self.compute_loss(rewards, log_probs)
        
        if self.use_entropy_regularization:
            loss += 0.01 * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=self.gradient_clip)
        optimizer.step()
        return loss.item()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    initialize('./logs/reinforce_train.log')
    logging.info(f'device: {device}')

    agent = Agent(env_id='CartPole-v1', gradient_clip=0., use_entropy_regularization=True)
    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)


if __name__ == '__main__':
    main()