"""
a2c: https://github.com/seungeunrho/minimalRL/blob/master/actor_critic.py

pay attention to rollout
"""
from __future__ import annotations
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

from collections import namedtuple

Item = namedtuple('Item', ['state', 'action', 'reward', 'done', 'next_state', 'log_prob'])


class Actor(nn.Module):

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


class Critic(nn.Module):
    def __init__(self,  num_states: int, hidden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


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
                 learning_rate: float = 0.002,
                 stop_on_reward: float = 500,
                 num_episodes: int = 1000,
                 model_path: str = './model/ac/runs.pt',
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
        self.recent_rewards = []
        self.device = device
        self.use_entropy_regularization = use_entropy_regularization
        self.log_interval = log_interval

        self.actor_net: nn.Module = None
        self.critic_net: nn.Module = None
        self.actor_optimizer: optim.Optimizer = None
        self.critic_optimizer: optim.Optimizer = None

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

    def compute_returns(self, rewards: torch.Tensor):
        
        n = len(rewards)
        discounts = self.discount_factor ** torch.arange(n, device=rewards.device)
        returns = torch.flip(
            torch.cumsum(torch.flip(rewards * discounts, dims=[0]), dim=0),
            dims=[0]
        ) / discounts
        return returns

    def run(self, is_training: bool = False, render: bool = False):
        
        env = gym.make(self.env_id, render_mode='human' if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        actor_net = Actor(num_states=num_states, num_actions=num_actions, hidden_dim=self.hidden_dim)
        actor_net.to(self.device)

        critic_net = Critic(num_states=num_states,  hidden_dim=self.hidden_dim)
        critic_net.to(self.device)

        log_interval = self.log_interval
        
        if is_training:
            logging.info(f'start training env: {self.env_id}')
            actor_optimizer = optim.Adam(actor_net.parameters(), lr=self.learning_rate)
            critic_optimizer = optim.Adam(critic_net.parameters(), lr=self.learning_rate)
            best_reward = -math.inf
        else:
            logging.info(f'start evaluating env: {self.env_id}')
            actor_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            actor_net.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            done = False
            episode_reward = 0
            log_probs = []
            rewards = []
            recent_rewards = []
            values = []
            next_values = []
            dones = []
            while (not done):
                action_probs = actor_net(state)
                if is_training:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                    critic_value = critic_net(state)
                    values.append(critic_value)
                    # data.append(Item(state, action, reward, terminated, new_state, value=critic_value, log_prob=log_prob))
                else:
                    action = torch.argmax(action_probs, dim=-1)
                
                new_state, reward, terminated, truncated, info = env.step(action=action.item())

                recent_rewards.append(reward)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                rewards.append(reward)
                with torch.no_grad():
                    next_value = critic_net(new_state)
                    next_values.append(next_value)
                episode_reward += reward
                
                state = new_state
                done = terminated or truncated
                dones.append(torch.tensor(terminated, dtype=torch.float32, device=self.device))

            if is_training:

                if episode_reward > best_reward:
                    
                    increase_rate = (episode_reward - best_reward) / best_reward
                    logging.info(f'Episode {episode}, New best reward: {episode_reward:0.1f}, ({increase_rate:.2%})')
                    best_reward = episode_reward
                    torch.save(actor_net.state_dict(), self.model_path)
                entropy_loss = 0

                log_probs = torch.stack(log_probs)
                rewards = torch.stack(rewards)
                values = torch.stack(values)
                next_values = torch.stack(next_values)
                dones = torch.stack(dones)
                loss = self.optimize(actor_optimizer, critic_optimizer, rewards = rewards, values=values, log_probs=log_probs, next_values=next_values,
                                     dones=dones)
                if (episode + 1) % log_interval == 0:
                    avg_reward = np.mean(recent_rewards)
                    logging.info(f'Episode {episode}, best reward: {best_reward}, reward: {episode_reward}, avg({log_interval}): {avg_reward}, loss: {loss:.3f}')
            else:
                logging.info(f'Episode {episode}, reward: {episode_reward}')
        env.close()

    def optimize(self, 
                 actor_optimizer: optim.Optimizer, 
                 critic_optimizer: optim.Optimizer, 
                 rewards: torch.Tensor,
                 values: torch.Tensor,
                 log_probs: torch.Tensor,
                 next_values: torch.Tensor,
                 dones: torch.Tensor,
                 ):
        # shape = values.shape
        # rewards = torch.reshape(rewards, shape)
        # values = torch.reshape(values, shape)
        # log_probs = torch.reshape(log_probs, shape)
        # next_values = torch.reshape(next_values, shape)
        # dones = torch.reshape(dones, shape)
        
        # rewards will be diminished by 100
        # https://github.com/seungeunrho/minimalRL/blob/master/actor_critic.py#L43
        rewards = rewards.view(-1) / 100
        values = values.view(-1)
        log_probs = log_probs.view(-1)
        next_values = next_values.view(-1)
        dones = dones.view(-1)
        td_target = (rewards + (1 - dones) * self.discount_factor * next_values).detach()
        ## actor loss
        advantages = (td_target - values).detach()
        actor_loss = (-log_probs * advantages).sum()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        ## critic loss
        critic_loss = F.smooth_l1_loss(values, td_target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        return actor_loss.item()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    initialize('./logs/ac.log')
    logging.info(f'device: {device}')

    agent = Agent(env_id='CartPole-v1', gradient_clip=0., use_entropy_regularization=False)
    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)


if __name__ == '__main__':
    main()