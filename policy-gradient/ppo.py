"""
ppo: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py

ppo-for-beginers: https://github.com/ericyangyu/PPO-for-Beginners

hands-on-RL: https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95

"""
from __future__ import annotations
import argparse
import itertools
import logging
import math
import os

import gymnasium as gym
from gymnasium import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state', 'log_prob'])


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
                 env: Env,
                 env_id: str = 'CartPole-v1',
                 gamma: float = 0.98,
                 lammbda: float = 0.9, 
                 epsilon: float = 0.1, 
                 hidden_dim: int = 128,
                 learning_rate: float = 0.0005,
                 stop_on_reward: float = 500,
                 num_episodes: int = 1000,
                 model_path: str = './model/ppo/runs',
                 gradient_clip: float = 1.0,
                 device: str = 'cpu',
                 use_entropy_regularization: bool = False,
                 log_interval: int = 20,
                 epochs: int = 10,
                 ):
        self.env = env
        self.gamma = gamma
        self.lammbda = lammbda
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.device = device
        self.learning_rate = learning_rate

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.actor = Actor(num_states=num_states, num_actions=num_actions, hidden_dim=self.hidden_dim)
        self.actor.to(device=device)
        self.critic = Critic(num_states=num_states, hidden_dim=self.hidden_dim)
        self.critic.to(device=device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # self.optimizer = optim.Adam(itertools.chain(self.actor.parameters()), self.critic.parameters(), lr=self.learning_rate)
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.actor_model_path = os.path.join(self.model_path, 'actor.pt')
        self.critic_model_path = os.path.join(self.model_path, 'critic.pt')

        self.gradient_clip = gradient_clip
        self.recent_rewards = []
        self.use_entropy_regularization = use_entropy_regularization
        self.log_interval = log_interval
        self.epochs = epochs

        self.transitions = []

    def compute_returns(self, rewards: torch.Tensor):
        
        n = len(rewards)
        discounts = self.discount_factor ** torch.arange(n, device=rewards.device)
        returns = torch.flip(
            torch.cumsum(torch.flip(rewards * discounts, dims=[0]), dim=0),
            dims=[0]
        ) / discounts
        return returns
    
    def compute_gae(self, x: torch.FloatTensor):
        n = x.shape[0]
        x = x * (self.gamma * self.lammbda) ** torch.arange(n, device=self.device).view(n, -1)
        results = torch.flip(torch.cumsum(torch.flip(x, [0],), 0), [0])
        return results
    
    def to_tensor(self, x, shape: int | None = None):
        return torch.FloatTensor(x, device=self.device)
    
    def train(self, ):

        env: Env = self.env
        best_reward = -math.inf
        log_interval = self.log_interval

        for episode in itertools.count(1):
            state, _ = env.reset()
            
            done = False
            episode_reward = 0
            rewards = []
            while not done:
                action_probs = self.actor(self.to_tensor(state))
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                new_state, reward, terminated, truncated, info = env.step(action=action.item())
                done = terminated or truncated
                episode_reward += reward
                rewards.append(reward)
                self.transitions.append((state, action, reward/100, log_prob, new_state, done))
                state = new_state

            loss = self.update()
            if episode_reward > best_reward:
                increase_rate = (episode_reward - best_reward) / best_reward
                logging.info(f'Episode {episode}, New best reward: {episode_reward:0.1f}, ({increase_rate:.2%})')
                torch.save(self.actor.state_dict(), self.actor_model_path)
                best_reward = episode_reward
            if episode % log_interval == 0:
                avg_reward = np.mean(rewards)
                logging.info(f'Episode {episode}, best reward: {best_reward}, reward: {episode_reward}, avg({log_interval}): {avg_reward}, loss: {loss:.3f}')
        
    def eval(self, ):
        env: Env = self.env
        self.actor.load_state_dict(torch.load(self.actor_model_path, map_location=self.device))

        for episode in itertools.count(1):
            state, _ = env.reset()
            
            done = False
            episode_reward = 0
            rewards = []
            while not done:
                action_probs = self.actor(self.to_tensor(state))
                action = torch.argmax(action_probs, dim=-1)
                new_state, reward, terminated, truncated, info = env.step(action=action.item())
                done = terminated or truncated
                episode_reward += reward
                rewards.append(reward)
                state = new_state

            logging.info(f'Episode {episode}, reward: {episode_reward}')

    def batch(self):
        states, actions, rewards, log_probs, new_states, dones = zip(*self.transitions)
        states = torch.tensor(np.stack(states), device=self.device)
        actions = torch.tensor(np.stack(actions), device=self.device).view(-1, 1)
        rewards = torch.tensor(np.stack(rewards), device=self.device).view(-1, 1)
        log_probs = torch.stack(log_probs).view(-1, 1)
        new_states = torch.tensor(np.stack(new_states), device=self.device)
        dones = torch.tensor(1 - np.stack(dones, dtype=np.float32), device=self.device).view(-1, 1)

        return states, actions, rewards, log_probs, new_states, dones
    
    def update(self, ):
        states, actions, rewards, log_probs, new_states, dones = self.batch()

        for _ in range(self.epochs):

            td_target = rewards + self.gamma * self.critic(new_states) * dones
            delta: torch.FloatTensor = td_target - self.critic(states)
            delta = delta.detach()

            # advantages = self.compute_gae(delta)
            advantages = delta
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            curr_action_probs: torch.Tensor = self.actor(states)
            curr_log_probs = torch.log(curr_action_probs.gather(-1, actions))

            ratio = torch.exp(curr_log_probs - log_probs.detach())
            loss1 = ratio * advantages
            loss2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(loss1, loss2).mean()

            curr_values = self.critic(states)
            critic_loss = F.smooth_l1_loss(curr_values, td_target.detach())
            loss = actor_loss + critic_loss

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            # loss.backward()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        return loss.item()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    initialize('./logs/ppo.log')
    logging.info(f'device: {device}')

    env_id = 'CartPole-v1'
    env = gym.make(env_id, render_mode = 'human' if not args.train else None)
    agent = Agent(env, )
    if args.train:
        agent.train()
    else:
        agent.eval()


if __name__ == '__main__':
    main()