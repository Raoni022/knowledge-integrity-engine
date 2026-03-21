from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch: List[Transition] = random.sample(list(self.buffer), batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_capacity: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        current_q = self.q_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
