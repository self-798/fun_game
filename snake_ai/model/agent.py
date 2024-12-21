# snake_ai/model/agent.py

import torch
import torch.optim as optim
import random
from dqn_model import DQN
from memory import ReplayMemory
from config import BATCH_SIZE, GAMMA, LR, EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE_FREQ, REPLAY_MEMORY_SIZE, ACTION_SIZE
import numpy as np

try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

        if HAS_DML:
            self.device = torch_directml.device()
        else:
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()

    def select_action(self, state):
        """选择动作"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.action_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """从回放内存中学习"""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)
        self.steps_done += 1

        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())