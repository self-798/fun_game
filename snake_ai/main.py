import pygame
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import torch_directml


# Constants
GRID_SIZE = 20
SNAKE_SPEED = 10

# Utils
def generate_random_position(grid_size):
    x = random.randint(0, grid_size - 1)
    y = random.randint(0, grid_size - 1)
    return (x, y)

# Food
class Food:
    def __init__(self):
      self.position = generate_random_position(GRID_SIZE)
    
    def new_position(self):
        self.position = generate_random_position(GRID_SIZE)

# Snake
class Snake:
    def __init__(self):
      self.body = [(GRID_SIZE//2, GRID_SIZE//2)] #初始位置在中心
      self.direction = (1,0) # 初始方向是向右
      
    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        # 边界处理 (使用 % 来实现从另一边穿出)
        new_head = (new_head[0] % GRID_SIZE, new_head[1] % GRID_SIZE)
        self.body.insert(0, new_head)
        self.body.pop() 
    
    def grow(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        # 边界处理
        new_head = (new_head[0] % GRID_SIZE, new_head[1] % GRID_SIZE)
        self.body.insert(0,new_head)
    
    def change_direction(self, direction):
        if (direction[0] * -1, direction[1] * -1) != self.direction: #不能反向移动
            self.direction = direction
    
    def is_collision(self):
        head = self.body[0]
        if not (0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE) : #撞墙
            return True
        if head in self.body[1:]: #撞自己
            return True
        return False


# Game
class Game:
    def __init__(self):
        pygame.init()
        self.grid_size = GRID_SIZE
        self.cell_size = 20
        self.screen_width = self.grid_size * self.cell_size
        self.screen_height = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
        
    def game_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snake.change_direction((0,-1))
                    if event.key == pygame.K_DOWN:
                        self.snake.change_direction((0,1))
                    if event.key == pygame.K_LEFT:
                        self.snake.change_direction((-1,0))
                    if event.key == pygame.K_RIGHT:
                        self.snake.change_direction((1,0))
                    
            self.snake.move()
            if self.snake.body[0] == self.food.position:
               self.snake.grow()
               self.food.new_position()
            
            if self.snake.is_collision():
                print("Game Over")
                return

            self.screen.fill((0,0,0)) #背景黑色
            
            for part in self.snake.body:
               pygame.draw.rect(self.screen, (0,255,0), (part[0]*self.cell_size, part[1]*self.cell_size, self.cell_size, self.cell_size))
            
            pygame.draw.rect(self.screen,(255,0,0), (self.food.position[0]*self.cell_size, self.food.position[1]*self.cell_size, self.cell_size, self.cell_size))
            
            pygame.display.flip()
            self.clock.tick(SNAKE_SPEED)


# Model Config
LEARNING_RATE = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.999995
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# Replay Memory
class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Agent
class Agent:
    def __init__(self,state_size, action_size, device):
      self.state_size = state_size
      self.action_size = action_size
      self.policy_net = DQN(state_size, 128, action_size).to(device)
      self.target_net = DQN(state_size, 128, action_size).to(device)
      self.target_net.load_state_dict(self.policy_net.state_dict())
      self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
      self.memory = ReplayMemory(MEMORY_SIZE)
      self.gamma = GAMMA
      self.epsilon = EPSILON_START
      self.device = device
    
    def select_action(self, state):
       if random.random() > self.epsilon:
           with torch.no_grad():
               state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
               q_values = self.policy_net(state)
               return np.argmax(q_values.cpu().detach().numpy())
       else:
           return random.choice(range(self.action_size))
    
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon*EPSILON_DECAY)
    
    def learn(self,batch_size):
        if len(self.memory) < batch_size:
           return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        states = torch.FloatTensor(states).to(self.device)
        actions = np.array(actions).reshape(-1,1)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = np.array(dones)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算Q值
        q_values = self.policy_net(states).gather(1, actions).squeeze() #gather() 从指定维度获取值

        # 计算下一个状态的Q值
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.gamma*next_q_values *(1-dones)

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Trainer
NUM_EPISODES = 1000
BATCH_SIZE = 64
TARGET_UPDATE = 10
def process_state(game):
    state = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=int)
    head = game.snake.body[0]
    state[0, head[0], head[1]] = 1 # 蛇头位置
    for part in game.snake.body[1:]: # 蛇身
        state[1, part[0], part[1]] = 1
    state[2,game.food.position[0],game.food.position[1]] = 1 #食物的位置
    return state.flatten()

def train():
    device = torch_directml.device() if torch_directml.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using DirectML for training.")
    game = Game()
    state_size = 4*GRID_SIZE*GRID_SIZE
    action_size = 4
    agent = Agent(state_size,action_size, device)
    
    for episode in range(NUM_EPISODES):
        game = Game()
        state = process_state(game)
        done = False
        total_reward = 0
        steps = 0
        while not done:
           if steps > 500:
              done = True
              reward = -10
           action = agent.select_action(state)
           
           if action == 0:
                game.snake.change_direction((0,-1))
           if action == 1:
                game.snake.change_direction((0,1))
           if action == 2:
                game.snake.change_direction((-1,0))
           if action == 3:
                game.snake.change_direction((1,0))
           game.snake.move()
           
           reward = 0
           
           if game.snake.is_collision():
                done = True
                reward = -10
           elif game.snake.body[0] == game.food.position:
               game.snake.grow()
               game.food.new_position()
               reward = 10
           
           next_state = process_state(game)
           agent.memory.push(state,action, reward, next_state, done)
           agent.learn(BATCH_SIZE)
           agent.update_epsilon()

           state = next_state
           total_reward += reward
           steps +=1
        
        if episode % TARGET_UPDATE == 0:
              agent.update_target_net()
              print("Target network updated.")
        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon}")

# Eval
def eval_model():
    game = Game()
    game.game_loop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Snake AI Training and Evaluation")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or eval")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval_model()