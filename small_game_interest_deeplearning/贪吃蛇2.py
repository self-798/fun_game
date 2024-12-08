import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import random
import numpy as np

####################################################
# 环境定义
####################################################
class SnakeEnv:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size//2, self.grid_size//2)]
        self.direction = (0,1)  # 初始向右
        self.spawn_food()
        self.score = 0
        return self._get_state()

    def spawn_food(self):
        while True:
            r = random.randint(0, self.grid_size-1)
            c = random.randint(0, self.grid_size-1)
            if (r,c) not in self.snake:
                self.food = (r,c)
                break

    def step(self, action):
        # action: 0=上,1=下,2=左,3=右
        if action == 0:
            self.direction = (-1,0)
        elif action == 1:
            self.direction = (1,0)
        elif action == 2:
            self.direction = (0,-1)
        elif action == 3:
            self.direction = (0,1)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        reward = 0
        done = False

        # 撞墙或撞到自己
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size) or (new_head in self.snake):
            reward = -100
            done = True
            return self._get_state(), reward, done

        # 移动蛇身
        self.snake.insert(0, new_head)

        # 吃果实
        if new_head == self.food:
            reward = 50
            self.score += 50
            self.spawn_food()
            done = False
        else:
            # 没有果实则移动尾巴
            self.snake.pop()
            # 存活奖励
            reward = 0.1
            done = False
            
        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for (r,c) in self.snake:
            state[r,c] = 0.5  # 蛇
        fr,fc = self.food
        state[fr,fc] = 1.0  # 食物
        return state

####################################################
# Q网络定义 (改进的CNN架构)
####################################################
class QNet(nn.Module):
    def __init__(self, grid_size=32, num_actions=4):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32->16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16->8
        )
        # After two pools: (32,8,8) => 32*8*8=2048
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

####################################################
# 训练超参数
####################################################
NUM_EPISODES = 1000
MAX_STEPS = 500
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999

def main():
    device = torch_directml.device()
    env = SnakeEnv(grid_size=32)
    num_actions = 4

    qnet = QNet(grid_size=32, num_actions=num_actions).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        for step in range(MAX_STEPS):
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            # epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, num_actions-1)
            else:
                with torch.no_grad():
                    q_values = qnet(s)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            s_next = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_next = qnet(s_next).max(dim=1)[0].item()
            target = reward + (GAMMA * q_next if not done else 0.0)
            q_current = qnet(s)[0,action]

            loss = loss_fn(q_current, torch.tensor([target], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

        # epsilon衰减
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Episode {episode+1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # 保存模型
    torch.save(qnet.state_dict(), "snake_qnet_32x32.pth")
    print("Model saved as snake_qnet_32x32.pth")

if __name__ == "__main__":
    main()
