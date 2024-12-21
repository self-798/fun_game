# snake_ai/train/eval.py

import torch
from game.game import Game
from model.agent import Agent
from model.config import STATE_SIZE, ACTION_SIZE
import time

def evaluate(model_path, num_episodes=10):
    game = Game()
    agent = Agent(STATE_SIZE, ACTION_SIZE)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    total_rewards = []

    for episode in range(num_episodes):
        state = game.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            state = next_state
            total_reward += reward
            time.sleep(0.1)
        total_rewards.append(total_reward)

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    game.close()

if __name__ == '__main__':
    evaluate('snake_ai_model.pth', num_episodes=10)