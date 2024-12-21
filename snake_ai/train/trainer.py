# snake_ai/train/trainer.py

import torch
from game.game import Game
from model.agent import Agent
from model.config import STATE_SIZE, ACTION_SIZE

def train(num_episodes=10000):
    game = Game()
    agent = Agent(STATE_SIZE, ACTION_SIZE)
    total_rewards = []

    for episode in range(num_episodes):
        state = game.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = sum(total_rewards[-100:])/100
            print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward (Last 100): {avg_reward}, Epsilon: {agent.epsilon:.4f}")

    game.close()
    return agent

if __name__ == '__main__':
    trained_agent = train(num_episodes=1000)
    # 保存模型
    torch.save(trained_agent.policy_net.state_dict(), 'snake_ai_model.pth')