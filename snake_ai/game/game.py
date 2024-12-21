# snake_ai/game/game.py

import pygame
import time
from snake import Snake
from food import Food
from utils import get_random_food_position
from constants import GRID_WIDTH, GRID_HEIGHT, GRID_SIZE, FPS, BACKGROUND_COLOR, REWARD_FOOD, REWARD_STEP, REWARD_DEATH, ACTIONS

class Game:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((GRID_SIZE * GRID_WIDTH, GRID_SIZE * GRID_HEIGHT))
        pygame.display.set_caption("Snake AI")
        self.snake = Snake()
        self.food = Food(get_random_food_position(self.snake.get_positions(), GRID_WIDTH, GRID_HEIGHT))
        self.score = 0
        self.game_over = False
        self.reward = 0

    def _handle_input(self, action):
        """处理智能体的动作"""
        self.snake.set_direction(ACTIONS[action])

    def _update(self):
        """更新游戏状态"""
        if not self.game_over:
            self.reward = REWARD_STEP
            self.snake.move()
            if self.snake.check_collision():
                self.game_over = True
                self.reward = REWARD_DEATH
                return True
            if self.snake.get_head_position() == self.food.get_position():
                self.snake.grow()
                self.score += 1
                self.food.set_position(get_random_food_position(self.snake.get_positions(), GRID_WIDTH, GRID_HEIGHT))
                self.reward = REWARD_FOOD
            return False

    def _draw(self):
        """绘制游戏画面"""
        self.screen.fill(BACKGROUND_COLOR)
        for pos in self.snake.get_positions():
            pygame.draw.rect(self.screen, self.snake.color, (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, self.food.color, (self.food.get_position()[0] * GRID_SIZE, self.food.get_position()[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.flip()

    def reset(self):
        """重置游戏状态"""
        self.snake = Snake()
        self.food = Food(get_random_food_position(self.snake.get_positions(), GRID_WIDTH, GRID_HEIGHT))
        self.score = 0
        self.game_over = False
        return self.get_state()

    def get_state(self):
        """获取游戏状态"""
        head = self.snake.get_head_position()
        food = self.food.get_position()
        snake_body = self.snake.get_positions()
        state = (head[0], head[1], food[0], food[1])
        state += tuple(x for position in snake_body for x in position)
        return state

    def step(self, action):
        """执行一步游戏"""
        self._handle_input(action)
        done = self._update()
        self._draw()
        self.clock.tick(FPS)
        return self.get_state(), self.reward, done

    def close(self):
        pygame.quit()