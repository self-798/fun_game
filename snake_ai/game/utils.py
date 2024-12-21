# snake_ai/game/utils.py

import random

def get_random_food_position(snake_positions, grid_width, grid_height):
    """生成随机的食物位置，避开蛇身"""
    while True:
        x = random.randint(0, grid_width - 1)
        y = random.randint(0, grid_height - 1)
        if (x, y) not in snake_positions:
            return x, y