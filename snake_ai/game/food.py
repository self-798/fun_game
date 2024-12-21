# snake_ai/game/food.py

from snake_ai.game.constants import FOOD_COLOR

class Food:
    def __init__(self, position, color=FOOD_COLOR):
        self.position = position
        self.color = color

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position