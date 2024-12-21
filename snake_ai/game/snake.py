# snake_ai/game/snake.py

from constants import UP, SNAKE_START_LENGTH, SNAKE_COLOR, SNAKE_START_POSITION
class Snake:
    def __init__(self, start_position=SNAKE_START_POSITION, start_length=SNAKE_START_LENGTH, color=SNAKE_COLOR):
        self.positions = [start_position]
        self.color = color
        self.length = start_length
        self.direction = UP

    def move(self):
        head = self.positions[0]
        dx, dy = self.direction
        new_head = (head[0] + dx, head[1] + dy)
        self.positions.insert(0, new_head)
        if len(self.positions) > self.length:
            self.positions.pop()

    def get_head_position(self):
        return self.positions[0]

    def get_positions(self):
        return self.positions

    def set_direction(self, direction):
        self.direction = direction

    def grow(self):
        self.length += 1

    def check_collision(self):
        head = self.get_head_position()
        # 检查是否撞到边界
        if head[0] < 0 or head[0] >= 20 or head[1] < 0 or head[1] >= 20:
            return True
        # 检查是否撞到自身
        if head in self.positions[1:]:
            return True
        return False