# snake_ai/game/constants.py

# 游戏界面尺寸
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_SIZE * GRID_WIDTH
WINDOW_HEIGHT = GRID_SIZE * GRID_HEIGHT

# 贪吃蛇初始位置和大小
SNAKE_START_LENGTH = 3
SNAKE_START_POSITION = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
SNAKE_COLOR = (0, 255, 0)  # 绿色

# 食物颜色
FOOD_COLOR = (255, 0, 0)  # 红色

# 背景颜色
BACKGROUND_COLOR = (0, 0, 0) #黑色

# 移动方向
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# 动作空间
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# 游戏帧率
FPS = 10  # 你可以根据需要调整帧率

# 奖励
REWARD_FOOD = 10
REWARD_STEP = -0.1
REWARD_DEATH = -10