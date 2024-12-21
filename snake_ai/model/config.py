# snake_ai/model/config.py

# 模型参数
STATE_SIZE = 120 # 状态大小
ACTION_SIZE = 4  # 动作空间大小
HIDDEN_SIZE = 128   # 隐藏层大小

# 训练参数
BATCH_SIZE = 32
GAMMA = 0.99      # 折扣因子
LR = 0.001       # 学习率
EPSILON_START = 1.0  # 初始探索率
EPSILON_END = 0.01    # 最终探索率
EPSILON_DECAY = 0.0001  # 探索率衰减率
TARGET_UPDATE_FREQ = 10  # 目标网络更新频率
REPLAY_MEMORY_SIZE = 10000 # 回放内存大小