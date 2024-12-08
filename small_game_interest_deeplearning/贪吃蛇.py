import random
import pickle
import threading
import time

# 定义 Q-learning 参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.5  # 探索率
q_table_filename = 'trained_model.pkl'

# 定义网格大小和蛇的初始位置
grid_size = 20
tile_count = 20  # 每个游戏区域的网格数
cols, rows = 4, 4
ai_count = cols * rows  # 16 个 AI

# 加载模型
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        print(f"成功加载 Q 表: {filename}")
        return q_table
    except FileNotFoundError:
        print(f"未找到 Q 表文件: {filename}，将创建新的 Q 表。")
        return {}

# 保存模型
def save_model(q_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Q 表已保存到 {filename}")

# 获取曼哈顿距离
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# 获取奖励函数
def get_reward(next_state, snake, food):
    if next_state == food:
        return 50  # 吃到食物奖励调整为 50

    x, y = next_state
    collision = x < 0 or x >= tile_count or y < 0 or y >= tile_count or next_state in snake
    if collision:
        return -2000  # 撞到墙壁或自己，惩罚调整为 -2000

    # 计算蛇头与食物的曼哈顿距离变化
    current_distance = manhattan_distance(snake[0], food)
    new_distance = manhattan_distance(next_state, food)
    if new_distance < current_distance:
        return 0.1  # 接近食物，给予小额正奖励
    elif new_distance > current_distance:
        return -0.1  # 远离食物，给予小额负奖励
    else:
        return -0.05  # 保持距离，给予微小负奖励

# 获取下一个状态（仅计算，不修改蛇的状态）
def get_next_state(state, action):
    x, y = state
    if action == 0:  # 上
        return (x, y - 1)
    elif action == 1:  # 右
        return (x + 1, y)
    elif action == 2:  # 下
        return (x, y + 1)
    elif action == 3:  # 左
        return (x - 1, y)
    else:
        return state

# 更新 Q 值（使用锁确保线程安全）
q_table_lock = threading.Lock()

def update_q_table(state, action, reward, next_state, q_table, alpha, gamma):
    with q_table_lock:
        current_q = q_table.get(state, [0, 0, 0, 0])[action]
        max_future_q = max(q_table.get(next_state, [0, 0, 0, 0]))
        updated_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table.setdefault(state, [0, 0, 0, 0])[action] = updated_q

# ε-greedy 策略选择动作，防止蛇头往上一个蛇身的方向移动
def choose_action(state, q_table, epsilon, last_action):
    if random.random() < epsilon or state not in q_table:
        return random.choice([0, 1, 2, 3])  # 随机选择一个动作
    
    with q_table_lock:
        q_values = q_table[state].copy()
    
    # 防止蛇头往上一个蛇身的方向移动
    opposite_actions = {0:2, 1:3, 2:0, 3:1}
    if last_action is not None:
        opposite_action = opposite_actions[last_action]
        q_values[opposite_action] = -float('inf')  # 禁止相反方向移动
    
    max_q = max(q_values)
    if q_values.count(max_q) > 1:
        # 多个最大 Q 值，随机选择其中一个
        actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(actions)
    else:
        return q_values.index(max_q)

# 生成新的食物位置
def generate_food(snake):
    attempts = 0
    while True:
        new_food = (random.randint(0, tile_count - 1), random.randint(0, tile_count - 1))
        if new_food not in snake:
            return new_food
        attempts += 1
        if attempts > 100:
            print("无法生成新的食物位置，可能蛇已填满整个区域。")
            return snake[0]  # 防止无限循环

# AI 玩家类
class AIPlayer(threading.Thread):
    def __init__(self, ai_id, q_table, epsilon):
        threading.Thread.__init__(self)
        self.ai_id = ai_id
        self.q_table = q_table  # 共享 Q 表
        self.reset()
        self.epsilon = epsilon
        self.running = True
        self.lock = threading.Lock()

    def reset(self):
        self.snake = [(tile_count // 2, tile_count // 2)]
        self.food = generate_food(self.snake)
        self.score = 0
        self.done = False
        self.state = self.snake[0]
        self.last_action = None

    def run(self):
        while self.running:
            if not self.done:
                self.step()
            else:
                self.reset()
            # 可以根据需要调整步频
            time.sleep(0.001)  # 1ms 间隔，避免占用过多 CPU

    def step(self):
        action = choose_action(self.state, self.q_table, self.epsilon, self.last_action)
        next_state = get_next_state(self.state, action)
        reward = get_reward(next_state, self.snake, self.food)

        if reward == 50:
            self.snake.insert(0, next_state)  # 吃到食物，增长
            self.food = generate_food(self.snake)
            self.score += 10
        elif reward == -2000:
            self.done = True
        else:
            self.snake.insert(0, next_state)
            self.snake.pop()  # 移动

        # 更新 Q 表
        update_q_table(self.state, action, reward, next_state, self.q_table, alpha, gamma)
        self.state = next_state
        self.last_action = action

    def stop(self):
        self.running = False

# 初始化 Q 表
q_table = load_model(q_table_filename)

# 创建所有 AI 玩家
ai_players = [AIPlayer(ai_id, q_table, epsilon) for ai_id in range(ai_count)]

# 启动所有 AI 玩家线程
for ai in ai_players:
    ai.start()

# 主训练循环
try:
    frame = 0
    save_interval = 1000  # 每 1000 帧保存一次
    report_interval = 1000  # 每 1000 帧报告一次平均分数

    while True:
        frame += 1

        if frame % report_interval == 0:
            total_score = sum(ai.score for ai in ai_players)
            average_score = total_score / ai_count
            print(f"第 {frame} 帧，平均得分: {average_score:.2f}")
        
        if frame % save_interval == 0:
            save_model(q_table, q_table_filename)

        # 可选：动态调整 ε 探索率
        if frame % 10000 == 0 and epsilon > 0.01:
            epsilon *= 0.99
            print(f"动态调整 ε 探索率: {epsilon:.4f}")
            for ai in ai_players:
                ai.epsilon = epsilon

        # 控制训练速度
        # time.sleep(0.001)  # 1ms 间隔，避免占用过多 CPU

except KeyboardInterrupt:
    print("训练中断，保存模型。")
    save_model(q_table, q_table_filename)

finally:
    # 停止所有 AI 玩家线程
    for ai in ai_players:
        ai.stop()
    for ai in ai_players:
        ai.join()

    # 最终保存模型
    save_model(q_table, q_table_filename)
    print("训练完成。")
