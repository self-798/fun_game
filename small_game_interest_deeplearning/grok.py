import pygame
import random
import sys

class SnakeGame:
    def __init__(self):
        self.screen_width = 640
        self.screen_height = 480
        self.snake_size = 20
        self.snake_speed = 5
        self.score = 0

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.snake_x = random.randint(0, (self.screen_width - self.snake_size) // self.snake_size)
        self.snake_y = random.randint(0, (self.screen_height - self.snake_size) // self.snake_size)
        self.direction_x = 1
        self.direction_y = 0

    def draw_snake(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (self.snake_x * self.snake_size, self.snake_y * self.snake_size, self.snake_size, self.snake_size))

    def update_snake_position(self):
        new_head_position_x = self.snake_x + self.direction_x
        new_head_position_y = self.snake_y + self.direction_y

        if (new_head_position_x < 0 or new_head_position_x > (self.screen_width - self.snake_size) // self.snake_size) or \
           (new_head_position_y < 0 or new_head_position_y > (self.screen_height - self.snake_size) // self.snake_size):
            pygame.quit()
            sys.exit()

        self.snake_x = new_head_position_x
        self.snake_y = new_head_position_y

    def check_collision_with_food(self):
        food_x = random.randint(0, (self.screen_width - self.snake_size) // self.snake_size)
        food_y = random.randint(0, (self.screen_height - self.snake_size) // self.snake_size)
        if [food_x * self.snake_size, food_y * self.snake_size] in [(i[0], i[1]) for i in [(j, k) for j in range(self.screen_width // self.snake_size) for k in range(self.screen_height // self.snake_size)]]:
            return False

        return True

    def update_score(self):
        self.score += 1
        print("分数：", self.score)

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction_y = -1
                self.direction_x = 0
            elif keys[pygame.K_DOWN]:
                self.direction_y = 1
                self.direction_x = 0
            elif keys[pygame.K_LEFT]:
                self.direction_x = -1
                self.direction_y = 0
            elif keys[pygame.K_RIGHT]:
                self.direction_x = 1
                self.direction_y = 0

            self.update_snake_position()

            if not self.check_collision_with_food():
                print("游戏结束!")
                pygame.quit()
                sys.exit()

            self.screen.fill((0, 0, 0))
            self.draw_snake()

            font = pygame.font.SysFont("Arial", 24)
            text = font.render("Score: " + str(self.score), True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

            pygame.display.update()
            self.clock.tick(6)

if __name__ == "__main__":
    game = SnakeGame()
    game.main_loop()

