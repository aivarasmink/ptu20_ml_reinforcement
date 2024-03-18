import pygame
import random
from enum import Enum
from collections import namedtuple

# Initialize Pygame
pygame.init()

# Font for displaying text
font = pygame.font.SysFont('arial', 25)

# Define colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Constants
BLOCK_SIZE = 20
SPEED = 20

# Define direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define Point namedtuple for representing coordinates
Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, width=640, height=480):
        """
        Initialize the Snake game.

        Parameters:
            width (int): Width of the game window.
            height (int): Height of the game window.
        """
        self.width = width
        self.height = height

        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')

        # Clock to control the game's speed
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        """
        Place food randomly on the game grid.
        """
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        """
        Play a single step of the game.

        Returns:
            bool: True if the game is over, False otherwise.
            int: The current score.
        """
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Move the snake
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # Check for collision
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return game over and score
        return game_over, self.score

    def _is_collision(self):
        """
        Check if the snake has collided with the boundary or itself.

        Returns:
            bool: True if collision occurred, False otherwise.
        """
        # Check boundary collision
        if (self.head.x > self.width - BLOCK_SIZE or self.head.x < 0 or
                self.head.y > self.height - BLOCK_SIZE or self.head.y < 0):
            return True
        # Check self-collision
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        Update the game UI.
        """
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        """
        Move the snake in the specified direction.

        Parameters:
            direction (Direction): Direction to move the snake.
        """
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)


if __name__ == '__main__':
    # Create and run the game
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score', score)
    pygame.quit()
