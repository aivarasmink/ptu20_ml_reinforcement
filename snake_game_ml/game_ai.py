import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

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
SPEED = 50

# Define direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define Point namedtuple for representing coordinates
Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, width=640, height=480):
        """
        Initialize the SnakeGameAI class.

        Parameters:
        - width: Width of the game window.
        - height: Height of the game window.
        """
        self.width = width
        self.height = height

        # Initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')

        # Clock to control the game's speed
        self.clock = pygame.time.Clock()
        self.reset()

        # Initialize game state
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def reset(self):
        """
        Reset the game state.
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.width / 2, self.height / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        Place food randomly on the game window.
        """
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Execute a step in the game.

        Parameters:
        - action: Action to be taken by the agent.

        Returns:
        - reward: Reward obtained from the action.
        - game_over: Boolean indicating if the game is over.
        - score: Current score.
        """
        self.frame_iteration += 1
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Check for collision
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Check for collision with boundaries or self.

        Parameters:
        - pt: Point to check for collision (default is head).

        Returns:
        - True if collision occurs, False otherwise.
        """
        if pt is None:
            pt = self.head
        # Check boundary collision
        if (pt.x > self.width - BLOCK_SIZE or pt.x < 0 or
                pt.y > self.height - BLOCK_SIZE or pt.y < 0):
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        Update the game's user interface.
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

    def _move(self, action):
        """
        Move the snake based on the given action.

        Parameters:
        - action: Action to be taken by the snake.
        """
        #[straight, left, up, down]      
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r ->d -> l -> up
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r ->up ->l -> down

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
