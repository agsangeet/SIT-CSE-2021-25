import numpy as np
import random
import pygame

# Constants
GRID_SIZE = 10
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]  # 8 degrees of movement
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration factor
CELL_SIZE = 50  # Size of each cell in the grid

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
pygame.display.set_caption("Cat and Mouse Game")
clock = pygame.time.Clock()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Define the environment (Cat and Mouse Grid)
class CatMouseEnv:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Place mouse, cat, and cheese in random positions
        self.mouse_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        return self.get_state()

    def get_state(self):
        return tuple(self.mouse_pos + self.cat_pos + self.cheese_pos)

    def is_valid(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def step(self, action):
        # Move mouse
        new_mouse_pos = [self.mouse_pos[0] + action[0], self.mouse_pos[1] + action[1]]
        if self.is_valid(new_mouse_pos):
            self.mouse_pos = new_mouse_pos

        # Move cat towards mouse
        self.move_cat()

        # Check for win/lose condition
        if self.mouse_pos == self.cheese_pos:
            reward = 1  # Mouse gets cheese
            done = False  # Game continues after getting the cheese
            self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]  # New cheese position
        elif self.mouse_pos == self.cat_pos:
            reward = -1  # Cat catches mouse
            done = True  # Game ends when the cat catches the mouse
        else:
            reward = 0  # No one has won yet
            done = False

        return self.get_state(), reward, done

    def move_cat(self):
        if self.cat_pos[0] < self.mouse_pos[0]:
            self.cat_pos[0] += 1
        elif self.cat_pos[0] > self.mouse_pos[0]:
            self.cat_pos[0] -= 1

        if self.cat_pos[1] < self.mouse_pos[1]:
            self.cat_pos[1] += 1
        elif self.cat_pos[1] > self.mouse_pos[1]:
            self.cat_pos[1] -= 1

    def render(self):
        # Draw grid
        screen.fill(WHITE)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pygame.draw.rect(screen, BLACK, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        # Draw mouse, cat, and cheese
        pygame.draw.rect(screen, BLUE, pygame.Rect(self.mouse_pos[1] * CELL_SIZE, self.mouse_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(self.cat_pos[1] * CELL_SIZE, self.cat_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, YELLOW, pygame.Rect(self.cheese_pos[1] * CELL_SIZE, self.cheese_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

# Q-learning Agent
class QLearningAgent:
    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.q_table = {}  # Q-table: maps state-action pairs to rewards
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)
        max_next_q_value = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_next_q_value - old_q_value)
        self.q_table[(state, action)] = new_q_value

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:  # Explore
            return random.choice(self.actions)
        else:  # Exploit
            q_values = [self.get_q_value(state, action) for action in self.actions]
            max_q = max(q_values)
            return self.actions[q_values.index(max_q)]

# Main loop to run the simulation with pygame visualization
def run_simulation(episodes=500):
    env = CatMouseEnv(GRID_SIZE)
    agent = QLearningAgent(ACTIONS)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Handle Pygame events (for closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Choose action and take a step in the environment
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

            # Render the environment
            env.render()

            # Control the frame rate
            clock.tick(5)

            if done:
                break  # Terminate the game if the cat catches the mouse

        if episode % 50 == 0:
            print(f"Episode {episode}: Q-table size {len(agent.q_table)}")


if __name__ == "__main__":
    run_simulation()