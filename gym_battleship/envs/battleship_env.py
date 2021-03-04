import gym
import numpy as np
from abc import ABC
from gym import spaces
from typing import Tuple
from copy import deepcopy
from collections import namedtuple

Ship = namedtuple('Ship', ['min_x', 'max_x', 'min_y', 'max_y'])
Action = namedtuple('Action', ['x', 'y'])


class BattleshipEnv(gym.Env, ABC):
    def __init__(self, board_size: Tuple = None, ship_sizes: dict = None, episode_steps: int = 100):
        self.ship_sizes = ship_sizes or {5: 1, 4: 1, 3: 2, 2: 1}
        self.board_size = board_size or (10, 10)

        self.board = None
        self.board_generated = None
        self.observation = None

        self.done = None
        self.step_count = None
        self.episode_steps = episode_steps

        self.action_space = spaces.Discrete(self.board_size[0] * self.board_size[1])
        self.observation_space = spaces.MultiBinary([2, self.board_size[0], self.board_size[1]])

    def step(self, raw_action: int) -> Tuple[np.ndarray, int, bool, dict]:
        assert (raw_action < self.board_size[0]*self.board_size[1]),\
            "Invalid action (Superior than size_board[0]*size_board[1])"

        action = Action(x=raw_action % self.board_size[0], y=raw_action // self.board_size[0])
        self.step_count += 1
        if self.step_count >= self.episode_steps:
            self.done = True

        if self.board[action.x, action.y] != 0:
            self.board[action.x, action.y] = 0
            self.observation[0, action.x, action.y] = 1
            if not self.board.any():
                self.done = True
                return self.observation, 100, self.done, {}
            return self.observation, 1, self.done, {}

        elif self.observation[1, action.x, action.y] == 1:
            return self.observation, 0, self.done, {}

        else:
            self.observation[1, action.x, action.y] = 1
            return self.observation, -1, self.done, {}

    def reset(self):
        self.set_board()
        self.board_generated = deepcopy(self.board)
        self.observation = np.zeros((2, *self.board_size), dtype=np.float32)
        self.step_count = 0
        return self.observation

    def set_board(self):
        self.board = np.zeros(self.board_size, dtype=np.float32)
        for ship_size, ship_count in self.ship_sizes.items():
            for _ in range(ship_count):
                self.place_ship(ship_size)

    def place_ship(self, ship_size):
        can_place_ship = False
        while not can_place_ship:
            ship = self.get_ship(ship_size, self.board_size)
            can_place_ship = self.is_place_empty(ship)
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = True

    @staticmethod
    def get_ship(ship_size, board_size) -> Ship:
        if np.random.choice(('Horizontal', 'Vertical')) == 'Horizontal':
            min_x = np.random.randint(0, board_size[0] + 1 - ship_size)
            min_y = np.random.randint(0, board_size[1])
            return Ship(min_x=min_x, max_x=min_x + ship_size, min_y=min_y, max_y=min_y + 1)
        else:
            min_x = np.random.randint(0, board_size[0])
            min_y = np.random.randint(0, board_size[1] + 1 - ship_size)
            return Ship(min_x=min_x, max_x=min_x + 1, min_y=min_y, max_y=min_y + ship_size)

    def is_place_empty(self, ship):
        return np.count_nonzero(self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y]) == 0
