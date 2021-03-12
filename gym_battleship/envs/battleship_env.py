import gym
import numpy as np
from abc import ABC
from gym import spaces
from copy import deepcopy
from typing import Tuple, Optional
from collections import namedtuple


Ship = namedtuple('Ship', ['min_x', 'max_x', 'min_y', 'max_y'])
Action = namedtuple('Action', ['x', 'y'])


class BattleshipEnv(gym.Env, ABC):
    def __init__(self,
                 board_size: Tuple = None,
                 ship_sizes: dict = None,
                 episode_steps: int = 100,
                 reward_dictionary: Optional[dict] = None):

        self.ship_sizes = ship_sizes or {5: 1, 4: 1, 3: 2, 2: 1}
        self.board_size = board_size or (10, 10)

        self.board = None  # Hidden state updated throughout the game
        self.board_generated = None  # Hidden state generated and left not updated (for debugging purposes)
        self.observation = None  # the observation is a (2, n, m) matrix, the first channel is for the missile that was launched on a cell that contained a boat, the second channel is for the missile launched that was launch on a cell that was not containing any boat

        self.done = None
        self.step_count = None
        self.episode_steps = episode_steps

        reward_dictionary = {} if reward_dictionary is None else reward_dictionary
        default_reward_dictionary = reward_dictionary or {  # todo further tuning of the rewards required
            'win': 100,
            'missed': 0,
            'touched': 1,
            'repeat_missed': -1,
            'repeat_touched': -0.5
            }
        self.reward_dictionary = {key: reward_dictionary.get(key, default_reward_dictionary[key]) for key in default_reward_dictionary.keys()}

        self.action_space = spaces.Discrete(self.board_size[0] * self.board_size[1])
        self.observation_space = spaces.MultiBinary([2, self.board_size[0], self.board_size[1]])

    def step(self, raw_action: int) -> Tuple[np.ndarray, int, bool, dict]:
        assert (raw_action < self.board_size[0]*self.board_size[1]),\
            "Invalid action (Superior than size_board[0]*size_board[1])"

        action = Action(x=raw_action % self.board_size[0], y=raw_action // self.board_size[0])
        self.step_count += 1

        # Check if the game is done (if true, the current step is the "last step")
        if self.step_count >= self.episode_steps:
            self.done = True

        # Touched (board[x, y] == 1)
        if self.board[action.x, action.y] == 1:
            self.board[action.x, action.y] = 0
            self.observation[0, action.x, action.y] = 1
            # Win (No boat left)
            if not self.board.any():
                self.done = True
                return self.observation, self.reward_dictionary['win'], self.done, {}
            return self.observation, self.reward_dictionary['touched'], self.done, {}

        # Repeat touched (observation[0, x, y] == 1)
        elif self.observation[0, action.x, action.y] == 1:
            return self.observation, self.reward_dictionary['repeat_touched'], self.done, {}

        # Repeat missed (observation[1, x, y] == 1)
        elif self.observation[1, action.x, action.y] == 1:
            return self.observation, self.reward_dictionary['repeat_missed'], self.done, {}

        # Missed (Action not repeated and boat(s) not touched)
        else:
            self.observation[1, action.x, action.y] = 1
            return self.observation, self.reward_dictionary['missed'], self.done, {}

    def reset(self) -> np.ndarray:
        self.set_board()
        self.board_generated = deepcopy(self.board)
        self.observation = np.zeros((2, *self.board_size), dtype=np.float32)
        self.step_count = 0
        return self.observation

    def set_board(self) -> None:
        self.board = np.zeros(self.board_size, dtype=np.float32)
        for ship_size, ship_count in self.ship_sizes.items():
            for _ in range(ship_count):
                self.place_ship(ship_size)

    def place_ship(self, ship_size: int) -> None:
        can_place_ship = False
        while not can_place_ship:  # todo add protection infinite loop
            ship = self.get_ship(ship_size, self.board_size)
            can_place_ship = self.is_place_empty(ship)
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = True

    @staticmethod
    def get_ship(ship_size: int, board_size: tuple) -> Ship:
        if np.random.choice(('Horizontal', 'Vertical')) == 'Horizontal':
            min_x = np.random.randint(0, board_size[0] + 1 - ship_size)
            min_y = np.random.randint(0, board_size[1])
            return Ship(min_x=min_x, max_x=min_x + ship_size, min_y=min_y, max_y=min_y + 1)
        else:
            min_x = np.random.randint(0, board_size[0])
            min_y = np.random.randint(0, board_size[1] + 1 - ship_size)
            return Ship(min_x=min_x, max_x=min_x + 1, min_y=min_y, max_y=min_y + ship_size)

    def is_place_empty(self, ship: Ship) -> bool:
        return np.count_nonzero(self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y]) == 0

    def render(self, mode='human'):
        board = np.full(self.board_size, ' ', dtype=str)
        board[self.observation[0] != 0] = '❌'
        board[self.observation[1] != 0] = '⚪'
        self._render(board)

    def render_board_generated(self):
        board = np.full(self.board_size, '⬜', dtype=str)
        board[self.board_generated != 0] = '⬛'
        self._render(board)

    @staticmethod
    def _render(board):
        import pandas as pd

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord('A'), ord('A') + num_columns)]
        index = [i + 1 for i in range(num_rows)]
        with pd.option_context('display.colheader_justify', 'right'):
            print(pd.DataFrame(board, columns=columns, index=index), end='\n')  # todo issue IDE not align properly the df
