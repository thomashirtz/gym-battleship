import numpy as np
from gym import spaces
from copy import deepcopy

from typing import Union
from typing import Tuple
from typing import Optional

from gym_battleship.environments.battleship import Ship
from gym_battleship.environments.battleship import Action
from gym_battleship.environments.battleship import BattleshipEnv


class AdversarialBattleshipEnv(BattleshipEnv):  # noqa
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining_ships: Optional[np.array] = None

        self.attacker_action_space = self.action_space
        self.attacker_observation_space = self.observation_space

        del self.action_space
        del self.observation_space

        self.defender_action_space = spaces.Tuple((
            spaces.Discrete(self.board_size[0] * self.board_size[1]),
            spaces.Discrete(2)
        ))
        self.defender_observation_space = spaces.Tuple((
            spaces.Box(low=0, high=max(self.ship_sizes.values()), shape=(max(self.ship_sizes.keys()),), dtype=int),
            spaces.Box(low=0, high=1, shape=(self.board_size[0], self.board_size[1]), dtype=int)
        ))

    def reset(self) -> Tuple[np.ndarray, np.array]:
        return self.defender_reset()

    def defender_reset(self) -> Tuple[np.ndarray, np.array]:
        self.done = False
        self.step_count = 0
        self.board = np.zeros(self.board_size, dtype=np.float32)
        self.remaining_ships: np.array = np.zeros(max(self.board_size), dtype=np.float32)
        for ship_size, ship_count in self.ship_sizes.items():
            self.remaining_ships[ship_size] = ship_count
        return self.board, self.remaining_ships

    def attacker_initialization(self) -> np.ndarray:
        self.board_generated = deepcopy(self.board)
        self.observation = np.zeros((2, *self.board_size), dtype=np.float32)
        return self.observation

    def attacker_step(self, raw_action: Union[int, tuple]) -> Tuple[np.ndarray, int, bool, dict]:
        return self.step(raw_action=raw_action)

    def _place_ship(self, ship: Ship) -> None:
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = True

    def _is_inside_board(self, ship: Ship) -> bool:
        if ship.max_x > self.board_size[0] or ship.max_y > self.board_size[1]:
            return False
        else:
            return True

    def _get_ship(self, x: int, y: int, ship_size: int, orientation: Union[str, int]) -> Ship:  # noqa
        if orientation == 'Horizontal' or orientation == 0:
            return Ship(min_x=x, max_x=x + ship_size, min_y=y, max_y=y + 1)
        elif orientation == 'Vertical' or orientation == 1:
            return Ship(min_x=x, max_x=x + 1, min_y=y, max_y=y + ship_size)
        else:
            raise ValueError

    def defender_step(self, raw_action: Union[int, tuple], orientation: Union[str, int]) -> \
            Optional[Tuple[Tuple[np.ndarray, np.array], float, bool, dict]]:

        if isinstance(raw_action, int):
            assert (0 <= raw_action < self.board_size[0] * self.board_size[1]), \
                "Invalid action (The encoded action is outside of the limits)"
            action = Action(x=raw_action % self.board_size[0], y=raw_action // self.board_size[0])

        elif isinstance(raw_action, tuple):
            assert (0 <= raw_action[0] < self.board_size[0] and 0 <= raw_action[1] < self.board_size[1]), \
                "Invalid action (The action is outside the board)"
            action = Action(x=raw_action[0], y=raw_action[1])

        else:
            raise AssertionError("Invalid action (Unsupported raw_action type)")

        assert orientation in (0, 1, "Horizontal", "Vertical"), f"Invalid Orientation: {orientation}"

        if self.done:
            reward = self.step_count / self.episode_steps
            return (self.board_generated, self.remaining_ships), reward, True, {}

        elif np.any(self.remaining_ships):

            ship_size = self._pop_biggest_ship_size()
            ship = self._get_ship(x=action.x, y=action.y, ship_size=ship_size, orientation=orientation)
            place_empty = self._is_place_empty(ship)
            inside_board = self._is_inside_board(ship)
            if place_empty and inside_board:
                self._place_ship(ship)

            need_to_wait_attacker = not np.any(self.remaining_ships)
            if need_to_wait_attacker:
                return None
            else:
                return (self.board, self.remaining_ships), 0, False, {}
        else:
            return None

    def _pop_biggest_ship_size(self):
        for i in range(len(self.remaining_ships)-1, -1, -1):
            if self.remaining_ships[i] > 0:
                self.remaining_ships[i] -= 1
                return i
        return 0  # todo maybe remove return statement

    def __dir__(self):
        original_dir = set(list(self.__dict__.keys()) + dir(self.__class__))
        for invalid_attribute in ['step', 'reset', '_set_board', '_get_ship', '_is_place_empty']:
            original_dir.remove(invalid_attribute)  # Attribute coming from the BattleshipEnv class
        return sorted(original_dir)
