from gym.envs.registration import register
from gym_battleship.battleship_env import BattleshipEnv

register(
    id='battleship-v0',
    entry_point='gym_battleship:BattleshipEnv',
)