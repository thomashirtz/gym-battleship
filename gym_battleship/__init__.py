from gym.envs.registration import register
from gym_battleship.environments.battleship import BattleshipEnv
from gym_battleship.environments.adversarial_battleship import AdversarialBattleshipEnv


register(
    id='Battleship-v0',
    entry_point='gym_battleship:BattleshipEnv',
)
register(
    id='AdversarialBattleship-v0',
    entry_point='gym_battleship:AdversarialBattleshipEnv',
)
