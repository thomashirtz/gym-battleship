from gym.envs.registration import register

register(
    id='battleship-v0',
    entry_point='gym_battleship.envs:BattleshipEnv',
)