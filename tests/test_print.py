import gym
import gym_battleship

env = gym.make('Battleship-v0')
env.reset()

for i in range(10):
    env.step(env.action_space.sample())
    env.render()

env.render_board_generated()
env.render()
