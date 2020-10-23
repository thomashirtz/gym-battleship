# gym-battleship
Battleship environment for reinforcement learning tasks

## Requirements

gym  
numpy

## Basics

Make an environment and run a random agent:
```
import gym
import gym_battleship
env = gym.make('battleship-v0')
env.reset()

for i in range(1):
    env.step(env.action_space.sample())
```

Get the action space and the observation space:
```
import gym
import gym_battleship
env = gym.make('battleship-v0')

ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]
```