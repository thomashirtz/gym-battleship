# gym-battleship
Battleship environment using the OpenAI environment toolkit.

## Requirements

gym  
numpy

## Basics

Make and initialize an environment:
```
import gym
import gym_battleship
env = gym.make('battleship-v0')
env.reset()
```

Get the action space and the observation space:
```
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]
```

Run a random agent:
```
for i in range(10):
    env.step(env.action_space.sample())
```

## Parameters

The original battleship game is played on a 10x10 grid and the fleet is composed 5 ships:  Carrier (occupies 5 spaces), Battleship (4), Cruiser (3), Submarine (3), and Destroyer (2).  

Therefore, the default parameters are :
```
ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1}
board_size = (10, 10)
```

It is possible to change the parameters when making the environment:
```
import gym
import gym_battleship
env = gym.make('battleship-v0', ship_sizes={4: 2, 3: 1}, board_size=(5, 5))
```
