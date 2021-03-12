# gym-battleship
Battleship environment using the OpenAI environment toolkit.

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

Observe the hidden game state:
```
print(env.board_generated)
```

## Customize environments

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

The other possible parameters are the different rewards as well as the maximum step:
```
import gym
import gym_battleship

env = gym.make('battleship-v0', episode_steps=10, reward_dictionary={'win': 200})
```

The default reward keys and values are the following:
```
reward_dictionary = {
    'win': 100,
    'missed': 0,
    'touched': 1,
    'repeat_missed': -1,
    'repeat_touched': -0.5
}
```
It is only necessary to pass to the environment the rewards that you want to edit.

## Render

Two functions exist to render the environment:
```
env.render_board_generated()
```
```
    A  B  C  D  E  F  G  H  I  J
1                               
2   ⬛                           
3   ⬛                          ⬛
4   ⬛                          ⬛
5   ⬛                          ⬛
6   ⬛                          ⬛
7                               
8   ⬛                 ⬛  ⬛  ⬛   
9   ⬛  ⬛  ⬛                     
10  ⬛                           
```
```
env.render()
```
```
    A  B  C  D  E  F  G  H  I  J
1                               
2                               
3               ⚪        ⚪      
4                               
5                               
6            ⚪                 ❌
7         ⚪                 ⚪   
8            ⚪           ❌      
9                               
10           ⚪  ⚪               
```
code snippet:

```
import gym
import gym_battleship

env = gym.make('battleship-v0')
env.reset()

for i in range(10):
    env.step(env.action_space.sample())
    env.render()

env.render_board_generated()
```

## todo

- Add docstring

## Requirements

gym  
numpy

## Installation

The command to install the repository via pip is:
```
pip install git+https://github.com/thomashirtz/gym-battleship#egg=gym-battleship
```
