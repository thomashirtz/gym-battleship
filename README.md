# gym-battleship
Battleship environment using the OpenAI environment toolkit.<img align="right" width="320"  src="battleship.png"> 

## Battleship Environment
### Basics

Make and initialize an environment:
```python
import gym
import gym_battleship
env = gym.make('Battleship-v0')
env.reset()
```

Get the action space and the observation space:
```python
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]
```

Run a random agent:
```python
for i in range(10):
    env.step(env.action_space.sample())
```

Observe the hidden game state:
```python
print(env.board_generated)
```
### Valid actions

There is two way to input the action.  
The first way is to input the tuple as it is:
```python
env = gym.make('battleship-v0')
env.reset()
action = (0, 0)
env.step(action)
```

The second way is to input an encoded action:
```python
env = gym.make('battleship-v0')
env.reset()
action = 10
env.step(action)
```

### Customize environments

The original battleship game is played on a 10x10 grid and the fleet is composed 5 ships:  Carrier (occupies 5 spaces), Battleship (4), Cruiser (3), Submarine (3), and Destroyer (2).  

Therefore, the default parameters are :
```python
ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1}
board_size = (10, 10)
```

It is possible to change the parameters when making the environment:
```python
import gym
import gym_battleship

env = gym.make('battleship-v0', ship_sizes={4: 2, 3: 1}, board_size=(5, 5))
```

The other possible parameters are the different rewards as well as the maximum step:
```python
import gym
import gym_battleship

env = gym.make('battleship-v0', episode_steps=10, reward_dictionary={'win': 200})
```

The default reward keys and values are the following:
```python
reward_dictionary = {
    'win': 100,
    'missed': 0,
    'touched': 1,
    'repeat_missed': -1,
    'repeat_touched': -0.5
}
```
It is only necessary to pass to the environment the rewards that you want to edit.

### Render

Two functions exist to render the environment:
```python
env.render_board_generated()
```
```python
env.render()
```
Examples of renders using on an Ipython notebook (`env.render()` on the left and `env.render_board_generated()` on the right)
![ipython-render](ipython-render.jpg)

<details>
    <summary>Code snippet for rendering</summary>

        import gym
        import gym_battleship

        env = gym.make('battleship-v0')
        env.reset()

        for i in range(10):
            env.step(env.action_space.sample())
            env.render()

        env.render_board_generated()
        
</details>

Unfortunately, the pretty print of dataframe in IDE or console will not be as nice as a dataframe displayed in a notebook.

<<<<<<< HEAD
## Adversarial Battleship Environment

### About
Adversarial battleship environment is an environment that aim to make a battle between two reinforcement learning agent.
This environment is inspired by the Generative Adversarial Algorithms (GANs).
One agent is the defender and has the goal to strategically place his ships, while the other has the role of the attacker,
like in the `Battleship-v0` environment.

### Basics

The following script shows how to train the two agents:

```
import gym
import gym_battleship
from collections import namedtuple

env = gym.make('AdversarialBattleship-v0')
step = namedtuple('step', ['state', 'reward', 'done', 'info'])

attacker_agent = env.attacker_action_space.sample
defender_agent = env.defender_action_space.sample

num_episodes = 1
for episode in range(num_episodes):

    defender_old_state = env.defender_reset()
    while True:
        defender_action = defender_agent()
        return_value = env.defender_step(*defender_action)
        if return_value:
            defender_step = step(*return_value)
            # defender_agent.learn(defender_old_state, defender_step.reward, defender_step.state, defender_step.done)
            defender_old_state = defender_step.state
        else:
            break

    attacker_step = step(*env.attacker_initialization(), False, {})
    env.render_board_generated()

    while not attacker_step.done:
        attacker_action = attacker_agent()
        attacker_step = step(*env.attacker_step(attacker_action))
        # defender_agent.learn(attacker_old_state, attacker_step.reward, attacker_step.state, attacker_step.done)
        attacker_old_state = attacker_step.state

    defender_step = step(*env.defender_step(*defender_action))
    # defender_agent.learn(defender_old_state, defender_step.reward, defender_step.state, defender_step.done)
```

First the defender, place his ships, then he received a `None` value to indicate that he needs to wait the action of the
attackers. Then the attackers plays like in `Battleship-v0`. Finally, the defender get his reward and can do its last
learning step.

The render and customization is identical as in the `Battleship-v0` environment.
=======
## Todo

- Write docstring
- Create methods to let agents choose the position of the ships

## Requirements

gym  
numpy

## Installation

The command to install the repository via pip is:
```bash
pip install git+https://github.com/thomashirtz/gym-battleship#egg=gym-battleship
```
