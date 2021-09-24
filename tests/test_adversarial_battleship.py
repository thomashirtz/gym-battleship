import gym
import gym_battleship
from collections import namedtuple

env = gym.make('AdversarialBattleship-v0')
step = namedtuple('step', ['state', 'reward', 'done', 'info'])

attacker_agent = env.attacker_action_space.sample
defender_agent = env.defender_action_space.sample

if __name__ == '__main__':

    num_epochs = 1
    for epoch in range(num_epochs):

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
