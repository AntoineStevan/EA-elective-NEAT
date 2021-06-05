from pprint import pprint

import numpy as np
from PIL import Image
from minatar import Environment


class MinatarWrapper(Environment):
    """

    """
    # def __init__(self, env_name, sticky_action_prob=.0, random_seed=0):
    #     super(MinatarWrapper, self).__init__(env_name, sticky_action_prob=sticky_action_prob, random_seed=random_seed)

    def reset(self):
        """
            Resets the environment.

            Return:
                (observation) the first observation.
        """
        super().reset()
        return self._state().flatten()

    def step(self, actions):
        """
            Resets the environment.

            Args:
                actions ():_______________________________________TODO__________________________________________________

            Return:
                (observation) the first observation.
        """
        reward, done = self.act(minatar_action(actions))
        state = self._state().flatten()

        return state, reward, done, {}

    def render(self, done=False):
        """
            Resets the environment.

            Args:
                done (bool):_________________________________________TODO_______________________________________________

            Return:
                (observation) the first observation.
        """
        self.display_state(time=50)
        state = self._state()
        state = state / np.max(state) * 256
        image = Image.fromarray(state)
        return image.convert('P')

    def _state(self):
        state = super().state()
        state = state.transpose((2, 0, 1))
        state = np.sum([state[i] * (i+1) for i in range(state.shape[0])], axis=0)
        return state


# == EA-elective-NEAT ==================================================================================================
def minatar_action(actions):
    actions = actions.flatten()
    action = np.random.choice(np.arange(actions.size), p=actions)
    return action
# ======================================================================================================================
