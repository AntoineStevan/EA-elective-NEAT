import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from minatar import Environment

import domain


class MinatarWrapper(Environment):
    """

    """
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
            Steps in the environment.

            Args:
                actions (): the action to take.

            Return:
                (tensor, float, bool, dict) new observation, reward, done signal and complementary informations.
        """
        # reward, done = self.act(minatar_action(actions))
        reward, done = self.act(5)
        state = self._state().flatten()

        return state, reward, done, {}

    def render(self, time=0, done=False):
        """
            Resets the environment.

            Args:
                time (int): the number of milliseconds for each frame. if 0, there will be no live animation.
                done (bool): tells if the episode is done.

            Return:
                (Image) the current image of the game.
        """
        if time:
            self.display_state(time=time)
        state = self._state()
        state = state / np.max(state) * 256
        image = Image.fromarray(state)
        return image.convert('P')

    def _state(self):
        """
            Reduces the dimensions of the raw observation and normalize it.
        """
        # get the obsservation.
        state = super().state()
        # transpose to make it human readable.
        state = state.transpose((2, 0, 1))
        # sums the object channels to have a single image.
        state = np.sum([state[i] * (i + 1) for i in range(state.shape[0])], axis=0)
        # normalize the image
        m, M = np.min(state), np.max(state)
        state = 2 * (state - m) / (M - m) - 1
        return state


# == EA-elective-NEAT ==================================================================================================
def minatar_action(actions):
    actions = actions.flatten()
    action = np.random.choice(np.arange(actions.size), p=actions)
    return action
# ======================================================================================================================
