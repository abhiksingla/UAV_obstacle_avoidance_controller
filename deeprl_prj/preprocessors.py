"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_prj import utils
from deeprl_prj.core import Preprocessor


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history_length = history_length
        self.past_states = None 
        self.past_states_ori = None 

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        row, col = state.shape
        if self.past_states is None:
            self.past_states = np.zeros((row, col, self.history_length))
        history = np.dstack((self.past_states, state))
        self.past_states = history[:, :, 1:]
        return history

    def process_state_for_network_ori(self, state):
        """You only want history when you're deciding the current action to take."""
        row, col = state.shape
        channel = 1
        if self.past_states_ori is None:
            self.past_states_ori = np.zeros((row, col, channel, self.history_length))
        history = np.concatenate((self.past_states_ori, np.expand_dims(state, -1)), axis=3)
        self.past_states_ori = history[:, :, :, 1:]
        return history

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.past_states = None
        self.past_states_ori = None 

    def get_config(self):
        return {'history_length': self.history_length}

class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        img = Image.fromarray(state).convert('L').resize((84, 84), Image.BILINEAR)
        state = np.array(img)
        return state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        return np.float32(self.process_state_for_memory(state) / 255.0)

    def process_state_for_network_ori(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        img = Image.fromarray(state)
        state = np.float32(np.array(img) / 255.0)
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        batch_size = len(samples)
        for i in range(batch_size):
            samples[i].state = np.float32(samples[i].state / 255.0)
            samples[i].next_state = np.float32(samples[i].next_state / 255.0)
        return samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        # return np.clip(reward, -1, 1) 
        return reward
    
    def reset(self):
        self.last_state = None
