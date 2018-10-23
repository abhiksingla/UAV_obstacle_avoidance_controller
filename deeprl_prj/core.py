"""Core classes."""

import numpy as np
from PIL import Image

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. 

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.
        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.
        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass

class ReplayMemory:
    """Interface for replay memories.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. 
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, args):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.memory_size = args.replay_memory_size
        self.history_length = args.num_frames
        self.actions = np.zeros(self.memory_size, dtype = np.int8)
        self.rewards = np.zeros(self.memory_size, dtype = np.int8)
        self.screens = np.zeros((self.memory_size, args.frame_height, args.frame_width), dtype = np.uint8)
        self.terminals = np.zeros(self.memory_size, dtype = np.bool)
        self.current = 0

    def append(self, state, action, reward, is_terminal):
        self.actions[self.current % self.memory_size] = action
        self.rewards[self.current % self.memory_size] = reward
        self.screens[self.current % self.memory_size] = state
        self.terminals[self.current % self.memory_size] = is_terminal
        # img = Image.fromarray(state, mode = 'L')
        # path = "./tmp/%05d-%s.png" % (self.current, is_terminal)
        # img.save(path)
        self.current += 1

    def get_state(self, index):
        state = self.screens[index - self.history_length + 1:index + 1, :, :]
        # history dimention last
        return np.transpose(state, (1, 2, 0)) 

    def sample(self, batch_size):
        samples = []
        indexes = []
        # ensure enough frames to sample
        assert self.current > self.history_length
        # -1 because still need next frame
        end = min(self.current, self.memory_size) - 1

        while len(indexes) < batch_size: 
            index = np.random.randint(self.history_length - 1, end)
            # sampled state shouldn't contain episode end
            if self.terminals[index - self.history_length + 1: index + 1].any():
                continue
            indexes.append(index)

        for idx in indexes:
            new_sample = Sample(self.get_state(idx), self.actions[idx],
                self.rewards[idx], self.get_state(idx + 1), self.terminals[idx])
            samples.append(new_sample)
        return samples

    def clear(self):
        self.current = 0
