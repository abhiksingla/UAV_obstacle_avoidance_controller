'''Keras DQN Agent implementation. Includes Basic Dueling Double DQN and Temporal Attention DQN.'''

from deeprl_prj.policy import *
from deeprl_prj.objectives import *
from deeprl_prj.preprocessors import *
from deeprl_prj.utils import *
from deeprl_prj.core import *

import keras
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, Merge, Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute, multiply)
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import sys
from gym import wrappers
import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))

def create_model(input_shape, num_actions, mode, args, model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int, int), rows, cols, channels
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    assert(mode in ("linear", "duel", "dqn"))
    with tf.variable_scope(model_name):
        input_data = Input(shape = input_shape, name = "input")
        if mode == "linear": # We will never enter this loop
            flatten_hidden = Flatten(name = "flatten")(input_data) #(H, W, D, Batch)
            output = Dense(num_actions, name = "output")(flatten_hidden)
        # Directly come here for DQN
        else:
            if not(args.recurrent): # Only when "not" using DRQN
                h1 = Convolution2D(32, (8, 8), strides = 4, activation = "relu", name = "conv1")(input_data)
                h2 = Convolution2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2")(h1)
                h3 = Convolution2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3")(h2)
                context = Flatten(name = "flatten")(h3)
            # ENTER HERE FOR DRQN
            else:
                print('>>>> Defining Recurrent Modules...')
                input_data_expanded = Reshape((input_shape[0], input_shape[1], input_shape[2], 1), input_shape = input_shape) (input_data)
                input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=input_shape)(input_data_expanded) # (D, H, W, Batch)
                h1 = TimeDistributed(Convolution2D(32, (8, 8), strides = 4, activation = "relu", name = "conv1"), \
                    input_shape=(args.num_frames, input_shape[0], input_shape[1], 1))(input_data_TimeDistributed)
                h2 = TimeDistributed(Convolution2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2"))(h1)
                h3 = TimeDistributed(Convolution2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3"))(h2)
                flatten_hidden = TimeDistributed(Flatten())(h3)
                hidden_input = TimeDistributed(Dense(512, activation = 'relu', name = 'flat_to_512')) (flatten_hidden)
                if not(args.a_t):
                    context = LSTM(512, return_sequences=False, stateful=False, input_shape=(args.num_frames, 512)) (hidden_input)
                else:
                    if args.bidir:
                        hidden_input = Bidirectional(LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)), merge_mode='sum') (hidden_input)
                        all_outs = Bidirectional(LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)), merge_mode='sum') (hidden_input)
                    else:
                        all_outs = LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)) (hidden_input)
                    # attention
                    attention = TimeDistributed(Dense(1, activation='tanh'))(all_outs) 
                    # print(attention.shape)
                    attention = Flatten()(attention)
                    attention = Activation('softmax')(attention)
                    attention = RepeatVector(512)(attention)
                    attention = Permute([2, 1])(attention)
                    sent_representation = merge([all_outs, attention], mode='mul')
                    context = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512,))(sent_representation)
                    # print(context.shape)

            if mode == "dqn":
                h4 = Dense(512, activation='relu', name = "fc")(context)
                output = Dense(num_actions, name = "output")(h4)
            elif mode == "duel":
                value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(context)
                value = Dense(1, name = "value")(value_hidden)
                action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(context)
                action = Dense(num_actions, name = "action")(action_hidden)
                action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keep_dims = True), name = 'action_mean')(action) 
                output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'output')([action, value, action_mean])
    model = Model(inputs = input_data, outputs = output)
    print(model.summary())
    return model

def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters to implement the DQNAgnet. 

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self, args, num_actions):
        self.num_actions = num_actions
        input_shape = (args.frame_height, args.frame_width, args.num_frames)
        self.history_processor = HistoryPreprocessor(args.num_frames - 1)
        self.atari_processor = AtariPreprocessor()
        self.memory = ReplayMemory(args)
        self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon, args.exploration_steps)
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        self.num_burn_in = args.num_burn_in
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.num_frames = args.num_frames
        self.output_path = args.output
        self.output_path_videos = args.output + '/videos/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode
        self.q_network = create_model(input_shape, num_actions, self.net_mode, args, "QNet")
        self.target_network = create_model(input_shape, num_actions, self.net_mode, args, "TargetNet")
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        self.target_network.set_weights(self.q_network.get_weights())
        self.final_model = None
        self.compile()

        self.writer = tf.summary.FileWriter(self.output_path)

        print("*******__init__", input_shape)

    def compile(self, optimizer = None, loss_func = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is the place to create the target network, setup 
        loss function and any placeholders.
        """
        if loss_func is None:
            loss_func = mean_huber_loss
            # loss_func = 'mse'
        if optimizer is None:
            optimizer = Adam(lr = self.learning_rate)
            # optimizer = RMSprop(lr=0.00025)
        with tf.variable_scope("Loss"):
            state = Input(shape = (self.frame_height, self.frame_width, self.num_frames) , name = "states")
            action_mask = Input(shape = (self.num_actions,), name = "actions")
            qa_value = self.q_network(state)
            qa_value = merge([qa_value, action_mask], mode = 'mul', name = "multiply")
            qa_value = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims = True), name = "sum")(qa_value)

        self.final_model = Model(inputs = [state, action_mask], outputs = qa_value)
        self.final_model.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        return self.q_network.predict_on_batch(state)

    def select_action(self, state, is_training = True, **kwargs):
        """Select the action based on the current state.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(state)
        if is_training:
            if kwargs['policy_type'] == 'UniformRandomPolicy':
                return UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                return self.policy.select_action(q_values, is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            return GreedyPolicy().select_action(q_values)

    def update_policy(self, current_sample):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        batch_size = self.batch_size

        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = self.memory.sample(batch_size)
            samples = self.atari_processor.process_batch(samples)

            states = np.stack([x.state for x in samples])
            actions = np.asarray([x.action for x in samples])
            action_mask = np.zeros((batch_size, self.num_actions))
            action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            next_qa_value = self.target_network.predict_on_batch(next_states)

        if self.enable_ddqn:
            qa_value = self.q_network.predict_on_batch(next_states)
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        target = rewards + self.gamma * mask * next_qa_value

        return self.final_model.train_on_batch([states, action_mask], target), np.mean(target)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        This is where you sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is the Atari environment. 
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        is_training = True
        print("Training starts.")
        self.save_model(0)
        eval_count = 0

        state = env.reset()

        burn_in = True
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0

        # Logs
        losses_list = list()
        step_loss_list = list()
        step_reward = 0.0
        step_reward_raw = 0.0

        for t in range(self.num_burn_in + num_iterations):
            print ("iteration --> %s, episode --> %s" % (t, idx_episode))
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"
            action = self.select_action(action_state, is_training, policy_type = policy_type)
            processed_state = self.atari_processor.process_state_for_memory(state)

            # print("******* fit_action", action_state.shape)
            # print("******* fit_proecess", processed_state.shape)

            env.render()
            state, reward, done, info = env.step(action)

            processed_next_state = self.atari_processor.process_state_for_network(state)
            action_next_state = np.dstack((action_state, processed_next_state))
            action_next_state = action_next_state[:, :, 1:]

            processed_reward = self.atari_processor.process_reward(reward)

            self.memory.append(processed_state, action, processed_reward, done)
            current_sample = Sample(action_state, action, processed_reward, action_next_state, done)
            
            if not burn_in: 
                episode_frames += 1
                episode_reward += processed_reward
                episode_raw_reward += reward
                if episode_frames > max_episode_length:
                    done = True

            if not burn_in:
                step_reward += processed_reward
                step_reward_raw += reward
                step_losses = [t-last_burn-1, step_reward, step_reward_raw, step_reward / (t-last_burn-1), step_reward_raw / (t-last_burn-1)]
                step_loss_list.append(step_losses)


            if done:
                # adding last frame only to save last state
                last_frame = self.atari_processor.process_state_for_memory(state)
                # action, reward, done doesn't matter here
                self.memory.append(last_frame, action, 0, done)
                if not burn_in:
                    avg_target_value = episode_target_value / episode_frames
                    print(">>> Training: time %d, episode %d, length %d, reward %.0f, raw_reward %.0f, loss %.4f, target value %.4f, policy step %d, memory cap %d" % 
                        (t, idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss, 
                        avg_target_value, self.policy.step, self.memory.current))
                    sys.stdout.flush()
                    save_scalar(idx_episode, 'train/episode_frames', episode_frames, self.writer)
                    save_scalar(idx_episode, 'train/episode_reward', episode_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_raw_reward', episode_raw_reward, self.writer)
                    save_scalar(idx_episode, 'train/episode_loss', episode_loss, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_reward', episode_reward / episode_frames, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_target_value', avg_target_value, self.writer)
                    save_scalar(idx_episode, 'train_avg/avg_loss', episode_loss / episode_frames, self.writer)

                    # log losses
                    losses = [idx_episode, episode_frames, episode_reward, episode_raw_reward, episode_loss, episode_reward / episode_frames, avg_target_value, episode_loss / episode_frames]
                    losses_list.append(losses)

                    # reset values
                    episode_frames = 0
                    episode_reward = .0
                    episode_raw_reward = .0
                    episode_loss = .0
                    episode_target_value = .0
                    idx_episode += 1
                burn_in = (t < self.num_burn_in)
                state = env.reset()
                self.atari_processor.reset()
                self.history_processor.reset()

            if burn_in:
                last_burn = t

            if not burn_in:
                if t % self.train_freq == 0:
                    loss, target_value = self.update_policy(current_sample)
                    episode_loss += loss
                    episode_target_value += target_value
                # update freq is based on train_freq
                if t % (self.train_freq * self.target_update_freq) == 0:
                    # target updates can have the option to be hard or soft
                    # related functions are defined in deeprl_prj.utils
                    # here we use hard target update as default
                    self.target_network.set_weights(self.q_network.get_weights())
                if t % self.save_freq == 0:
                    self.save_model(idx_episode)

                    loss_array = np.asarray(losses_list)
                    print (loss_array.shape) # 10 element vector

                    # loss_path = os.path.join('./losses/loss_episode%s.csv' % (idx_episode))
                    loss_path = self.output_path + "/losses/loss_episodes" + str(idx_episode) + ".csv"
                    np.savetxt(loss_path, loss_array, fmt='%.5f', delimiter=',')

                    step_loss_array = np.asarray(step_loss_list)
                    print (step_loss_array.shape) # 10 element vector

                    step_loss_path = self.output_path + "/losses/loss_steps" + str(t-last_burn-1) + ".csv"
                    np.savetxt(step_loss_path, step_loss_array, fmt='%.5f', delimiter=',')


                # No evaluation while training
                # if t % (self.eval_freq * self.train_freq) == 0:
                #     episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 1, eval_count, max_episode_length, True)
                #     save_scalar(t, 'eval/eval_episode_reward_mean', episode_reward_mean, self.writer)
                #     save_scalar(t, 'eval/eval_episode_reward_std', episode_reward_std, self.writer)

        self.save_model(idx_episode)


    def save_model(self, idx_episode):
        safe_path = self.output_path + "/qnet" + str(idx_episode) + ".h5"
        self.q_network.save_weights(safe_path)
        print("Network at", idx_episode, "saved to:", safe_path)

    def evaluate(self, env, num_episodes, eval_count, max_episode_length=None, monitor=False):
        """Test your agent with a provided environment.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("Evaluation starts.")

        is_training = False
        if self.load_network:
            self.q_network.load_weights(self.load_network_path)
            print("Load network from:", self.load_network_path)
        # if monitor:
        #     env = wrappers.Monitor(env, self.output_path_videos, video_callable=lambda x:True, resume=True)
        state = env.reset()

        idx_episode = 1
        episode_frames = 0
        episode_reward = np.zeros(num_episodes)
        t = 0

        while idx_episode <= num_episodes:
            t += 1
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            action = self.select_action(action_state, is_training, policy_type = 'GreedyEpsilonPolicy')
            state, reward, done, info = env.step(action)
            episode_frames += 1
            episode_reward[idx_episode-1] += reward 
            if episode_frames > max_episode_length:
                done = True
            if done:
                print("Eval: time %d, episode %d, length %d, reward %.0f" %
                    (t, idx_episode, episode_frames, episode_reward[idx_episode-1]))
                eval_count += 1
                save_scalar(eval_count, 'eval/eval_episode_raw_reward', episode_reward[idx_episode-1], self.writer)
                save_scalar(eval_count, 'eval/eval_episode_raw_length', episode_frames, self.writer)
                sys.stdout.flush()
                state = env.reset()
                episode_frames = 0
                idx_episode += 1
                self.atari_processor.reset()
                self.history_processor.reset()

        reward_mean = np.mean(episode_reward)
        reward_std = np.std(episode_reward)
        print("Evaluation summury: num_episodes [%d], reward_mean [%.3f], reward_std [%.3f]" %
            (num_episodes, reward_mean, reward_std))
        sys.stdout.flush()

        return reward_mean, reward_std, eval_count
