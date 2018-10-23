'''Pure Tensorflow implementation. Includes Basic Dueling Double DQN and Temporal Attention DQN.'''

from deeprl_prj.policy import *
from deeprl_prj.objectives import *
from deeprl_prj.preprocessors import *
from deeprl_prj.utils import *
from deeprl_prj.core import *
from helper import *

import numpy as np
import sys
from gym import wrappers
import tensorflow as tf
print(tf.__version__)

"""Main DQN agent."""

class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, rnn_cell_1, myScope, rnn_cell_2=None):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.imageIn =  tf.placeholder(shape=[None,84,84,num_frames],dtype=tf.float32)
        self.image_permute = tf.transpose(self.imageIn, perm=[0, 3, 1, 2])
        self.image_reshape = tf.reshape(self.image_permute, [-1, 84, 84, 1])
        self.image_reshape_recoverd = tf.squeeze(tf.gather(tf.reshape(self.image_reshape, [-1, num_frames, 84, 84, 1]), [0]), [0])
        self.summary_merged = tf.summary.merge([tf.summary.image('image_reshape_recoverd', self.image_reshape_recoverd, max_outputs=num_frames)])
        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,1])
        self.conv1 = tf.contrib.layers.convolution2d( \
            inputs=self.image_reshape,num_outputs=32,\
            kernel_size=[8,8],stride=[4,4],padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv1')
        self.conv2 = tf.contrib.layers.convolution2d( \
            inputs=self.conv1,num_outputs=64,\
            kernel_size=[4,4],stride=[2,2],padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv2')
        self.conv3 = tf.contrib.layers.convolution2d( \
            inputs=self.conv2,num_outputs=64,\
            kernel_size=[3,3],stride=[1,1],padding='VALID', \
            activation_fn=tf.nn.relu, biases_initializer=None,scope=myScope+'_conv3')
        self.conv4 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv3), h_size, activation_fn=tf.nn.relu)
        
        #We take the output from the final convolutional layer and send it to a recurrent layer.
        #The input must be reshaped into [batch x trace x units] for rnn processing, 
        #and then returned to [batch x units] when sent through the upper levels.
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.convFlat = tf.reshape(self.conv4,[self.batch_size, num_frames, h_size])
        self.state_in_1 = rnn_cell_1.zero_state(self.batch_size, tf.float32)

        if args.bidir:
            self.state_in_2 = rnn_cell_2.zero_state(self.batch_size, tf.float32)
            self.rnn_outputs_tuple, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(\
                cell_fw=rnn_cell_1, cell_bw=rnn_cell_2, inputs=self.convFlat, dtype=tf.float32, \
                initial_state_fw=self.state_in_1, initial_state_bw=self.state_in_2, scope=myScope+'_rnn')
            # print "====== len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0] ", len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0].get_shape().as_list(), self.rnn_outputs_tuple[1].get_shape().as_list() # [None, 10, 512]
            # As we have Bi-LSTM, we have two output, which are not connected. So merge them
            self.rnn_outputs = tf.concat([self.rnn_outputs_tuple[0], self.rnn_outputs_tuple[1]], axis=2)
            # self.rnn_outputs = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.rnn_outputs_double), h_size, activation_fn=None)
            self.rnn_output_dim = h_size * 2
        else:
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(\
                inputs=self.convFlat,cell=rnn_cell_1, dtype=tf.float32, \
                initial_state=self.state_in_1, scope=myScope+'_rnn')
            # print "====== self.rnn_outputs ", self.rnn_outputs.get_shape().as_list() # [None, 10, 512]
            self.rnn_output_dim = h_size

        # attention machanism
        if not(args.a_t):
            self.rnn_last_output = tf.slice(self.rnn_outputs, [0, num_frames-1, 0], [-1, 1, -1])
            self.rnn = tf.squeeze(self.rnn_last_output, [1])
        else:
            if args.global_a_t:
                self.rnn_outputs_before = tf.slice(self.rnn_outputs, [0, 0, 0], [-1, num_frames-1, -1])
                self.attention_v = tf.reshape(tf.slice(self.rnn_outputs, [0, num_frames-1, 0], [-1, 1, -1]), [-1, self.rnn_output_dim, 1])
                self.attention_va = tf.tanh(tf.matmul(self.rnn_outputs_before, self.attention_v))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs_before, self.attention_a), axis=1)
                self.rnn = tf.concat([self.rnn, tf.squeeze(tf.slice(self.rnn_outputs, [0, num_frames-1, 0], [-1, 1, -1]), [1])], axis=1)
            else:
                with tf.variable_scope(myScope+'_attention'):
                    self.attention_v = tf.get_variable(name='atten_v', shape=[self.rnn_output_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
                self.attention_va = tf.tanh(tf.map_fn(lambda x: tf.matmul(x, self.attention_v), self.rnn_outputs))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs, self.attention_a), axis=1)
        # print "========== self.rnn ", self.rnn.get_shape().as_list() #[None, 1024]

        if args.net_mode == "duel":
            #The output from the recurrent player is then split into separate Value and Advantage streams
            self.ad_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu, scope=myScope+'_fc_advantage_hidden')
            self.Advantage = tf.contrib.layers.fully_connected(self.ad_hidden, num_actions, activation_fn=None, scope=myScope+'_fc_advantage')
            self.value_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu, scope=myScope+'_fc_value_hidden')
            self.Value = tf.contrib.layers.fully_connected(self.value_hidden, 1, activation_fn=None, scope=myScope+'_fc_value')       
            #Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        else:
            self.Qout = tf.contrib.layers.fully_connected(self.rnn, num_actions, activation_fn=None)

        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

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

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the class 
    provides.

    We have provided docstrings to go along with our suggested API.

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
        self.output_path_images = args.output + '/images/'
        self.save_freq = args.save_freq
        self.load_network = args.load_network
        self.load_network_path = args.load_network_path
        self.enable_ddqn = args.ddqn
        self.net_mode = args.net_mode
        self.args = args

        self.h_size = 512
        self.tau = 0.001
        # self.q_network = create_model(input_shape, num_actions, self.net_mode, args, "QNet")
        # self.target_network = create_model(input_shape, num_actions, self.net_mode, args, "TargetNet")
        tf.reset_default_graph()
        #We define the cells for the primary and target q-networks
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        cellT = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
        if args.bidir:
            cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            cellT_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            self.q_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cell, rnn_cell_2=cell_2, myScope="QNet")
            self.target_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cellT, rnn_cell_2=cellT_2, myScope="TargetNet")
        else:
            self.q_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cell, myScope="QNet")
            self.target_network = Qnetwork(args, h_size=self.h_size, num_frames=self.num_frames, num_actions=self.num_actions, rnn_cell_1=cellT, myScope="TargetNet")
        
        print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
        self.eval_freq = args.eval_freq
        self.no_experience = args.no_experience
        self.no_target = args.no_target
        print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

        # initialize target network
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        trainables = tf.trainable_variables()
        print(trainables, len(trainables))
        self.targetOps = updateTargetGraph(trainables, self.tau)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        updateTarget(self.targetOps, self.sess)
        self.writer = tf.summary.FileWriter(self.output_path)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        # return self.q_network.predict_on_batch(state)
        # print state.shape
        # Qout = self.sess.run(self.q_network.rnn_outputs,\
        #             feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        Qout = self.sess.run(self.q_network.Qout,\
                    feed_dict={self.q_network.imageIn: state, self.q_network.batch_size:1})
        # print Qout.shape
        return Qout

    def select_action(self, state, is_training = True, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

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
            # action_mask = np.zeros((batch_size, self.num_actions))
            # action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            # next_qa_value = self.target_network.predict_on_batch(next_states)
            next_qa_value = self.sess.run(self.target_network.Qout,\
                    feed_dict={self.target_network.imageIn: next_states, self.target_network.batch_size:batch_size})

        if self.enable_ddqn:
            # qa_value = self.q_network.predict_on_batch(next_states)
            qa_value = self.sess.run(self.q_network.Qout,\
                    feed_dict={self.q_network.imageIn: next_states, self.q_network.batch_size:batch_size})
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        # print rewards.shape, mask.shape, next_qa_value.shape, batch_size
        target = rewards + self.gamma * mask * next_qa_value

        if self.args.a_t and np.random.random()<1e-3:
            loss, _, rnn, attention_v, attention_a = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn, self.q_network.attention_v, self.q_network.attention_a], \
                        feed_dict={self.q_network.imageIn: states, self.q_network.batch_size:batch_size, \
                        self.q_network.actions: actions, self.q_network.targetQ: target})
            # print(attention_a[0])
        else:
            loss, _, rnn = self.sess.run([self.q_network.loss, self.q_network.updateModel, self.q_network.rnn], \
                        feed_dict={self.q_network.imageIn: states, self.q_network.batch_size:batch_size, \
                        self.q_network.actions: actions, self.q_network.targetQ: target})
        
        return loss, np.mean(target)

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
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
        for t in range(self.num_burn_in + num_iterations):
            action_state = self.history_processor.process_state_for_network(
                self.atari_processor.process_state_for_network(state))
            policy_type = "UniformRandomPolicy" if burn_in else "LinearDecayGreedyEpsilonPolicy"
            action = self.select_action(action_state, is_training, policy_type = policy_type)
            processed_state = self.atari_processor.process_state_for_memory(state)

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

            if not burn_in:
                if t % self.train_freq == 0:
                    loss, target_value = self.update_policy(current_sample)
                    episode_loss += loss
                    episode_target_value += target_value
                # update freq is based on train_freq
                if t % (self.train_freq * self.target_update_freq) == 0:
                    # self.target_network.set_weights(self.q_network.get_weights())
                    updateTarget(self.targetOps, self.sess)
                    print("----- Synced.")
                if t % self.save_freq == 0:
                    self.save_model(idx_episode)
                if t % (self.eval_freq * self.train_freq) == 0:
                    episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 20, eval_count, max_episode_length, True)
                    save_scalar(t, 'eval/eval_episode_reward_mean', episode_reward_mean, self.writer)
                    save_scalar(t, 'eval/eval_episode_reward_std', episode_reward_std, self.writer)

        self.save_model(idx_episode)


    def save_model(self, idx_episode):
        safe_path = self.output_path + "/qnet" + str(idx_episode) + ".cptk"
        self.saver.save(self.sess, safe_path)
        # self.q_network.save_weights(safe_path)
        print("+++++++++ Network at", idx_episode, "saved to:", safe_path)

    def restore_model(self, restore_path):
        self.saver.restore(self.sess, restore_path)
        print("+++++++++ Network restored from: %s", restore_path)

    def evaluate(self, env, num_episodes, eval_count, max_episode_length=None, monitor=True):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print("Evaluation starts.")
        plt.figure(1, figsize=(45, 20))

        is_training = False
        if self.load_network:
            # self.q_network.load_weights(self.load_network_path)
            # print("Load network from:", self.load_network_path)
            self.restore_model(self.load_network_path)
        if monitor:
            env = wrappers.Monitor(env, self.output_path_videos, video_callable=lambda x:True, resume=True)
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

            action_state_ori = self.history_processor.process_state_for_network_ori(
                self.atari_processor.process_state_for_network_ori(state))

            dice = np.random.random()

            state, reward, done, info = env.step(action)

            if dice < 0.1:
                attention_a = self.sess.run(self.q_network.attention_a,\
                            feed_dict={self.q_network.imageIn: action_state[None, :, :, :], self.q_network.batch_size:1})
                # print attention_a.shape #(1, 10, 1)
                attention_a = np.reshape(attention_a, (-1))
                for alpha_idx in range(action_state_ori.shape[3]):
                    plt.subplot(2, action_state_ori.shape[3]//2+1, alpha_idx+1)
                    img = action_state_ori[:, :, :, alpha_idx] #(210, 160, 3)
                    plt.imshow(img)
                    # plt.text(0, 1, 'Weight: %.4f'%(att ention_a[alpha_idx]) , color='black', weight='bold', backgroundcolor='white', fontsize=30)
                plt.subplot(2, action_state_ori.shape[3]//2+1, action_state_ori.shape[3]+2)
                plt.imshow(state)
                # plt.text(0, 1, 'Next state after taking the action %s'%(action), color='black', weight='bold', backgroundcolor='white', fontsize=20)
                plt.axis('off')
                plt.savefig('%sattention_ep%d-frame%d.png'%(self.output_path_images, eval_count, episode_frames))
                print('---- Image saved at: %sattention_ep%d-frame%d.png'%(self.output_path_images, eval_count, episode_frames))

            episode_frames += 1
            episode_reward[idx_episode-1] += reward 
            if episode_frames > max_episode_length:
                done = True
            if done:
                print("Eval: time %d, episode %d, length %d, reward %.0f. @eval_count %s" %
                    (t, idx_episode, episode_frames, episode_reward[idx_episode-1], eval_count))
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
