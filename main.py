#!/usr/bin/env python

import argparse
import os
import gym
from gym import wrappers
import tensorflow as tf
from future.builtins import input

# >>>>>>>>>>>>>>>>>>>>>>>>
# Different implementation of DQNAgent
# Uncomment the one you want to train and test

# Keras implementation. Includes Basic Dueling Double DQN and Temporal Attention DQN.
from deeprl_prj.dqn_keras import DQNAgent

# Pure Tensorflow implementation. Includes Basic Dueling Double DQN and Temporal Attention DQN.
# from deeprl_prj.dqn_tf_temporalAt import DQNAgent

# Pure Tensorflow implementation. Includes Basic Dueling Double DQN and Spatial Attention DQN.
# from deeprl_prj.dqn_tf_spatialAt import DQNAgent

# <<<<<<<<<<<<<<<<<<<<<<<<<

def get_output_folder(args, parent_dir, env_name, task_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1][0])
            print(folder_name)
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id) + '-' + task_name
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print('===== Folder did not exist; creating... %s'%parent_dir)
    else:
        print('===== Folder exists; delete? %s'%parent_dir)
        response = input("Press Enter to continue...")
        os.system('rm -rf %s/' % (parent_dir))
    os.makedirs(parent_dir+'/videos/')
    os.makedirs(parent_dir+'/images/')
    os.makedirs(parent_dir+'/losses/')
    return parent_dir

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='QuadCopter-v4', help='Atari env name')
    parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial exploration probability in epsilon-greedy')
    parser.add_argument('--final_epsilon', default=0.05, type=float, help='Final exploration probability in epsilon-greedy')
    parser.add_argument('--exploration_steps', default=24000, type=int, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
    parser.add_argument('--num_samples', default=40000, type=int, help='Number of training samples from the environment in training')
    parser.add_argument('--num_frames', default=4, type=int, help='Number of frames to feed to Q-Network')
    parser.add_argument('--frame_width', default=84, type=int, help='Resized frame width')
    parser.add_argument('--frame_height', default=84, type=int, help='Resized frame height')
    parser.add_argument('--replay_memory_size', default=50000, type=int, help='Number of replay memory the agent uses for training')
    parser.add_argument('--target_update_freq', default=200, type=int, help='The frequency with which the target network is updated')
    parser.add_argument('--train_freq', default=4, type=int, help='The frequency of actions wrt Q-network update')
    parser.add_argument('--save_freq', default=500, type=int, help='The frequency with which the network is saved')
    parser.add_argument('--eval_freq', default=200, type=int, help='The frequency with which the policy is evlauted')    
    parser.add_argument('--num_burn_in', default=10000, type=int, help='Number of steps to populate the replay memory before training starts')
    parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
    parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')
    parser.add_argument('--net_mode', default='dqn', help='choose the mode of net, can be linear, dqn, duel')
    parser.add_argument('--max_episode_length', default = 1000, type=int, help = 'max length of each episode')
    parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
    parser.add_argument('--ddqn', default=False, dest='ddqn', action='store_true', help='enable ddqn')
    parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
    parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
    parser.add_argument('--no_experience', default=False, action='store_true', help='do not use experience replay')
    parser.add_argument('--no_target', default=False, action='store_true', help='do not use target fixing')
    parser.add_argument('--monitor', default=False, action='store_true', help='record video')
    parser.add_argument('--task_name', default='', help='task name')
    parser.add_argument('--recurrent', default=False, dest='recurrent', action='store_true', help='enable recurrent DQN')
    parser.add_argument('--a_t', default=False, dest='a_t', action='store_true', help='enable temporal/spatial attention')
    parser.add_argument('--global_a_t', default=False, dest='global_a_t', action='store_true', help='enable global temporal attention')
    parser.add_argument('--selector', default=False, dest='selector', action='store_true', help='enable selector for spatial attention')
    parser.add_argument('--bidir', default=False, dest='bidir', action='store_true', help='enable two layer bidirectional lstm')

    args = parser.parse_args()
    args.output = get_output_folder(args, args.output, args.env, args.task_name)

    env = gym.make(args.env)
    print("==== Output saved to: ", args.output)
    print("==== Args used:")
    print(args)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    num_actions = env.action_space.n
    print(">>>> Game ", args.env, " #actions: ", num_actions)

    dqn = DQNAgent(args, num_actions)
    if args.train:
        print(">> Training mode.")
        dqn.fit(env, args.num_samples, args.max_episode_length)
    else:
        print(">> Evaluation mode.")
        dqn.evaluate(env, args.num_episodes_at_test, 0, args.max_episode_length, args.monitor)

if __name__ == '__main__':
    main()
