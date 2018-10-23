# UAV Obstacle Avoidance using Deep Recurrent Reinforcement Learning with Temporal Attention

The code is implemented in Tensorflow(version = 1.1.0) and Keras.  

## Requirements

The code is based on **Python 2**. Install dependency by running:

	pip install --user -r requirements.txt
    
## How to run

There are two types of DQN implementation with gpu: Keras and Tensorflow.  
You can choose different implementation by altering **line 15** in 
**main.py**

Train original DQN:

	python main.py --task_name 'DQN'
    
Train Double DQN:

	python main.py --ddqn --task_name 'Double_DQN'
    
Train Dueling DQN:

	python main.py --net_mode=duel --task_name 'Dueling_DQN'

Train Recurrent DQN:

	python main.py --num_frames 10 --recurrent --task_name 'Recurrent_DQN'
    
Train Recurrent Temporal Attention DQN: (Using **dqn_tf_temporalAt.py** by uncommenting **line 18** in **main.py**)

	python main.py --num_frames 10 --recurrent --a_t --selector --task_name 'TemporalAt_DQN'

Train Recurrent Spatial Attention DQN: (Using **dqn_tf_spatialAt.py** by uncommenting **line 21** in **main.py**)

	python main.py --num_frames 10 --recurrent --a_t --selector --task_name 'SpatialAt_DQN'

Test trained model (e.g. Spatial Attention DQN):

	python main.py --num_frames 10 --recurrent --a_t --selector --test \
    --load_network --load_network_path=PATH_TO_NET

## Acknowledgement

This code repository is highly inspired from work of Rui Zhu et al [link](https://github.com/chasewind007/Attention-DQN).
