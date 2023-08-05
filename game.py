
import random
import numpy as np
import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, gaussian
import os
import io
import base64
import time
import glob
from IPython.display import HTML
import torch.nn.functional as F
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack
from gym.wrappers import TransformReward
from gym.wrappers.monitoring.video_recorder import ImageEncoder


def make_env(env_name, clip_rewards = True, seed = None):
	# complete this function which returns an object 'env' using gym module
	# Use AtariPreprocessing, FrameStack, TransformReward(based on the clip_rewards variable passed in the arguments of the function), check their usage from internet
	# Use FrameStack to stack 4cd  frames
	# TODO
  env = gym.make(env_name)
  env = AtariPreprocessing(env)
  env = FrameStack(env, num_stack=4)
  if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))
  if seed is not None:
        env.seed(seed)

  return env

# Initialize the device based on CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

env_name = "BreakoutNoFrameskip-v4"
env = make_env(env_name)
action_space = env.action_space
observation_space = env.observation_space
state_shape = observation_space.shape
n_actions = action_space.n
# Step 1: Define the model architecture (same as used during training)
class DQNAgent(nn.Module):
	def __init__(self, state_shape, n_actions, epsilon):
		super(DQNAgent, self).__init__()  # Call the nn.Module constructor
    # Calculate the number of input channels for the first convolutional layer
		num_frames = state_shape[0]
    # First Convolutional Layer
		self.conv1 = nn.Conv2d(num_frames, 16, kernel_size=8, stride=4)
		self.relu1 = nn.ReLU()
    # Second Convolutional Layer
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.relu2 = nn.ReLU()

    # Flatten the output for the linear layer
		self.flatten = nn.Flatten()

    # Linear Layer
		conv_out_size = self.calculate_conv_output_size(state_shape)
		self.fc1 = nn.Linear(conv_out_size, 256)
		self.relu3 = nn.ReLU()

    # Final Linear Layer
		self.fc2 = nn.Linear(256, n_actions)

    # Epsilon for exploration in epsilon-greedy policy
		self.epsilon = epsilon
		# TODO
		# Here state_shape is the input shape to the neural network.
		# n_Actions is the number of actions
		# epsilon is the probability to explore, 1-epsilon is the probabiltiy to stick to the best actions
		# initialise a neural network containing the following layers:
		# 1)a convulation layer which accepts size = state_shape, in_channels = 4( state_shape is stacked with 4 frames using FrameStack ), out_channels = 16, kernel_size = 8, stride = 4 followed by ReLU activation
		# 2)a convulation layer, in_channels = 16, out_channels = 32, kernel_size = 4, stride = 2 followed by ReLU activation function
		# 3)layer to convert the output to a 1D output which is fed into a linear Layer with output size = 256 followed by ReLU actiovation
		# 4) linear Layer with output size = 'number of actions'(the qvalues of actions)

	def calculate_conv_output_size(self, state_shape):
		dummy_input = torch.zeros(1, *state_shape)
		dummy_output = self.conv1(dummy_input)
		dummy_output = self.conv2(dummy_output)
		conv_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
		return conv_output_size

	def forward(self, state_t):
		state_t = self.relu1(self.conv1(state_t))
		state_t = self.relu2(self.conv2(state_t))
		state_t = self.flatten(state_t)
		state_t = self.relu3(self.fc1(state_t))
		q_values = self.fc2(state_t)
		return q_values
		# return qvalues generated from the neural network

	def get_qvalues(self, state_t):
		q_values = self.forward(state_t)
		q_values_np = q_values.detach().cpu().numpy()
		return q_values_np
		# returns the numpy array of qvalues from the neural network

	def sample_actions(self, qvalues):
		batch_size = qvalues.shape[0]
		actions = []

		for _ in range(batch_size):
			if random.random() < self.epsilon:
				action = random.randint(0, qvalues.shape[0] - 1)
			else:
				action = qvalues[_].argmax().item()
			actions.append(action)
		return actions
		#TODO
		# sample_Actions based on the qvalues
		# Use epsilon for choosing between best possible current actions of the give batch_size(can be found from the qvalues object passed in argument) based on qvalues vs explorations(random action)
		# return actions
		# pass

    # ... Your other model methods ...




# Step 2: Load the pre-trained model
pretrained_model_path = 'model_9000000.pth'
device = torch.device('cpu')  # If you're using CPU in Colab
model_state_dict = torch.load(pretrained_model_path, map_location=device)
state_shape = env.observation_space.shape

# Step 3: Create the model and load the state_dict
model = DQNAgent(state_shape, n_actions, epsilon=0.1)
model_state_dict = torch.load(pretrained_model_path, map_location=device)
model_state_dict = {k.replace('network.', ''): v for k, v in model_state_dict.items()}

model.eval()

# Step 4: Create the Breakout environment

# Step 5: Run the Breakout game with the agent and visualize it
done = False
observation = env.reset()

frames=[]
while not done:
    # Convert the observation to a tensor and add a batch dimension
    observation_t = torch.tensor([observation], dtype=torch.float32)

    # Forward pass through the model to get action predictions
    with torch.no_grad():
        action_values = model(observation_t)

    # Choose the action with the highest Q-value as the agent's action
    action = torch.argmax(action_values, dim=1).item()

    # Take the chosen action in the environment and get the next observation, reward, and done flag
    observation, reward, done, _ = env.step(action)

    # Render the game screen to see the agent playing
     # Render the game screen and capture the frame
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    
    # Pause the rendering for a short time (optional)
    #time.sleep(0.01)
    import cv2

# ... Your previous code ...

# Display the captured frames as a video
video_path = "breakout_video5.mp4"
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (frames[0].shape[1], frames[0].shape[0]))
i=0
for frame in frames:
    video_writer.write(frame)
    print(i)
    i=i+1

video_writer.release()



