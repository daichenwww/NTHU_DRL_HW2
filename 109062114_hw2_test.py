import gym
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import random
import os

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def frame_preprocessing(observation):
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(observation, (84, 84),
                             interpolation=cv2.INTER_AREA)
    observation = np.expand_dims(observation, axis=0)
    observation = torch.tensor(observation, dtype=torch.float32)
    return observation

class DQN(torch.nn.Module):
    def __init__(self, input_channels=4, num_actions=12):
        super().__init__()
        # 4x84x84 -> 32x8x8 -> 64x4x4 -> 64x3x3 -> 512 -> num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DuelingDQN(torch.nn.Module):
    def __init__(self, input_channels=4, num_actions=12):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(y.size(0), -1)
        advantage = self.advantage(y)
        value = self.value(y)
        adv_mean = advantage.mean(1, keepdim=True)
        return value + advantage - adv_mean

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # load model state dict as cpu mode
        self.learning_Q = DuelingDQN().to('cpu')
        self.learning_Q.load_state_dict(torch.load(os.getcwd() + '/109062114_hw2_data', map_location='cpu'))
        self.learning_Q.eval()
        self.state_stack = None # 4x84x84

    def act(self, observation):
        # if state_stack is empty, fill it with the same frame
        observation = frame_preprocessing(observation) # 1x84x84
        if self.state_stack is None: 
            self.state_stack = observation
            self.state_stack = torch.cat([self.state_stack] * 4, dim=0)
        else:
            self.state_stack = torch.cat([self.state_stack[1:], observation], dim=0)
        with torch.no_grad():
            q_values = self.learning_Q(self.state_stack.unsqueeze(0))
            return torch.max(q_values, 1)[1].data.cpu().numpy()[0]

# agent = Agent()
# done = False
# state = env.reset()
# total_reward = 0
# while True:
#     if done:
#         break
#     action = agent.act(state)
#     state, reward, done, info = env.step(action)
#     total_reward += reward
#     env.render()

# print('Total Reward:', total_reward)
# env.close()