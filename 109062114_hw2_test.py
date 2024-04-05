import gym
import torch
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
import random
import os

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        return observation

    def observation(self, observation):
        return self.permute_orientation(observation)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(
            observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = SkipFrame(env)
# env = GrayScale(env)
# env = ResizeObservation(env, shape=84)
# env = gym.wrappers.FrameStack(env, num_stack=4)
# done = False
# state = env.reset()
# while True:
#     if done:
#         break
#     action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#     state, reward, done, info = env.step(action)
#     print(state.shape) # (4, 84, 84)
#     print(state)
#     exit()

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
    
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # load model state dict as cpu mode
        self.learning_Q = DQN().to('cpu')
        self.learning_Q.load_state_dict(torch.load(os.getcwd() + '/109062114_hw2_data_1000', map_location='cpu'))
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

agent = Agent()
done = False
state = env.reset()
total_reward = 0
while True:
    if done:
        break
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print('Total Reward:', total_reward)
env.close()