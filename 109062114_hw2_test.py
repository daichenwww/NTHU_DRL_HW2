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
    observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
    observation = observation / 255.0
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
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
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
        self.learning_Q = DuelingDQN(num_actions=12).to('cpu')
        self.learning_Q.load_state_dict(torch.load(os.getcwd() + '/109062114_hw2_data', map_location='cpu'))
        self.learning_Q.eval()
        self.state_stack = None # 4x84x84
        self.last_action = 0
        self.frame_skip = 0
        self.time_count = 0

    def act(self, observation):
        # if state_stack is empty, fill it with the same frame
        if self.time_count > 4000:
            self.state_stack = None
            self.frame_skip = 0
            self.time_count = 0
            self.last_action = 0
            return 0
        observation = frame_preprocessing(observation) # 1x84x84
        if self.frame_skip % 4 == 0:
            if self.state_stack is None: 
                self.state_stack = observation
                self.state_stack = torch.cat([self.state_stack] * 4, dim=0)
            else:
                self.state_stack = torch.cat([self.state_stack[1:], observation], dim=0)
            self.last_action = self.choose_action(self.state_stack.unsqueeze(0))
        self.frame_skip += 1
        self.time_count += 1
        return self.last_action
            
    def choose_action(self, state):
        if random.random() < 0.01:
            return random.choice([1, 2, 5, 6, 7 ])
        else:
            with torch.no_grad():
                q_values = self.learning_Q(state)
                return torch.max(q_values, 1)[1].data.cpu().numpy()[0]
        
if __name__ == '__main__':
    agent = Agent()
    done = False
    state = env.reset()
    total_reward = 0
    while True:
        if done:
            break
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        # print(info['x_pos'], info['y_pos'])
        total_reward += reward
        env.render()

    print('Total Reward:', total_reward)
    env.close()
