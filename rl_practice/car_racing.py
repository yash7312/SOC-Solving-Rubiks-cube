import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CarRacing-v2", domain_randomize=True)

