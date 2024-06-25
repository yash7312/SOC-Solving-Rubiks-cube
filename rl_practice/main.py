import random
import gym

env = gym.make("CartPole-v1", render_mode="human")

episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    score = 0
    done = False
    
    while not done:
        action = random.choice([0,1])
        _, reward, = env.step(action)
        