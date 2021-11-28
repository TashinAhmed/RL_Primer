import gym
import random

env = gym.make("CartPole-v1")

def random_games():
    for episode in range(10):
        env.reset()
        for frames in range(500):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(frames, next_state, reward, done, info, action)
            if done:
                break

random_games()