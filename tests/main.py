import gym
import sys
import numpy as np

sys.path.append("../gym_tictactoe")

import gym_tictactoe  # NOQA


def test_normal():
    print("--- test_normal ---")
    env = gym.make("tictactoe-v0")
    env.seed(0)
    obs = env.reset()
    done = False

    while not done:
        env.render()
        print(obs)
        action = np.random.choice(obs["legal_actions"])
        next_obs, reward, done, _ = env.step(action)
        print(f"Action[{action}], Reward[{reward}]")
        obs = next_obs
    env.render()
    print(obs)
    env.close()


def test_draw():
    print("--- test_draw ---")
    env = gym.make("tictactoe-v0")
    obs = env.reset()
    done = False
    actions = [0, 1, 2, 3, 5, 4, 6, 8, 7]

    for action in actions:
        obs, reward, done, _ = env.step(action)
        print(f"Action[{action}], Reward[{reward}]")
    env.render()
    print(obs)
    env.close()


if __name__ == "__main__":
    test_normal()
    test_draw()

