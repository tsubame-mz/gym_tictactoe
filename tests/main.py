import gym
import sys
import numpy as np

sys.path.append("../gym_tictactoe")

import gym_tictactoe  # NOQA

if __name__ == "__main__":
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
