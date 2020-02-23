from typing import Tuple
import gym
import numpy as np


class TicTacToeEnv(gym.Env):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    PLAYER_NUM = 2
    NUM_CELLS = 9
    TOKEN_LIST = {BLACK: "o", WHITE: "x", EMPTY: "-"}
    LINE_MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.full(self.NUM_CELLS, self.EMPTY).astype(int)
        self.player = self.BLACK
        self.done = False
        return self.get_obs()

    def step(self, action: int) -> Tuple:
        reward = 0.0
        if self.done:
            pass
        elif action not in self.current_legal_actions:
            reward = -1.0
            self.done = True
        else:
            self.board[action] = self.player
            self.done = self.judge()
            reward = +1.0 if self.done else 0.0
            self.player = (self.player + 1) % self.PLAYER_NUM

        return self.get_obs(), reward, self.done, {}

    def render(self, mode="human"):
        print("+" + "-" * 3 + "+")
        for y in range(3):
            print("|", end="")
            for x in range(3):
                idx = y * 3 + x
                print(self.TOKEN_LIST[self.board[idx]], end="")
            print("|")
        print("+" + "-" * 3 + "+")
        print(f"Player[{self.TOKEN_LIST[self.player]}({self.player})], Done[{self.done}]")

    def seed(self, seed=None):
        pass

    def close(self):
        pass

    def get_obs(self) -> np.ndarray:
        self.current_legal_actions = self.legal_actions
        return {"board": self.conditioned_board, "legal_actions": self.current_legal_actions, "to_play": self.player}

    def judge(self) -> bool:
        for mask in self.LINE_MASKS:
            line = self.board[mask]
            hit = np.all(np.where(line == self.player, True, False))
            if hit:
                return True
        return False

    @property
    def conditioned_board(self) -> np.ndarray:
        return np.array(
            [
                np.where(self.board == self.BLACK, 1, 0).reshape((3, 3)),
                np.where(self.board == self.WHITE, 1, 0).reshape((3, 3)),
                np.full((3, 3), self.player),
            ]
        )

    @property
    def legal_actions(self) -> np.ndarray:
        return np.where(self.board == self.EMPTY)[0].astype(np.int)
