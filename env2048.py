import numpy as np 

import gym
import random


class env_2048(gym.Env):

    """
    Action mapping: 
    0 : left
    1 : right
    2 : up
    3 : down
    """

    def __init__(self):

        self.dim = 4
        self.size = 16
        self.best = 0
        
        self.action_space = gym.spaces.Discrete(self.dim)
        self.observation_space = gym.spaces.Box(low=0, high=self.size + 1, shape=(self.dim, self.dim), dtype=np.uint8)

        self.action_map = {
            0: self.push_left,
            1: self.push_right,
            2: self.push_up,
            3: self.push_down
        }
    
        self.reset()
    
    def reset(self):
        self.board = [0] * self.size
        self.score = 0
        self.reward = 0
        self.spawn_tile(num=2)
        self.done = False
        return self.get_state()


    def step(self, action):
        
        if self.done:
            print("Game already over")
            exit()

        self.reward = 0
        self.action_map[action]()
        self.score += self.reward
        self.spawn_tile()
        
        if self.score > self.best:
            self.best = self.score
        if not self.possible_actions():
            self.done = True
            next_state = None
        else:
            next_state = self.get_state()

        return next_state, self.reward, self.done, {}

    def render(self):

        print("\nScore: {}\nBest:  {}".format(self.score, self.best))

        border = '-' * 17
        print('\n' + border)
        
        for r in range(0, self.size, self.dim):
            for c in range(self.dim):
                tile = self.tile_map(self.board[r + c])
                print('| ' + tile, end=' ')
            print('|\n' + border)
        
        print()

    def get_state(self):
        return tuple(self.board)
    
    def sample_action(self):
        return random.choice(self.possible_actions())

    def possible_actions(self, state=None):
        actions = []

        if state is None:
            state = self.board

        found = False

        for c in range(self.dim):
            if found:
                break

            column = self.get_col(c)
            if 0 in column:
                actions.append(2)
                actions.append(3)
                break

            last_t = 0
            for t in column:
                if t == last_t:
                    actions.append(2)
                    actions.append(3)
                    found = True
                    break
                else:
                    last_t = t

        found = False

        for r in range(self.dim):
            if found:
                break

            row = self.get_row(r)
            if 0 in row:
                actions.append(0)
                actions.append(1)
                break

            last_t = 0
            for t in row:
                if t == last_t:
                    actions.append(0)
                    actions.append(1)
                    found = True
                    break
                else:
                    last_t = t

        return actions

    def push_up(self):
        for c in range(self.dim):
            column = self.get_col(c)
            self.set_col(c, self.combine(column))

    def push_down(self):
        for c in range(self.dim):
            column = list(reversed(self.get_col(c)))
            new_column = list(reversed(self.combine(column)))
            self.set_col(c, new_column)

    def push_left(self):
        for r in range(self.dim):
            row = self.get_row(r)
            self.set_row(r, self.combine(row))

    def push_right(self):
        for r in range(self.dim):
            row = list(reversed(self.get_row(r)))
            new_row = list(reversed(self.combine(row)))
            self.set_row(r, new_row)
            
    def combine(self, input):
        no_zeros = [i for i in input if i != 0]
        output = []
        i = 0

        while i < len(no_zeros):
            if i + 1 < len(no_zeros) and no_zeros[i] == no_zeros[i+1]:
                output.append(no_zeros[i] + 1)
                self.reward += (2 ** (no_zeros[i] + 1))
                i += 2
            else:
                output.append(no_zeros[i])
                i += 1
        
        while len(output) < self.dim:
            output.append(0)
        return output

    def set_col(self, col, input):
        for t in range(col, self.size, self.dim):
            self.board[t] = input.pop(0)

    def set_row(self, row, input):
        for t in range(row * self.dim, self.dim * (row + 1)):
            self.board[t] = input.pop(0)

    def get_col(self, col):
        return [self.board[t] for t in range(col, self.size, self.dim)]

    def get_row(self, row):
        return [self.board[t] for t in range(row * self.dim, self.dim * (row+ 1))]

    def tile_map(self, num):
        if num == 0:
            return ' '
        else:
            return str(2 ** num)

    def empty_tiles(self):
        return [i for i, t in enumerate(self.board) if t == 0]

    def spawn_tile(self, num=1):
        for n in range(num):
            tile = np.random.choice(self.empty_tiles())
            fill_num = np.random.choice([1,2], p=[0.9,0.1])
            self.board[tile] = fill_num