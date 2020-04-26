import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class QuartoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, first_piece = 'random'):
        
        if first_piece == 'random':
            self.state = [np.zeros((4,4,4)), np.random.choice([-1, 1], size = 4), 1]
        elif type(first_piece) is np.ndarray and [a in [-1, 1] for a in first_piece] == [True for i in range(4)]:
            self.state = [np.zeros((4,4,4)), first_piece, 1]
        else:
            raise ValueError("first_piece must be 'random' or np.array of length 4, containing 1s and -1s")
        
        self.counter = 0                    #count actions in episode
        self.reward = 0
        self.done = False
        self.info = {}
        
    def check_if_done(self):
        if (np.abs(self.state[0].sum(axis = 0)) == 4).any():
            return True
        elif (np.abs(self.state[0].sum(axis = 1)) == 4).any():
            return True
        elif (np.abs(self.state[0][[0, 1, 2, 3], [0, 1, 2, 3],:].sum(axis = 0)) == 4).any():
            return True
        elif (np.abs(self.state[0][[0, 1, 2, 3], [3, 2, 1, 0],:].sum(axis = 0)) == 4).any():
            return True
        else:
            return False
        
    def step(self, action):
        # expected format of action: 
        # 2-tupel: 
        # first element: 2-Tupel, int-valued entries, range: 0 - 3 incl.
        # second element: np.array of shape (4,), entries in {-1, 1}
        if self.done:
            print('Game is already over!')
            return [self.state, self.reward, self.done, self.info]
        
        position, next_piece = action
        
        if not type(position) is tuple or not len(position) == 2:
            raise TypeError('action[0] must be tuple of length 2.')
        if not type(next_piece) or not next_piece.shape == (4,):
            raise TypeError('action[1] must be np.ndarray of shape (4,)')
        if not (np.abs(next_piece) == 1).all():
            raise ValueError('next_piece must only contain -1s and 1s')
        
        self.state[0][position[0], position[1], :] = self.state[1]
        self.state[1] = next_piece
        self.state[2] *= -1
        
        self.counter += 1
        self.done = self.check_if_done()
        
        if self.done:
            self.reward = -100 * self.state[2] 
        
        return self.state, self.reward, self.done, self.info
        
    def reset(self, first_piece = 'random'):
        if first_piece == 'random':
            self.state = [np.zeros((4,4,4)), np.random.choice([-1, 1], size = 4), 1]
        elif type(first_piece) is np.ndarray and [a in [-1, 1] for a in first_piece] == [True for i in range(4)]:
            self.state = [np.zeros((4,4,4)), first_piece, 1]
        else:
            raise ValueError("first_piece must be 'random' or np.array of length 4, containing 1s and -1s")
        
        self.counter = 0                    #count actions in episode
        self.reward = 0
        self.done = False
        self.info = {}
        
        return self.state
        
    def render(self):
        D = np.array([['l', 'd'], ['s', 'b'], ['c', 's'], ['p', 'w']])
        rendered_M = np.array(['-oo-' for i in range(16)]).reshape((4,4))
        for i in range(4):
            for j in range(4):
                if not self.state[0][i,j,0] == 0:
                    norm_ind = ((self.state[0][i, j, :] + 1) // 2).astype(int)
                    rendered_M[i, j] = ''.join(D[[0,1,2,3], norm_ind])                    
        print(rendered_M)       