from collections import namedtuple, deque
from random import sample
import torch as th

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','mask'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)