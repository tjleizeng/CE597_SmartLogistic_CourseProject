import math
import random

import torch as th
from torch.optim import RMSprop
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.distributions import Categorical
import numpy as np

from memory import Transition, ReplayMemory

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 1

device = th.device("cpu")#"cuda" if th.cuda.is_available() else "cpu")

class QLearner:
    def __init__(self, input_shape, n_action):
        self.policy_net = FCAgent(input_shape, n_action).to(device)
        self.optimiser = RMSprop(params=self.policy_net.parameters(), lr=0.001)
        self.target_net = copy.deepcopy(self.policy_net).to(device)
        self.memory = ReplayMemory(10000)
        self.n_action = n_action

        self.steps_done = 0

    def select_action(self, state, steps_done,masks):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1 * steps_done / EPS_DECAY)
        if (sample > eps_threshold) or (steps_done == -1): # steps_done -1 means test mode
            with th.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return th.argmax(self.policy_net(state)-1e6*(1-np.array(masks))).view(1,1)
        else:
            return th.tensor([np.where(np.array(masks)==1)[0][random.randrange(sum(masks))]], device = device, dtype=th.long).view(1,1)

    def optimize_mode(self):
        if (len(self.memory) < BATCH_SIZE):
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # non_final_mask = th.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), dtype=th.bool)
        # non_final_next_states = th.cat([s for s in batch.next_state
        #                                    if s is not None])
        state_batch = th.stack([th.tensor(i) for i in batch.state]).to(device)
        action_batch = batch.action # th.stack([th.tensor(i) for i in batch.action]).to(device)
        reward_batch = th.stack([th.tensor(i) for i in batch.reward]).to(device)

        state_action_values = self.policy_net(state_batch.float()) #.gather(1, action_batch)
        # print(state_action_values.shape)
        # print(action_batch)
        state_action_values = th.stack([state_action_values[i,action_batch[i]] for i in range(128)])
        next_state_values = th.zeros(BATCH_SIZE, device=device)
        next_state_values = self.target_net(state_batch.float())#.max(1)[0].detach()

        expected_state_action_values = (th.stack([next_state_values[i,action_batch[i]] for i in range(128)]) * GAMMA).flatten() + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(th.reshape(state_action_values,(-1,)), expected_state_action_values)
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

    def update_targets(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Update target network")

    def save_models(self, path):
        th.save(self.policy_net.state_dict(), '{}_model.pth'.format(path))
        th.save(self.optimiser.state_dict(), "{}_opt.th".format(path))

    def load_models(self, path):
        self.policy_net.load_state_dict(th.load('{}_model.pth'.format(path)))
        self.policy_net.eval()
        # Not quite right but I don't want to save target networks
        self.target_net.load_state_dict(th.load('{}_model.pth'.format(path)))
        self.target_net.eval()
        self.optimiser.load_state_dict(th.load("{}_opt.th".format(path), map_location=lambda storage, loc: storage))


class FCAgent(nn.Module):
    def __init__(self, input_shape, n_action):
        super(FCAgent, self).__init__()
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, n_action)

    def init_hidden(self):
        # make hidden states on same device as model
        pass
        #return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        inputs = inputs.to(device)
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = F.relu(self.fc3(h))
        return q