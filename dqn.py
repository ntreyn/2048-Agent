from rl_utils import NeuralNet, ReplayMemory, Transition
from rl_agent import RLAgent

from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQN(RLAgent):
    def __init__(self, env, params):
        super().__init__(env, **params.agent)

        self.env = env

        if params.device == 'cpu' or params.device == 'cuda':
            self.device = torch.device(params.device)
        else:
            print("Invalid device: default to cpu")
            self.device = torch.device('cpu')

        self.input_size = self.env.size
        self.hidden_size = 128
        self.output_size = self.env.action_space.n

        self.memory_size = 2560
        self.batch_size = 128
        self.clip = 10
        self.cur_step = 0
        self.episode = 0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.2
        self.decay_rate = 100000
        self.var_epsilon = True

        dqn_net = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.policy_net = dqn_net.to(self.device)
        self.target_net = dqn_net.to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(self.memory_size)
    
    def act(self, state):
        state = self.state_to_tensor(state)
        exp_exp_tradeoff = random.uniform(0,1)

        if self.training:
            self.cur_step += 1

        if self.training and exp_exp_tradeoff < self.epsilon:
            action = self.env.sample_action()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                policy_out = self.policy_net(state.to(self.device))
            self.policy_net.train()

            open_mask = torch.tensor([self.env.possible_actions()], dtype=torch.long, device=self.device)
            open_vals = torch.tensor([[-float('Inf')] * self.env.action_space.n], dtype=torch.float, device=self.device)
            open_vals[0][open_mask] = policy_out[0][open_mask]
            action = open_vals.max(1)[1].view(1, 1).item()

            # action = policy_out[0][open_mask].max(1)[1].view(1, 1).item()

            if action not in self.env.possible_actions():
                import pdb; pdb.set_trace()

        return action
    
    def learn(self, *args):
        self.episode = args[0]

        if self.var_epsilon:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-1.0 * self.cur_step / self.decay_rate)

        if self.memory.size() <= self.batch_size:
            return

        batch = self.sample()

        non_final_mask = torch.tensor(tuple([s is not None for s in batch.next_state]),
                                        device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)

        state_batch = state_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)

        action_batch = torch.cat(batch.action)
        action_batch = action_batch.to(self.device)
        reward_batch = torch.cat(batch.reward)

        state_q_vals = self.policy_net(state_batch).gather(1, action_batch)

        next_state_vals = torch.zeros(self.batch_size, device=self.device)

        next_state_vals[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_q_values = (next_state_vals * self.gamma) + reward_batch

        loss = F.mse_loss(state_q_vals, expected_state_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()

        if self.cur_step % 2 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def push(self, *args):
        state = self.state_to_tensor(args[0])
        action = torch.tensor([[args[1]]])
        reward = torch.tensor([args[2]], dtype=torch.float, device=self.device)
        done = torch.tensor([args[4]])

        if not done:
            next_state = self.state_to_tensor(args[3])
        else:
            next_state = args[3]

        self.memory.push(state, action, reward, next_state, done)

    def sample(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*list(zip(*transitions)))
        return batch

    def state_to_tensor(self, state):
        state_list = list(state)
        state_tensor = torch.tensor([state_list], dtype=torch.float)
        return state_tensor

