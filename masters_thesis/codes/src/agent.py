import random
import math
import torch
import torch.nn.functional as F
from replaymemory import ReplayMemory
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import utils


class Agent:
    def __init__(self, policy_net, target_net, durability, optimizer, name, constants):
        """An agent class that takes action on the environment and optimizes
        the action based on the reward.

        Parameters
        ----------
        policy_net : DQN
            [description]
        target_net : DQN
            [description]
        durability : int
            [description]
        optimizer : [type]
            [description]
        name : str
            The name of agent
        constants: Constants
            The hyper-parameters from Constants class
        """
        self.CONSTANTS = constants
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(policy_net.state_dict())
        self.durability = durability
        self.optimizer = optimizer
        self.name = name
        self.memory = ReplayMemory(self.CONSTANTS.MEMORY_SIZE)
        self.steps_done = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.n_best = 0
        self.policy_net_flag = False

    def select_action(self, state, is_first=False):
        sample = random.random()
        eps_threshold = self.CONSTANTS.EPS_END + (self.CONSTANTS.EPS_START - self.CONSTANTS.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.CONSTANTS.EPS_DECAY)
        self.steps_done += 1
        if is_first:
            self.writer.add_graph(self.policy_net, input_to_model=state.to(self.CONSTANTS.DEVICE),
                                  profile_with_cuda=True)
        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net_flag = True
                return self.policy_net(state.to(self.CONSTANTS.DEVICE)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.CONSTANTS.N_ACTIONS)]],
                                device=self.CONSTANTS.DEVICE, dtype=torch.long)

    def select_core_action(self, best_agent_state, flag, best_agent_action):
        self.steps_done += 1
        if flag:
            with torch.no_grad():
                if best_agent_state is None:
                    return self.policy_net(self.state.to(self.CONSTANTS.DEVICE)).max(1)[1].view(1, 1)
                else:
                    return self.policy_net(best_agent_state.to(self.CONSTANTS.DEVICE)).max(1)[1].view(1, 1)
        else:
            return best_agent_action

    def optimize_model(self):
        if len(self.memory) < self.CONSTANTS.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.CONSTANTS.BATCH_SIZE)

        # zip(*transitions) unzips the transitions into
        # Transition(*) creates new named tuple
        # batch.state - tuple of all the states (each state is a tensor)
        # batch.next_state - tuple of all the next states (each state is a tensor)
        # batch.reward - tuple of all the rewards (each reward is a float)
        # batch.action - tuple of all the actions (each action is an int)

        # Transition = ReplayMemory.get_transition()
        transition = self.CONSTANTS.TRANSITION
        batch = transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=utils.get_device(), dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cuda')

        state_batch = torch.cat(batch.state).to('cuda')
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.CONSTANTS.BATCH_SIZE, device=self.CONSTANTS.DEVICE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.CONSTANTS.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def set_tf_writer(self, path):
        self.writer = self._set_tf_writer(path)

    def _set_tf_writer(self, path):
        if self.name == "core":
            writer = SummaryWriter(log_dir="{}/tf-board/core/".format(path))
        else:
            writer = SummaryWriter(log_dir="{}/tf-board/{}".format(path, self.name))
        return writer

    def get_state(self):
        return self.state

    def get_next_state(self):
        return self.next_state

    def get_init_state(self):
        return self.init_state

    def get_name(self):
        return self.name

    def get_policy_net_flag(self):
        return self.policy_net_flag

    def set_init_state(self, state):
        self.init_state = state

    def set_state(self, state):
        self.state = state
        self.next_state = state

    def set_env(self, env):
        self.env = env

    def get_env(self):
        return self.env

    def set_action(self, action):
        self.action = action

    def get_action(self):
        return self.action

    def get_durability(self):
        return self.durability

    def get_policy_net(self):
        return self.policy_net

    def reduce_durability(self, value):
        self.durability = self.durability - value

    def heal_durability(self, value):
        self.durability = self.durability + value

    def set_done_state(self, done):
        self.done = done

    def set_total_reward(self, reward):
        self.reward = reward
        self.total_reward += reward

    def reset_total_reward(self):
        self.total_reward = 0.0

    def get_reward(self):
        return self.reward

    def best_counter(self):
        self.n_best += 1

    def get_n_best(self):
        return self.n_best

    def get_total_reward(self):
        return self.total_reward

    def set_step_retrun_value(self, obs, done, info):
        self.obs = obs
        self.done = done
        self.info = info

    def is_done(self):
        return self.done
