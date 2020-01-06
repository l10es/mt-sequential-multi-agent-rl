import math
import time
from itertools import count

import cloudpickle
import torch
import torch.nn.functional as F
from torch import nn

import utils

SQRT2 = math.sqrt(2.0)
ACT = nn.ReLU


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ACT(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            ACT(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            ACT()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, n_actions)
        )

    # @torch.jit.script_method
    def forward(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        # __constants__ = ['n_actions']

        super(DDQN, self).__init__()

        # self.n_actions = n_actions

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ACT(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            ACT(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            ACT()
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, n_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, 1)
        )

        # def scale_grads_hook(module, grad_out, grad_in):
        #     """scale gradient by 1/sqrt(2) as in the original paper"""
        #     grad_out = tuple(map(lambda g: g / SQRT2, grad_out))
        #     return grad_out
        #
        # self.fc_adv.register_backward_hook(scale_grads_hook)
        # self.fc_val.register_backward_hook(scale_grads_hook)

    # @torch.jit.script_method
    def forward(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        adv = self.fc_adv(x)
        val = self.fc_val(x)

        return val + adv - adv.mean(1).unsqueeze(1)

    # # @torch.jit.script_method
    # def value(self, x):
    #     x = x.float() / 255
    #     x = self.convs(x)
    #     x = x.view(x.size(0), -1)
    #
    #     return self.fc_val(x)


class LanderDQN(nn.Module):
    def __init__(self, n_state, n_actions, nhid=64):
        super(LanderDQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_state, nhid),
            ACT(),
            nn.Linear(nhid, nhid),
            ACT(),
            nn.Linear(nhid, n_actions)
        )

    # @torch.jit.script_method
    def forward(self, x):
        x = self.layers(x)
        return x


class RamDQN(nn.Module):
    def __init__(self, n_state, n_actions):
        super(RamDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_state, 256),
            ACT(),
            nn.Linear(256, 128),
            ACT(),
            nn.Linear(128, 64),
            ACT(),
            nn.Linear(64, n_actions)
        )

    # @torch.jit.script_method
    def forward(self, x):
        return self.layers(x)


class DQNbn(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class NonBatchNormalizedDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(NonBatchNormalizedDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


def train(envs, agents, core_env, core_agent, n_episodes, agent_n, exp, exp_name, render=False,):
    """
    Training step.

    In this code, we use the multi-agents to create candidate for core agent.
    The core agent and environment is main RL set. In addition, each agent has
    own environment and durability. Each agent's reward is checked for the
    specified number of episodes, and if an agent is not selected as the
    best-agent, that agent's durability is reduced.

    Parameters
    ----------
    envs: list of Environment
        List of environment for multi-agent
    agents: list of Agent
        List of multi-agents to create candidates for core_agent
    core_env: Environment
        Main environment of this train step
    core_agent: Agent
        Main agent of this train step
    n_episodes: int
        The number of episodes
    agent_n : int
        The number of agent
    exp: Experiment
        The Experiment object used by hyperdash
    exp_name: str
        The name of experiment
    render: boolean, default False
        Flag for whether to render the environment
    """
    for episode in range(n_episodes):
        # 0. Initialize the environment, state and agent params
        obs = core_env.reset()
        core_state = utils.get_state(obs)
        core_agent.reset_total_reward()
        core_agent.set_state(core_state)
        for agent in agents:
            obs = agent.get_env().reset()
            state = utils.get_state(obs)
            agent.set_state(state)
            agent.reset_total_reward()
            # agent.durability = DEFAULT_DURABILITY

        for t in count():
            # if t % 20 != 0:
            #     print(str(t) + " ", end='')
            # else:
            #     print("\n")
            exp.log("agent_durability:{}".format([agent.get_durability() for agent in agents]))
            for agent in agents:
                agent.writer.add_scalar("internal/durability/{}".format(agent.get_name()), agent.get_durability(),
                                        core_agent.steps_done)
            #     print(str(t) + " ", end='')

            # 1. Select action from environment of each agent
            for agent in agents:
                if agent.get_state() is not None and len(agents) > 1:
                # if agent.get_state() is not None:
                    # agent.set_env(core_agent.get_env())
                    # agent.set_state(core_agent.get_state())
                    # agent.set_init_state(core_agent.get_state())
                    agent.set_init_state(agent.get_state())
                    if episode == 0 and t == 0:
                        action = agent.select_action(agent.get_state(), is_first=True)
                    else:
                        action = agent.select_action(agent.get_state())
                    agent.set_action(action)

            # 2. Proceed step of each agent
            for agent in agents:
                if agent.get_state() is not None:
                # if agent.get_state() is not None and len(agents) > 1:
                    obs, reward, done, info = agent.get_env().step(agent.get_action())
                    agent.set_step_retrun_value(obs, done, info)

                    agent.set_total_reward(reward)
                    # Agent reward value
                    # print("Agent:{}, Reward:{}, State:{}".format(agent.name, reward, agent.get_state()))
                    # print("Agent:{}, Reward:{}".format(agent.name, reward))

                    if not done:
                        next_state = utils.get_state(obs)
                    else:
                        next_state = None

                    reward = torch.tensor([reward], device=agent.CONSTANTS.DEVICE)
                    agent.memory.push(agent.get_state(), agent.get_action().to('cpu'), next_state, reward.to('cpu'))
                    agent.set_state(next_state)

                    if agent.steps_done > agent.CONSTANTS.INITIAL_MEMORY:
                        agent.optimize_model()

                        if agent.steps_done % agent.CONSTANTS.TARGET_UPDATE == 0:
                            agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # print("\n")
            # print([agent.get_total_reward() for agent in agents])
            exp.log([agent.get_total_reward() for agent in agents])
            # print(str(t) + " ", end='')

            # ---------------
            # Proposal method
            # ---------------

            # 3. Select best agent in this step
            if len(agents) > 1:
                best_agent = utils.select_best_agent(agents, core_agent.CONSTANTS.ROULETTE_MODE,
                                                     max_reward=core_agent.CONSTANTS.MAX_REWARD,
                                                     min_reward=core_agent.CONSTANTS.MIN_REWARD)
                # best_agent.best_counter()
                [agent.best_counter() for agent in agents if agent.get_name() == best_agent.get_name()]
                # for agent in agents:
                #     if agent.get_name() == best_agent.get_name():
                #         agent.best_counter()
                core_agent.memory.push(best_agent.get_init_state(), best_agent.get_action().to('cpu'),
                                       best_agent.get_next_state(),
                                       torch.tensor([best_agent.get_reward()],
                                                    device=best_agent.CONSTANTS.DEVICE).to('cpu'))
                for agent in agents:
                    agent.writer.add_scalar("internal/reward/{}/all_step".format(agent.get_name()),
                                            agent.get_total_reward(), core_agent.steps_done)
                    agent.writer.add_scalar("internal/obtained_reward/{}".format(agent.get_name()),
                                            agent.get_obtained_reward(), episode)
                    # core_agent_action = best_agent.get_action()
                    # best_agent_state = best_agent.get_state()
                    # policy_net_flag = best_agent.get_policy_net_flag()
                    # best_agent_action = best_agent.get_action()

                # 3.5 Only best_agent can heal own durability at specific iteration
                if t % core_agent.CONSTANTS.DURABILITY_HEALING_FREQUENCY == 0 and len(agents) > 1:
                    # best_agent.heal_durability(core_agent.CONSTANTS.DEFAULT_DURABILITY_INCREASED_LEVEL)
                    [agent.heal_durability(core_agent.CONSTANTS.DEFAULT_DURABILITY_INCREASED_LEVEL)
                     for agent in agents if agent.get_name() == best_agent.get_name()]

            # Best_agent information
            # exp.log("{}: Current best agent: {}, Disabilities:{}".format(t, best_agent.name,
            #                                                              [agent.durability() for agent in agents]))
            # print("{}: Current best agent: {}, Reward:{}".format(t, best_agent.name, best_agent.get_total_reward()))
                exp.log("{}: Current best agent: {}, Reward:{}".format(t, best_agent.get_name(),
                                                                       best_agent.get_total_reward()))

            # 4. Check the agent durability in specified step
            if t % core_agent.CONSTANTS.DURABILITY_CHECK_FREQUENCY == 0:
                if len(agents) > 1:
                    # index = [i for i in range(len(agents)) if i not in best_agents]
                    index = [i for i, agent in enumerate(agents) if agent.get_name() != best_agent.get_name()]
                    for i in index:
                        if agents[i].get_state() is not None:
                            agents[i].reduce_durability(core_agent.CONSTANTS.DEFAULT_DURABILITY_DECREASED_LEVEL)

            # 5. Kill agent
            if len(agents) > 1:
                for i, agent in enumerate(agents):
                    if agent.get_durability() <= 0:
                        del agents[i]

            # 6. Main step of core agent
            # core_agent_action = core_agent.select_core_action(best_agent_state, policy_net_flag, best_agent_action)
            if episode == 0 and t == 0:
                core_agent_action = core_agent.select_action(core_agent.get_state(), is_first=True)
            else:
                core_agent_action = core_agent.select_action(core_agent.get_state())
            core_agent.set_action(core_agent_action)

            core_obs, core_reward, core_done, core_info = core_agent.get_env().step(core_agent.get_action())
            core_agent.set_step_retrun_value(core_obs, core_done, core_info)

            core_agent.set_done_state(core_done)
            core_agent.set_total_reward(core_reward)

            if not core_done:
                core_next_state = utils.get_state(core_obs)
            else:
                core_next_state = None

            core_reward = torch.tensor([core_reward], device=core_agent.CONSTANTS.DEVICE)
            core_agent.memory.push(core_agent.get_state(), core_agent.get_action().to('cpu'), core_next_state,
                                   core_reward.to('cpu'))
            core_agent.set_state(core_next_state)

            if core_agent.steps_done > core_agent.CONSTANTS.INITIAL_MEMORY:
                core_agent.optimize_model()

                if core_agent.steps_done % core_agent.CONSTANTS.TARGET_UPDATE == 0:
                    core_agent.target_net.load_state_dict(core_agent.policy_net.state_dict())

            if core_agent.is_done():
                print("\n")
                break

            exp.log("{} steps | Current core_agent reward: {} | Episode:{}\n".format(t, core_agent.get_total_reward(),
                                                                                     episode))
            core_agent.writer.add_scalar("core/reward/all_step", core_agent.get_total_reward(), core_agent.steps_done)
            for agent in agents:
                agent.writer.add_scalar("internal/reward/{}/episode".format(agent.get_name()),
                                        agent.get_total_reward(), episode)
            # print("Current core_agent reward: {}".format(core_agent.get_total_reward()))

        # ----------------------
        # End of proposal method
        # ----------------------

        if episode % core_agent.CONSTANTS.MODEL_SAVING_FREQUENCY == 0:
            for agent in agents:
                with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-policy".format(agent.get_name()), 'wb') as f:
                    cloudpickle.dump(agent.policy_net, f)
                with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-target".format(agent.get_name()), 'wb') as f:
                    cloudpickle.dump(agent.target_net, f)
                agent.writer.add_scalar("internal/obtained_reward/{}".format(agent.get_name()),
                                        agent.get_obtained_reward(), episode)
            with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-policy".format(core_agent.get_name()), 'wb') as f:
                cloudpickle.dump(core_agent.target_net, f)
            with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-target".format(core_agent.get_name()), 'wb') as f:
                cloudpickle.dump(core_agent.target_net, f)

        t_reward = core_agent.get_total_reward()
        o_reward = core_agent.get_obtained_reward()
        exp.metric("total_reward", t_reward)
        exp.metric("steps", t)
        exp.metric("obtained_reward", o_reward)
        out_str = 'Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(
            core_agent.steps_done, episode, t, core_agent.get_total_reward())
        if episode % 20 == 0:
            print(out_str)
            out_str = str("\n" + out_str + "\n")
            exp.log(out_str)
        else:
            # print(out_str)
            exp.log(out_str)
        with open(core_agent.CONSTANTS.TRAIN_LOG_FILE_PATH, 'a') as f:
            f.write(str(out_str) + "\n")
        core_agent.writer.add_scalar("core/reward/total", core_agent.get_total_reward(), episode)
        core_agent.writer.add_scalar("core/steps/total", t, episode)
        core_agent.writer.add_scalars("telemetry", {"steps": t,
                                                    "reward": core_agent.get_total_reward()}, episode)
        core_agent.writer.add_scalar("core/obtained_reward/", core_agent.get_obtained_reward(), episode)
    core_env.close()
    core_agent.writer.close()
    for agent in agents:
        agent.writer.close()
    for agent in agents:
        agent.get_env().close()
    del best_agent
    # return best_agent


def test(env, n_episodes, policy, exp, exp_name, agent, render=True):
    for episode in range(n_episodes):
        obs = env.reset()
        state = utils.get_state(obs)
        total_reward = 0.0
        for _ in count():
            action = policy(state.to('cuda')).max(1)[1].view(1, 1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = utils.get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                out_str = "Finished Episode {} (test) with reward {}".format(episode, total_reward)
                exp.log(out_str)
                with open(agent.CONSTANTS.TEST_LOG_FILE_PATH, 'wt') as f:
                    f.write(out_str)
                break
    env.close()


def single_train(envs, agents, core_env, core_agent, n_episodes, agent_n, exp, exp_name, render=False,):
    """
    Training step for single-agent settings.

    Parameters
    ----------
    envs: list of Environment
        List of environment for multi-agent
    agents: list of Agent
        List of multi-agents to create candidates for core_agent
    core_env: Environment
        Main environment of this train step
    core_agent: Agent
        Main agent of this train step
    n_episodes: int
        The number of episodes
    agent_n : int
        The number of agent
    exp: Experiment
        The Experiment object used by hyperdash
    exp_name: str
        The name of experiment
    render: boolean, default False
        Flag for whether to render the environment
    """
    print("INFO: Single mode...")
    for episode in range(n_episodes):
        # 0. Initialize the environment, state and agent params
        obs = core_env.reset()
        core_state = utils.get_state(obs)
        core_agent.reset_total_reward()
        core_agent.set_state(core_state)

        for t in count():
            if episode == 0 and t == 0:
                core_agent_action = core_agent.select_action(core_agent.get_state(), is_first=True)
            else:
                core_agent_action = core_agent.select_action(core_agent.get_state())
            core_agent.set_action(core_agent_action)

            core_obs, core_reward, core_done, core_info = core_agent.get_env().step(core_agent.get_action())
            core_agent.set_step_retrun_value(core_obs, core_done, core_info)

            core_agent.set_done_state(core_done)
            core_agent.set_total_reward(core_reward)

            if not core_done:
                core_next_state = utils.get_state(core_obs)
            else:
                core_next_state = None

            core_reward = torch.tensor([core_reward], device=core_agent.CONSTANTS.DEVICE)
            core_agent.memory.push(core_agent.get_state(), core_agent.get_action().to('cpu'), core_next_state, core_reward.to('cpu'))
            core_agent.set_state(core_next_state)

            if core_agent.steps_done > core_agent.CONSTANTS.INITIAL_MEMORY:
                core_agent.optimize_model()

                if core_agent.steps_done % core_agent.CONSTANTS.TARGET_UPDATE == 0:
                    core_agent.target_net.load_state_dict(core_agent.policy_net.state_dict())

            if core_agent.is_done():
                print("\n")
                break

            exp.log("{}: Current core_agent reward: {} | Episode:{}\n".format(t, core_agent.get_total_reward(), episode))
            core_agent.writer.add_scalar("core/reward/all_step", core_agent.get_total_reward(), core_agent.steps_done)
            # print("Current core_agent reward: {}".format(core_agent.get_total_reward()))

        if episode % core_agent.CONSTANTS.MODEL_SAVING_FREQUENCY == 0:
            with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-policy".format(core_agent.get_name()), 'wb') as f:
                cloudpickle.dump(core_agent.target_net, f)
            with open(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/model_tmp/{}-target".format(core_agent.get_name()), 'wb') as f:
                cloudpickle.dump(core_agent.target_net, f)

        t_reward = core_agent.get_total_reward()
        o_reward = core_agent.get_obtained_reward()
        exp.metric("total_reward", t_reward)
        exp.metric("steps", t)
        exp.metric("obtained_reward", o_reward)
        out_str = 'Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(
            core_agent.steps_done, episode, t, core_agent.get_total_reward())
        if episode % 20 == 0:
            print(out_str)
            out_str = str("\n" + out_str + "\n")
            exp.log(out_str)
        else:
            exp.log(out_str)
        with open(core_agent.CONSTANTS.TRAIN_LOG_FILE_PATH, 'a') as f:
            f.write(str(out_str) + "\n")
        core_agent.writer.add_scalar("core/reward/total", core_agent.get_total_reward(), episode)
        core_agent.writer.add_scalar("core/steps/total", t, episode)
        core_agent.writer.add_scalars("telemetry", {"steps": t,
                                                    "reward": core_agent.get_total_reward()}, episode)
        core_agent.writer.add_scalar("core/obtained_reward/", core_agent.get_obtained_reward(), episode)
    core_env.close()
    core_agent.writer.close()
