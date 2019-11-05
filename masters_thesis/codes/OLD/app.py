import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import os
import time
import json

import gym
from collections import deque
from hyperdash import Experiment
import cv2

import base64

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# Runtime settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))
cv2.ocl.setUseOpenCL(False)
time_stamp = str(int(time.time()))
random.seed(0)
np.random.seed(0)

# Hyper parameters
BATCH_SIZE = 32 # @param
GAMMA = 0.99 # @param
EPS_START = 1 # @param
EPS_END = 0.02 # @param
EPS_DECAY = 1000000 # @param
TARGET_UPDATE = 1000 # @param
DEFAULT_DURABILITY = 10 # @param
LEARNING_RATE = 1e-4 # @param
INITIAL_MEMORY = 10000 # @param
MEMORY_SIZE = 10 * INITIAL_MEMORY # @param
DEFAULT_DURABILITY_DECREASED_LEVEL = 1 # @param
DURABILITY_CHECK_FREQUENCY = 40 # @param

# Some settings
ENV_NAME = "PongNoFrameskip-v4" # @param
EXP_NAME = "PongNoFrameskip-v4_" + time_stamp # @param
RENDER = False # @param

RUN_NAME = "videos_proposal" # @param
output_directory = os.path.abspath(
    os.path.join(os.path.curdir, "./Runs", ENV_NAME + "_" + RUN_NAME + "_" + time_stamp))

TRAIN_LOG_FILE_PATH = output_directory + "/" + ENV_NAME + "_train_" + time_stamp + ".log" # @param
TEST_LOG_FILE_PATH = output_directory + "/" + ENV_NAME + "_test_" + time_stamp + ".log" # @param
PARAMETER_LOG_FILE_PATH = output_directory + "/" + ENV_NAME + "_params_" + time_stamp + ".json" # @param

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

hyper_params = {"BATCH_SIZE": BATCH_SIZE, "GAMMA": GAMMA, "EPS_START": EPS_START,
                "EPS_END": EPS_END, "EPS_DECAY": EPS_DECAY,
                "TARGET_UPDATE": TARGET_UPDATE,
                "DEFAULT_DURABILITY": DEFAULT_DURABILITY,
                "LEARNING_RATE": LEARNING_RATE,
                "INITIAL_MEMORY": INITIAL_MEMORY, "MEMORY_SIZE": MEMORY_SIZE,
                "DEFAULT_DURABILITY_DECREASED_LEVEL": DEFAULT_DURABILITY_DECREASED_LEVEL,
                "DURABILITY_CHECK_FREQUENCY": DURABILITY_CHECK_FREQUENCY,
                "ENV_NAME" : ENV_NAME, "EXP_NAME": EXP_NAME, 
                "TRAIN_LOG_FILE_PATH": TRAIN_LOG_FILE_PATH,
                "TEST_LOG_FILE_PATH": TEST_LOG_FILE_PATH,
                "PARAMETER_LOG_FILE_PATH": PARAMETER_LOG_FILE_PATH,
                "RENDER": str(RENDER)}


# TODO : To change the deprecated function to Agent clsss fuction
def train(envs, agents, core_env, core_agent, n_episodes, agent_n, exp, render=False):
    """
    Training step.

    In this code, we use the multi-agents to create candidate for core agent.
    The core agent and environment is main RL set. In addition, each agent has
    own environment and durabiliry. Each agent's reward is checked for the
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
    render: boolean, default False
        Flag for whether to render the environment
    """
    for episode in range(n_episodes):
        obs = core_env.reset()
        states = []

        for agent, env in zip(agents, envs):
            obs = env.reset()
            state = get_state(obs)
            states.append(state)

        state = get_state(obs)
        core_state = state

        for t in count():
            if t % 20 != 0:
                print(str(t) + " ", end='')
            else:
                print("\n")
                print([agent.get_durability() for agent in agents])
                print(str(t) + " ", end='')

            for agent, env, i in zip(agents, envs, range(len(states))):
                envs[i] = core_env
                action = agent.select_action(state)
                agent.set_action(action)

            # if render:
            #     env.render()
            
            for agent, env, state_i in zip(agents, envs, range(len(states))):
                obs, reward, done, info = env.step(agent.action)
                agent.set_step_retrun_value(obs, reward, done, info)

                agent.total_reward += reward

                if not done:
                    next_state = get_state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=device)

                agent.memory.push(states[state_i], action.to('cpu'), next_state, reward.to('cpu'))
                states[state_i] = next_state

                if agent.steps_done > INITIAL_MEMORY:
                    agent.optimize_model()

                    if agent.steps_done % TARGET_UPDATE == 0:
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # ---------------
            # Proposal method
            # ---------------
            
            # Select best agent in this steps
            reward_list = [agent.total_reward for agent in agents]
            best_agents = [i for i, v in enumerate(reward_list) if v == max(reward_list)]
            best_agent_index = random.choice(best_agents)
            best_agent = agents[best_agent_index]
            
            
            # Best_agent infomation
            exp.log("Current best agent: {}".format(best_agent.name))

            # Check the agent durability in specified step.
            if t % DURABILITY_CHECK_FREQUENCY == 0:
                if len(agents) != 0:
                    index = [i for i in range(len(agents)) if i not in best_agents]
                    for i in index:
                        agents[i].reduce_durability(DEFAULT_DURABILITY_DECREASED_LEVEL)
                # Swap agents when agent durability reach to 0.
                # Also restore the agent durability to default durability.

                # ----
                # @TODO: Implements WORST_SWAP logic
                # For instance, if agent.durability <= 0, remove the agent from agents list at end of the episoad.
                # ----
                # for agent, i in zip(agents, range(len(agents)):
                #     if agent.durability <= 0:
                        # agent = copy.deepcopy(agents[best_agent_index])
                        # agent.durability = DEFAULT_DURABILITY
                        # envs[i] = copy.deepcopy(envs[best_agent_index])
            
                
            # core_agent step
            # core_agent_action = core_agent.select_action(core_state)
            core_agent_action = best_agent.get_action()
            core_agent.set_action(core_agent_action)

            # core_obs = best_agent.obs
            # core_reward = best_agent.reward
            # core_done = best_agent.done
            # core_info = best_agent.info
            core_obs, core_reward, core_done, core_info = core_env.step(
                core_agent.action)
            core_agent.set_step_retrun_value(core_obs, core_reward,
                                                core_done, core_info)
            core_agent.set_done_state(core_done)
            core_agent.total_reward += core_reward

            if not core_done:
                core_next_state = get_state(core_obs)
            else:
                core_next_state = None

            core_reward = torch.tensor([core_reward], device=device)

            core_agent.memory.push(core_state, action.to('cpu'),
                                    core_next_state, core_reward.to('cpu'))
            core_state = core_next_state

            if core_agent.steps_done > INITIAL_MEMORY:
                core_agent.optimize_model()

                if core_agent.steps_done % TARGET_UPDATE == 0:
                    core_agent.target_net.load_state_dict(core_agent.policy_net.state_dict())

            if core_agent.is_done():
                print("\n")
                break
        
        # Swap agent
        if len(agents) > 1 and episode % DURABILITY_CHECK_FREQUENCY == 0:
            for agent, env, i in zip(agents, envs, range(len(agents))):
                if agent.durability <= 0:
                    del agents[i]
                    del envs[i]
                    del states[i]

        # ----------------------
        # End of proposal method
        # ----------------------
        
        exp.metric("total_reword", core_agent.total_reward)
        out_str = 'Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(
            core_agent.steps_done, episode, t, core_agent.total_reward)
        if episode % 20 == 0:
            print(out_str)
            out_str = str("\n" + out_str + "\n")
            exp.log(out_str)
        else:
            # print(out_str)
            exp.log(out_str)
        with open(TRAIN_LOG_FILE_PATH, 'wt') as f:
            f.write(out_str)
    env.close()


# TODO : To change the deprecated function to Agent clsss fuction
def test(env, n_episodes, policy, exp, render=True):
    # Save video as mp4 on specified directory
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = env.get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = env.get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                out_str = "Finished Episode {} with reward {}".format(
                    episode, total_reward)
                print(out_str)
                exp.log(out_str)
                with open(TEST_LOG_FILE_NAME, 'wt') as f:
                    f.write(out_str)
                break

    env.close()


def main():
    # Create Agent
    agents = []

    policy_net_0 = DQN(n_actions=4).to(device)
    target_net_0 = DQN(n_actions=4).to(device)
    optimizer_0 = optim.Adam(policy_net_0.parameters(), lr=LEARNING_RATE)
    agents.append(Agent(policy_net_0, target_net_0, DEFAULT_DURABILITY,
                        optimizer_0, "cnn-dqn0"))
    agents.append(Agent(policy_net_0, target_net_0, DEFAULT_DURABILITY,
                        optimizer_0, "cnn-dqn1"))
    policy_net_1 = DQNbn(n_actions=4).to(device)
    target_net_1 = DQNbn(n_actions=4).to(device)
    optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=LEARNING_RATE)
    agents.append(Agent(policy_net_1, target_net_1, DEFAULT_DURABILITY,
                        optimizer_1, "cnn-bn-dqn0"))
    agents.append(Agent(policy_net_1, target_net_1, DEFAULT_DURABILITY,
                        optimizer_1, "cnn-bn-dqn1"))

    core_agent = Agent(policy_net_0, target_net_0, DEFAULT_DURABILITY, optimizer_0,
                    "core")

    AGENT_N = len(agents)

    # time_stamp = str(int(time.time()))
    hyper_params["AGENT_N"] = AGENT_N
    json_params = json.dumps(hyper_params)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(PARAMETER_LOG_FILE_PATH, 'wt') as f:
        f.write(json_params)
    
    # create environment
    # TODO: Create Environment class

    # env = gym.make(ENV_NAME)
    # env = make_env(env)
    envs = []
    for i in range(AGENT_N):
        env = Environment()
        env = env.get_env()
        envs.append(env)

    core_env = Environment()
    core_env = core_env.get_env()

    # setup optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # steps_done = 0

    # Deprecated
    # initialize replay memory
    # memory = ReplayMemory(MEMORY_SIZE)

    # Hyperdash experiment
    exp = Experiment(EXP_NAME, capture_io=False)
    print("Learning rate:{}".format(LEARNING_RATE))
    exp.param("Learning rate", LEARNING_RATE)
    exp.param("Environment", ENV_NAME)
    exp.param("Batch size", BATCH_SIZE)
    exp.param("Gamma", GAMMA)
    exp.param("Episode start", EPS_START)
    exp.param("Episode end", EPS_END)
    exp.param("Episode decay", EPS_DECAY)
    exp.param("Target update", TARGET_UPDATE)
    exp.param("Render", str(RENDER))
    exp.param("Initial memory", INITIAL_MEMORY)
    exp.param("Memory size", MEMORY_SIZE)

    # train model
    train(envs, agents, core_env, core_agent, 400, AGENT_N, exp)
    exp.end()
    torch.save(policy_net, output_directory + "/dqn_pong_model")

    # test model
    test_env = Environment()
    test_env = env.get_env()

    policy_net = torch.load(output_directory + "/dqn_pong_model")
    exp_test = Experiment(str(EXP_NAME + "_test_step"), capture_io=False)
    test(test_env, 1, policy_net, exp_test, render=False)
    exp_test.end()