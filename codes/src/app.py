from agent import Agent
from torch import optim
import sys
import models
import utils
import torch
from environment import Environment
from hyperdash import Experiment


def _create_agents(configs):
    try:
        agents = []
        for args in configs:
            hyper_parameters = utils.Hyperparameter()
            if args["model"] == "DQN":
                policy_net = models.DQN(n_actions=4).to()
                target_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
            elif args["model"] == "DDQN":
                policy_net = models.DDQN(n_actions=4).to()
                target_net = models.DDQN(n_actions=4).to(hyper_parameters.DEVICE)
            elif args["model"] == "NonBatchNormalizedDQN":
                policy_net = models.NonBatchNormalizedDQN(n_actions=4).to()
                target_net = models.NonBatchNormalizedDQN(n_actions=4).to(hyper_parameters.DEVICE)
            # elif args["model"] == "RamDQN":
            #     policy_net = models.RamDQN(n_actions=4).to()
            #     target_net = models.RamDQN(n_actions=4).to(hyper_parameters.DEVICE)
            else:
                policy_net = models.DQN(n_actions=4).to()
                target_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
            optimizer = optim.Adam(policy_net.parameters(), lr=hyper_parameters.LEARNING_RATE)
            agents.append(Agent(policy_net, target_net, hyper_parameters.DEFAULT_DURABILITY, optimizer, configs["name"],
                                hyper_parameters))
        return agents
    except KeyError:
        print("P_RuntimeError: Some arguments is missing.")
        sys.exit(1)


def create_agents():
    agents = []
    # TODO: Change to "Read from file" logic about CONSTANTS value.
    CONSTANTS0 = utils.Hyperparameter()

    # TODO: Change code to call _create_agents function
    policy_net_0 = models.DQN(n_actions=4).to()
    target_net_0 = models.DQN(n_actions=4).to(CONSTANTS0.DEVICE)
    optimizer_0 = optim.Adam(policy_net_0.parameters(), lr=CONSTANTS0.LEARNING_RATE)
    agents.append(Agent(policy_net_0, target_net_0, CONSTANTS0.DEFAULT_DURABILITY,
                        optimizer_0, "cnn-dqn0", CONSTANTS0))
    agents.append(Agent(policy_net_0, target_net_0, CONSTANTS0.DEFAULT_DURABILITY,
                        optimizer_0, "cnn-dqn1", CONSTANTS0))
    policy_net_1 = models.DDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    target_net_1 = models.DDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=CONSTANTS0.LEARNING_RATE)
    agents.append(Agent(policy_net_1, target_net_1, CONSTANTS0.DEFAULT_DURABILITY,
                        optimizer_1, "cnn-ddqn0", CONSTANTS0))
    agents.append(Agent(policy_net_1, target_net_1, CONSTANTS0.DEFAULT_DURABILITY,
                        optimizer_1, "cnn-ddqn1", CONSTANTS0))
    
    core_agent = Agent(policy_net_0, target_net_0, CONSTANTS0.DEFAULT_DURABILITY, optimizer_0, "core", CONSTANTS0)
    return agents, core_agent


def create_envs(agents, core_agent):
    envs = []
    for agent in agents:
        env = Environment(agent.CONSTANTS)
        env = env.get_env()
        envs.append(env)

    core_env = Environment(core_agent.CONSTANTS)
    core_env = core_env.get_env()

    for agent, env in zip(agents, envs):
        agent.set_env(env)
    core_agent.set_env(core_env)
    return envs, core_env


def hyper_dash_settings(exp_name):
    exp = Experiment(exp_name, capture_io=False)
    # print("Learning rate:{}".format(LEARNING_RATE))
    # exp.param("Learning rate", LEARNING_RATE)
    # exp.param("Environment", ENV_NAME)
    # exp.param("Batch size", BATCH_SIZE)
    # exp.param("Gamma", GAMMA)
    # exp.param("Episode start", EPS_START)
    # exp.param("Episode end", EPS_END)
    # exp.param("Episode decay", EPS_DECAY)
    # exp.param("Target update", TARGET_UPDATE)
    # exp.param("Render", str(RENDER))
    # exp.param("Initial memory", INITIAL_MEMORY)
    # exp.param("Memory size", MEMORY_SIZE)
    return exp


def main():
    # Main function flow
    # 0. Load experiment conditions
    exp = hyper_dash_settings("DUMMY")

    # 1. Create Agents
    agents, core_agent = create_agents()

    # 2. Create Environments
    envs, core_env = create_envs(agents, core_agent)

    # 3. Train model
    models.train(envs, agents, core_env, core_agent, 400, len(agents), exp)
    exp.end()
    torch.save(core_agent.policy_net, core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/dqn_pong_model")

    # 4. Test model
    # test_env = Environment()
    # test_env = env.get_env()
    #
    # policy_net = torch.load(output_directory + "/dqn_pong_model")
    # exp_test = Experiment(str(EXP_NAME + "_test_step"), capture_io=False)
    # test(test_env, 1, policy_net, exp_test, render=False)
    # exp_test.end()
    pass


if __name__ == "__main__":
    main()
