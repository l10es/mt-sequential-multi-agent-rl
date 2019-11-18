import argparse
import csv
import sys

import torch
from hyperdash import Experiment
from torch import optim

import models
import utils
from agent import Agent
from environment import Environment


def _load_params(file_path):
    """
    Loading the hyperparameters from specific file.

    Parameters
    ----------
    file_path : str
        File path of config files

    Returns
    -------
        List of parameter dict

    """
    config_list = []
    exp_name = file_path.split("/")[-1].replace(".csv", "")
    with open(file_path, 'r+') as f:
        reader = csv.reader(f)
        for agent_config in reader:
            try:
                params_dict = {"name": agent_config[0],
                               "model": agent_config[1],
                               "batch_size": int(agent_config[2]),
                               "gamma": float(agent_config[3]),
                               "eps_start": int(agent_config[4]),
                               "eps_end": float(agent_config[5]),
                               "eps_decay": int(agent_config[6]),
                               "target_update": int(agent_config[7]),
                               "default_durability": int(agent_config[8]),
                               "learning_rate": float(agent_config[9]),
                               "initial_memory": int(agent_config[10]),
                               "n_episode": int(agent_config[11]),
                               "default_durability_decreased_level": int(agent_config[12]),
                               "default_durability_increased_level": int(agent_config[13]),
                               "default_check_frequency": int(agent_config[14]),
                               "default_healing_frequency": int(agent_config[15]),
                               "env_name": agent_config[16],
                               "exp_name": agent_config[17],
                               "render": bool(agent_config[18]),
                               "run_name": agent_config[19],
                               "output_directory_path": agent_config[20],
                               "hyper_dash": bool(agent_config[21]),
                               "model_saving_frequency": int(agent_config[22])}
                config_list.append(params_dict)
            except ValueError:
                pass
    return config_list, exp_name


def _create_agents(config_list):
    """
    Create agents with different hyper-parameters.

    Parameters
    ----------
    config_list : list of dict
        List of parameters dict. Each dict has configurations
        such as model name, learning rate, etc..

    Returns
    -------
        Created agents list and core agent object.

    """
    try:
        agents = []
        for config in config_list:
            hyper_parameters = utils.Hyperparameter(batch_size=config["batch_size"], gamma=config["gamma"],
                                                    eps_start=config["eps_start"], eps_end=config["eps_end"],
                                                    eps_decay=config["eps_decay"], target_update=config["target_update"],
                                                    default_durability=config["default_durability"],
                                                    learning_rate=config["learning_rate"],
                                                    initial_memory=config["initial_memory"],
                                                    n_episode=config["n_episode"],
                                                    default_durability_decreased_level=config["default_durability_decreased_level"],
                                                    default_durability_increased_level=config["default_durability_increased_level"],
                                                    default_check_frequency=config["default_check_frequency"],
                                                    default_healing_frequency=config["default_healing_frequency"],
                                                    env_name=config["env_name"], exp_name=config["exp_name"],
                                                    render=config["render"],
                                                    run_name=config["run_name"],
                                                    output_directory_path=config["output_directory_path"],
                                                    hyper_dash=config["hyper_dash"],
                                                    model_saving_frequency=["default_model_saving_frequency"],
                                                    parameters_name=config["name"])
            if config["name"] != "core":
                if config["model"] == "DQN":
                    policy_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
                    target_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
                elif config["model"] == "DDQN":
                    policy_net = models.DDQN(n_actions=4).to(hyper_parameters.DEVICE)
                    target_net = models.DDQN(n_actions=4).to(hyper_parameters.DEVICE)
                elif config["model"] == "DQNbn":
                    policy_net = models.DQNbn(n_actions=4).to(hyper_parameters.DEVICE)
                    target_net = models.DQNbn(n_actions=4).to(hyper_parameters.DEVICE)
                elif config["model"] == "NonBatchNormalizedDQN":
                    policy_net = models.NonBatchNormalizedDQN(n_actions=4).to()
                    target_net = models.NonBatchNormalizedDQN(n_actions=4).to(hyper_parameters.DEVICE)
                # elif args["model"] == "RamDQN":
                #     policy_net = models.RamDQN(n_actions=4).to(hyper_parameters.DEVICE)
                #     target_net = models.RamDQN(n_actions=4).to(hyper_parameters.DEVICE)
                else:
                    policy_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
                    target_net = models.DQN(n_actions=4).to(hyper_parameters.DEVICE)
                optimizer = optim.Adam(policy_net.parameters(), lr=hyper_parameters.LEARNING_RATE)
                agents.append(Agent(policy_net, target_net, hyper_parameters.DEFAULT_DURABILITY, optimizer,
                                    config["name"], hyper_parameters))
            else:
                # For core agent
                policy_net = models.NonBatchNormalizedDQN(n_actions=4).to(hyper_parameters.DEVICE)
                target_net = models.NonBatchNormalizedDQN(n_actions=4).to(hyper_parameters.DEVICE)
                optimizer = optim.Adam(policy_net.parameters(), lr=hyper_parameters.LEARNING_RATE)
                core_agent = Agent(policy_net, target_net, hyper_parameters.DEFAULT_DURABILITY, optimizer,
                                   config["name"], hyper_parameters)
            print("Agent:{} has been done".format(config["name"]))
        try:
            core_agent
        except Exception as e:
            print("P_RuntimeError:0x1000 Core agent has not been defined.")
            tb = sys.exc_info()[2]
            print(e.with_traceback(tb))
            sys.exit(1)
        return agents, core_agent
    except Exception as e:
        print("P_RuntimeError:0x1001 Some arguments is missing.")
        tb = sys.exc_info()[2]
        print(e.with_traceback(tb))
        sys.exit(1)


def create_agents(config_list):
    # agents = []
    # # TODO: Change to "Read from file" logic about CONSTANTS value.
    agents, core_agent = _create_agents(config_list)
    # CONSTANTS0 = utils.Hyperparameter()
    # print(CONSTANTS0.DEVICE)
    #
    # # TODO: Change code to call _create_agents function
    # policy_net_0 = models.NonBatchNormalizedDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    # target_net_0 = models.NonBatchNormalizedDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    # optimizer_0 = optim.Adam(policy_net_0.parameters(), lr=CONSTANTS0.LEARNING_RATE)
    # agents.append(Agent(policy_net_0, target_net_0, CONSTANTS0.DEFAULT_DURABILITY,
    #                     optimizer_0, "cnn-dqn0", CONSTANTS0))
    # agents.append(Agent(policy_net_0, target_net_0, CONSTANTS0.DEFAULT_DURABILITY,
    #                     optimizer_0, "cnn-dqn1", CONSTANTS0))
    # policy_net_1 = models.DDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    # target_net_1 = models.DDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    # optimizer_1 = optim.Adam(policy_net_1.parameters(), lr=CONSTANTS0.LEARNING_RATE)
    # agents.append(Agent(policy_net_1, target_net_1, CONSTANTS0.DEFAULT_DURABILITY,
    #                     optimizer_1, "cnn-ddqn0", CONSTANTS0))
    # agents.append(Agent(policy_net_1, target_net_1, CONSTANTS0.DEFAULT_DURABILITY,
    #                     optimizer_1, "cnn-ddqn1", CONSTANTS0))
    #
    # core_policy_net = models.NonBatchNormalizedDQN(n_actions=4).to()
    # core_target_net = models.NonBatchNormalizedDQN(n_actions=4).to(CONSTANTS0.DEVICE)
    # core_agent = Agent(core_policy_net, core_target_net, CONSTANTS0.DEFAULT_DURABILITY,
    #                    optimizer_0, "core", CONSTANTS0)
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


def create_test_envs(agent):
    test_env = Environment(agent.CONSTANTS)
    test_env = test_env.get_env()
    return test_env


def hyper_dash_settings(exp_name):
    exp = Experiment(exp_name, capture_io=False)
    return exp


def main(file_path):
    # Main function flow
    # 0. Load experiment conditions
    config_list, exp_name = _load_params(file_path)
    exp = hyper_dash_settings(exp_name)

    # 1. Create Agents
    agents, core_agent = create_agents(config_list)

    # 2. Create Environments
    envs, core_env = create_envs(agents, core_agent)

    # 3. Train model
    # best_agent = models.train(envs, agents, core_env, core_agent, core_agent.CONSTANTS.N_EPISODE, len(agents), exp)
    models.train(envs, agents, core_env, core_agent, core_agent.CONSTANTS.N_EPISODE, len(agents), exp)
    exp.end()
    # torch.save(best_agent.policy_net, best_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/dqn_pong_model")
    for agent in agents:
        torch.save(agent.policy_net,
                   agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/{}/internal-agent/{}".format(exp_name, agent.get_name()))
    torch.save(core_agent.policy_net,
               core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/{}/core-agent/{}".format(exp_name, core_agent.get_name()))

    # 4. Test model
    test_env = create_test_envs(core_agent)

    policy_net = torch.load(core_agent.CONSTANTS.OUTPUT_DIRECTORY_PATH + "/{}/core-agent/{}".format(exp_name, core_agent.get_name()))
    exp_test = hyper_dash_settings(exp_name + "_test")
    models.test(test_env, 1, policy_net, exp_test, exp_name, render=False, agent=core_agent)
    exp_test.end()
    # pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This program using for \"Multi-agent reinforcement learning with different parameter configurations using agent durability\"")
    parser.add_argument("-fp", "--file_path", type=str,
                        default="./configs/exp-dummy.csv",
                        help="File path of agents config for experiment")
    args = parser.parse_args()

    main(args.file_path)
