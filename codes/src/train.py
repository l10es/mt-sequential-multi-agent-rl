import random
from itertools import count

import torch

import utils


def train(envs, agents, core_env, core_agent, n_episodes, agent_n, exp, constants, render=False):
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
    constants: Constants
        The hyper-parameters
    render: boolean, default False
        Flag for whether to render the environment
    """
    for episode in range(n_episodes):
        # 0. Initialize the environment, state and agent params
        obs = core_env.reset()
        core_state = utils.get_state(obs)
        core_agent.total_reward = 0.0
        core_agent.set_state(core_state)
        for agent in agents:
            obs = agent.get_env().reset()
            state = utils.get_state(obs)
            agent.set_state(state)
            agent.total_reward = 0.0
            # agent.durability = DEFAULT_DURABILITY

        for t in count():
            # if t % 20 != 0:
            #     print(str(t) + " ", end='')
            # else:
            #     print("\n")
            #     print([agent.get_durability() for agent in agents])
            #     print(str(t) + " ", end='')

            # 1. Select action from environment of each agent
            for agent in agents:
                agent.set_env(core_agent.get_env())
                action = agent.select_action(agent.get_state())
                agent.set_action(action)

            # 2. Proceed step of each agent
            for agent in agents:
                obs, reward, done, info = agent.get_env().step(agent.get_action())
                agent.set_step_retrun_value(obs, reward, done, info)

                agent.set_total_reward(reward)

                if not done:
                    next_state = utils.get_state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=constants.DEVICE)

                agent.memory.push(agent.get_state(), agent.get_action().to('cpu'),
                                  next_state, reward.to('cpu'))
                agent.set_state(next_state)

                if agent.steps_done > constants.INITIAL_MEMORY:
                    agent.optimize_model()

                    if agent.steps_done % constants.TARGET_UPDATE == 0:
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # ---------------
            # Proposal method
            # ---------------

            # 3. Select best agent in this step
            reward_list = [agent.get_total_reward() for agent in agents]
            best_agents = [i for i, v in enumerate(reward_list) if v == max(reward_list)]
            best_agent_index = random.choice(best_agents)
            best_agent = agents[best_agent_index]
            if t % constants.DURABILITY_HEALING_FREQUENCY == 0:
                best_agent.heal_durability(constants.DEFAULT_DURABILITY_INCREASED_LEVEL)

            # Best_agent information
            exp.log("Current best agent: {}".format(best_agent.name))

            # 4. Check the agent durability in specified step
            if t % constants.DURABILITY_CHECK_FREQUENCY == 0:
                if len(agents) > 1:
                    index = [i for i in range(len(agents)) if i not in best_agents]
                    for i in index:
                        agents[i].reduce_durability(constants.DEFAULT_DURABILITY_DECREASED_LEVEL)

            # 5. Main step of core agent
            core_agent_action = best_agent.get_action()
            core_agent.set_action(core_agent_action)

            core_obs, core_reward, core_done, core_info = core_agent.get_env().step(
                core_agent.get_action())
            core_agent.set_step_retrun_value(core_obs, core_reward, core_done, core_info)

            core_agent.set_done_state(core_done)
            core_agent.set_total_reward(core_reward)

            if not core_done:
                core_next_state = utils.get_state(core_obs)
            else:
                core_next_state = None

            core_reward = torch.tensor([core_reward], device=constants.DEVICE)

            core_agent.memory.push(core_agent.get_state(),
                                   core_agent.get_action().to('cpu'),
                                   core_next_state, core_reward.to('cpu'))
            core_agent.set_state(core_next_state)

            if core_agent.steps_done > constants.INITIAL_MEMORY:
                core_agent.optimize_model()

                if core_agent.steps_done % constants.TARGET_UPDATE == 0:
                    core_agent.target_net.load_state_dict(core_agent.policy_net.state_dict())

            if core_agent.is_done():
                print("\n")
                break

        # 6. Swap agent
        if len(agents) > 1 and episode % constants.DURABILITY_CHECK_FREQUENCY == 0:
            for agent, i in zip(agents, range(len(agents))):
                if agent.durability <= 0:
                    del agents[i]

        # ----------------------
        # End of proposal method
        # ----------------------

        exp.metric("total_reward", core_agent.get_total_reward())
        out_str = 'Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(
            core_agent.steps_done, episode, t, core_agent.total_reward)
        if episode % 20 == 0:
            print(out_str)
            out_str = str("\n" + out_str + "\n")
            exp.log(out_str)
        else:
            # print(out_str)
            exp.log(out_str)
        with open(constants.TRAIN_LOG_FILE_PATH, 'wt') as f:
            f.write(out_str)
    core_env.close()
