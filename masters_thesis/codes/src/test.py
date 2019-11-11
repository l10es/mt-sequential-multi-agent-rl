import time
from itertools import count

import gym


def test(env, n_episodes, policy, exp, constants, render=True):
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
                with open(constants.TEST_LOG_FILE_NAME, 'wt') as f:
                    f.write(out_str)
                break

    env.close()
