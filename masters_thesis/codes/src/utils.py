import os
import random
import time
import json
from collections import namedtuple

import numpy as np
import torch
from PIL import Image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_observation(self, observation):
    if not self.enable_image:
        return observation
    img = Image.fromarray(observation)
    img = img.resize(self.shape).convert('L')  # resize and convert to grayscale
    return np.array(img) / 255


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


class Hyperparameter:
    def __init__(self, batch_size=32, gamma=0.99, eps_start=1, eps_end=0.02, eps_decay=1000000, target_update=1000,
                 default_durability=1000, learning_rate=1e-4, initial_memory=10000, n_episode=400,
                 default_durability_decreased_level=1,
                 default_durability_increased_level=1, default_check_frequency=80, default_healing_frequency=100,
                 env_name="PongNoFrameskip-v4", exp_name="PongNoFrameskip-v4", render=False,
                 run_name="videos_proposal", output_directory_path="./Runs",
                 hyper_dash=False, parameters_name="default"):
        # Runtime settings
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TRANSITION = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))
        # cv2.ocl.setUseOpenCL(False)
        time_stamp = str(int(time.time()))
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Hyper parameters
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.N_EPISODE = n_episode
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update
        self.DEFAULT_DURABILITY = default_durability
        self.LEARNING_RATE = learning_rate
        self.INITIAL_MEMORY = initial_memory
        self.MEMORY_SIZE = 10 * self.INITIAL_MEMORY
        self.DEFAULT_DURABILITY_DECREASED_LEVEL = default_durability_decreased_level
        self.DEFAULT_DURABILITY_INCREASED_LEVEL = default_durability_increased_level
        self.DURABILITY_CHECK_FREQUENCY = default_check_frequency
        self.DURABILITY_HEALING_FREQUENCY = default_healing_frequency

        # Some settings
        self.ENV_NAME = env_name
        self.EXP_NAME = exp_name + "_" + time_stamp
        self.RENDER = render
        self.HYPER_DASH = hyper_dash
        self.RUN_NAME = run_name
        self.OUTPUT_DIRECTORY_PATH = os.path.abspath(os.path.join(os.path.curdir,
                                                                  output_directory_path,
                                                                  self.ENV_NAME + "_" + self.RUN_NAME + "_" + time_stamp))

        self.TRAIN_LOG_FILE_PATH = self.OUTPUT_DIRECTORY_PATH + "/" + self.ENV_NAME + "_train_" + time_stamp + ".log"
        self.TEST_LOG_FILE_PATH = self.OUTPUT_DIRECTORY_PATH + "/" + self.ENV_NAME + "_test_" + time_stamp + ".log"
        self.PARAMETER_LOG_FILE_PATH = self.OUTPUT_DIRECTORY_PATH + "/" + self.ENV_NAME + "_params_" + time_stamp + ".json"
        self.PARAMETERS_NAME = parameters_name

        if not os.path.exists(self.OUTPUT_DIRECTORY_PATH):
            os.makedirs(self.OUTPUT_DIRECTORY_PATH)

        self.HYPER_PARAMS = {"BATCH_SIZE": self.BATCH_SIZE, "GAMMA": self.GAMMA, "EPS_START": self.EPS_START,
                             "EPS_END": self.EPS_END, "EPS_DECAY": self.EPS_DECAY,
                             "TARGET_UPDATE": self.TARGET_UPDATE,
                             "N_EPISODE": self.N_EPISODE,
                             "DEFAULT_DURABILITY": self.DEFAULT_DURABILITY,
                             "LEARNING_RATE": self.LEARNING_RATE,
                             "INITIAL_MEMORY": self.INITIAL_MEMORY, "MEMORY_SIZE": self.MEMORY_SIZE,
                             "DEFAULT_DURABILITY_DECREASED_LEVEL": self.DEFAULT_DURABILITY_DECREASED_LEVEL,
                             "DURABILITY_CHECK_FREQUENCY": self.DURABILITY_CHECK_FREQUENCY,
                             "DURABILITY_HEALING_FREQUENCY": self.DURABILITY_HEALING_FREQUENCY,
                             "ENV_NAME": self.ENV_NAME, "EXP_NAME": self.EXP_NAME,
                             "OUTPUT_DIRECTORY_PATH": self.OUTPUT_DIRECTORY_PATH,
                             "TRAIN_LOG_FILE_PATH": self.TRAIN_LOG_FILE_PATH,
                             "TEST_LOG_FILE_PATH": self.TEST_LOG_FILE_PATH,
                             "PARAMETER_LOG_FILE_PATH": self.PARAMETER_LOG_FILE_PATH,
                             "RENDER": str(self.RENDER),
                             "PARAMETERS_NAME": self.PARAMETERS_NAME}

        json_params = json.dumps(self.HYPER_PARAMS)
        with open(self.PARAMETER_LOG_FILE_PATH, 'wt') as f:
            f.write(json_params)
