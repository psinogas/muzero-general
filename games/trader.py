import datetime
import os

import numpy
import torch
import pandas as pd

import random
import csv

from matplotlib import pyplot as plt

from .abstract_game import AbstractGame

# TODO: reward on time : the longer it takes more expensive (cost of capital/ oportunity)

sintetic: bool = False

data: list = [[],[],[],[]]

# 1 asset, raw history, long short trader, max 1 trade (open+close)
history_size = 1 # 10 secs 5 mins
sin_data: list = [
    0.0,  0.2079,  0.4067,  0.5878,  0.7431,  0.866,  0.9511,  0.9945,  0.9945,  0.9511,  0.866,  0.7431,  0.5878,  0.4067,  0.2079,
    0.0, -0.2079, -0.4067, -0.5878, -0.7431, -0.866, -0.9511, -0.9945, -0.9945, -0.9511, -0.866, -0.7431, -0.5878, -0.4067, -0.2079,
    0.0,  0.2079,  0.4067,  0.5878,  0.7431,  0.866,  0.9511,  0.9945,  0.9945,  0.9511,  0.866,  0.7431,  0.5878,  0.4067,  0.2079,
    0.0, -0.2079, -0.4067, -0.5878, -0.7431, -0.866, -0.9511, -0.9945, -0.9945, -0.9511, -0.866, -0.7431, -0.5878, -0.4067, -0.2079,
    0.0,  0.2079,  0.4067,  0.5878,  0.7431,  0.866,  0.9511,  0.9945,  0.9945,  0.9511,  0.866,  0.7431,  0.5878,  0.4067,  0.2079,
    0.0, -0.2079, -0.4067, -0.5878, -0.7431, -0.866, -0.9511, -0.9945, -0.9945, -0.9511, -0.866, -0.7431, -0.5878, -0.4067, -0.2079,
    0.0,  0.2079,  0.4067,  0.5878,  0.7431,  0.866,  0.9511,  0.9945,  0.9945,  0.9511
    ]

RAW_VALUE = 0
DELTA_VALUE = 1
SMA_20 = 2
EMA_20 = 3

HISTORY_SIZE = 60 # seconds, ou 60*60

KML_UPPER_1 = 4
KML_UPPER_2 = 5
KML_UPPER_3 = 6
KML_LOWER_1 = 7
KML_LOWER_2 = 8
KML_LOWER_3 = 9

class MuZeroConfig:

    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. By default muzero uses every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (1, 2, history_size)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = [0, 1, 2] # Hold Buy Sell
        # List of players. You should only edit the length
        self.players = [0]
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = HISTORY_SIZE

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = len(sin_data)*10  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.978  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = []
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = []
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = []

        # Fully Connected Network
        self.encoding_size = 5
        # Define the hidden layers in the representation network
        self.fc_representation_layers = [16]
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [16]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [16]
        # Define the hidden layers in the value network
        self.fc_value_layers = [16]
        # Define the hidden layers in the policy network
        self.fc_policy_layers = [16]

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 30000
        self.batch_size = 32  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 10
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 1
        self.train_on_gpu = True if torch.cuda.is_available() else False  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0064  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        # Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 5000
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 7
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.2  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):

    def __init__(self, seed=None):
        self.env = TradingEnv()
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        observation, reward, done = self.env.step(action)
        # return [[observation]], reward*10, done
        return [observation], reward, done

    def legal_actions(self):
        return self.env.legal_actions()
        # return [0, 1, 2] # Hold Buy Sell

    def reset(self):
        return [self.env.reset()]

    def render(self):
        self.env.render()
        # input("Press enter to take a step ")

    def action_to_string(self, action_number):
        actions = {
            0: "Hold",
            1: "Buy",
            2: "Sell"
        }
        return f"{action_number}. {actions[action_number]}"

class TradingEnv:

    def fetch_raw(self) -> list:
        print("FETCH")
        out_list: list
        if sintetic:
            out_list = [10+x for x in sin_data]
        else:
            out_list = []
            with open('/Users/sinogas/Dev/ALSIMind/ALSIMind-algos/output/bot/alsibot.20200715.LINKUSDT.csv', 'rU') as infile:
                reader = csv.DictReader(infile, fieldnames=["ts","symbol","price","volume"])
                for row in reader:
                    out_list.append(float(row["price"]))
        print(f"read {len(out_list)} records.")
        return out_list

    def calc_delta(self, in_list: list) -> list:
        print("CALC DELTA")
        out_list: list = []
        previous = None
        for x in in_list:
            if previous is not None:
                out_list.append((x-previous)/previous)
            previous = x
        out_list.append(0)
        return out_list

    def calc_sma(self, in_list: list, size: int = 20) -> list:
        print("CALC SMA")
        dataFrame = pd.DataFrame(in_list)
        out_list = dataFrame.rolling(window=size).mean()
        out_list = out_list.values.tolist()
        return out_list

    def calc_ema(self, in_list: list, size: int = 20) -> list:
        print("CALC EMA")
        dataFrame = pd.DataFrame(in_list)
        out_list = dataFrame.rolling(window=size).mean()
        out_list = out_list.values.tolist()
        return out_list

    def __init__(self, size=history_size):

        if len(data[RAW_VALUE]) == 0:
            data[RAW_VALUE] = self.fetch_raw()
            data[DELTA_VALUE] = self.calc_delta(data[RAW_VALUE])
            data[SMA_20] = self.calc_sma(data[DELTA_VALUE], 20)
            data[EMA_20] = self.calc_ema(data[DELTA_VALUE], 20)

        # All calculation in price value, then converted to % to the current price == (calc-price)/price or calc/price

        # price delta delta

        # X periodos : SMA / EMA ??? (% ao valor actual)

        # X periodos : BB% (Volatilidade)

        # X key market levels: idealmente em relacao de % face ao vamos i.e. distancia ao valor % ao key market level (+ para cima (resistance) - para baixo(suporte) )
        #  last X zeros that difer more than X% (to be considered as KML suppport)

        # outros indicadores??? como descobrir indicadores que expliquem o q esta a acontecer.... estrategias de outros...

        self.window_size = size
        self.open_price = 0
        self.open_ts = -1
        self.random = random.Random()

    def seed(self, value):
        self.random.seed(value)

    def legal_actions(self):
        legal_actions =  [0, 1, 2] # Hold, Buy, Sell
        if self.open_price > 0:
            legal_actions.remove(1) # Long open, Remove Buy
        if self.open_price < 0:
            legal_actions.remove(2) # Sort open, Remove Sell
        return legal_actions

    def step(self, action):
        reward = 0
        finished = False
        price = data[RAW_VALUE][self.ts]

        if action in self.legal_actions():
            self.ts += 1
            if self.ts==len(data[RAW_VALUE]):
                finished = True

        if action not in self.legal_actions():
            pass

        elif action == 0: # Hold
            if finished: # finished -> force close position
                if self.open_price > 0:
                    action = 2
                if self.open_price < 0:
                    action = 1

        elif action == 1: # Buy
            if self.open_ts == -1:
                self.open_price = price # Open Long
                self.open_ts = self.ts-1
            else:
                finished = True # Close Short

        elif action == 2: # Sell
            if self.open_ts == -1:
                self.open_price = -price # Open Short
                self.open_ts = self.ts-1
            else:
                finished = True # Close Long

        trade_type = "OPEN"

        if self.open_price > 0:
            reward = (price - self.open_price) / self.open_price # Long P&L
            trade_type = "LONG"

        if self.open_price < 0:
            reward = (self.open_price + price) / self.open_price # Short P&L
            trade_type = "SHORT"

        if reward != 0:
            reward -= 2*float(0.075)/100 # 2 x exchange fee

        if finished:
            if reward<=0:
                reward_eval="---"
            if reward>0:
                reward_eval="+++"
            print(f"{reward_eval} {self.ts-1:5} - {trade_type:5} ({self.open_ts-self.start_ts:3}+{self.ts-self.open_ts-1:3}s): open@{abs(self.open_price):7.4f} close@{price:8.4f} P&L:{100*reward:5.2f}%   ")
        else:
            reward = 0

        # reward can be the unrealized profit (now is the realized profit or zero)
        return self.get_observation(), reward, finished

    def reset(self):
        self.ts = self.random.randint(self.window_size,len(data[RAW_VALUE])-1)
        self.start_ts = self.ts
        self.open_price = 0
        self.open_ts = -1
        return self.get_observation()

    def render(self):
        im = [
            data[DELTA_VALUE][self.ts-self.window_size:self.ts],
            data[RAW_VALUE][self.ts-self.window_size:self.ts]
        ]
        # im.append(self.open_price)
        # plt.plot(im)
        # plt.show()
        # print(self.ts,": ",im)

    def get_observation(self):
        observation = [
            data[DELTA_VALUE][self.ts-self.window_size:self.ts],
            [0]*len(data[DELTA_VALUE][self.ts-self.window_size:self.ts])
            # data[RAW_VALUE][self.ts-self.window_size:self.ts]
        ]
        # observation.append(self.open_price)
        return observation
