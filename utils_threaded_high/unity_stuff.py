from utils_threaded_high import functions
from config import ea_config, interface_config
# Unity ML agents modules
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.logging_util import set_log_level, DEBUG

set_log_level(DEBUG)


# Python modules
import numpy as np
np.random.seed(interface_config['seed'])


# Frank
import socket
import multiprocessing
import os

HIGHEST_WORKER_ID = 65535 - UnityEnvironment.BASE_ENVIRONMENT_PORT

env = None

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def is_worker_id_open(worker_id: int) -> bool:
    return not is_port_in_use(UnityEnvironment.BASE_ENVIRONMENT_PORT + worker_id)

def get_worker_id() -> int:
    pid = np.random.randint(HIGHEST_WORKER_ID)
    while not is_worker_id_open(pid):
        print("Socket is occupied, trying a new worker_id")
        pid = np.random.randint(HIGHEST_WORKER_ID)
    return pid

def get_env(no_graphics:bool=True, path_to_unity_exec:str=None):
    
    global env
    pid = multiprocessing.Process()._identity[0]

    if (env == None):
        print('FØØØØØØØØØØKK')
        worker_id = get_worker_id()
        #print(f"Opening Unity at '{path_to_unity_exec}', will try to use socket {worked_id} ({worked_id})")
        env = UnityEnvironment( 
                                seed=interface_config['seed'],
                                file_name=path_to_unity_exec,
                                no_graphics=no_graphics,
                                worker_id=worker_id
                                )
        env.reset()
    return env

def close_env():
    global env
    global channel
    if (env != None):
        env.close()
        env = None
        channel = None
# Frank end

def evaluate_in_unity(actions) -> list:
    root = os.path.abspath('')
    executable_path = os.path.join(root,'executables/')
    runs_path = os.path.join(root,'runs/')


    env = get_env(
                    no_graphics=interface_config['no_graphics'],
                    path_to_unity_exec=functions.get_executable(executable_path)
                    )

    env.reset()
    behavior_names = list(env.behavior_specs.keys())
    num_agents = len(behavior_names)

    if len(actions) > num_agents:
        print(f"Need more agents, training with {num_agents} agents and {len(actions)} actions")
    positions = []
    # Looping over number of movements to evaluate
    #so the crwaler falls down to it start position 
    for i in range(30):
        env.step()

    for i in range(len(actions[0][0])): # for 10 or 200

        # Looping over all actions (one for each agent) to evaluate
        # - one action is composed of many movements
        for action_i, individual in enumerate(actions): # for 30 agents
            
            # Shape the sequence of movements
            step = np.array([np.array([movement_dir[i] for movement_dir in individual])]) # for 12

            # Creates a datastructure that Unity understands
            action_tuple = ActionTuple()
            action_tuple.add_continuous(step)

            # Sets actions for all agents in a behavior name.
            env.set_actions(behavior_names[action_i], action_tuple) 

            if i >= len(actions[0][0]) - 1:
                decision_steps, _ = env.get_steps(behavior_names[action_i])
                positions.append(decision_steps.obs[0][0][:3])

        env.step()

    return positions

class UnityInterface():

    def __init__(   self,
                    executable_file: str = None,
                    no_graphics: bool = False,
                    worker_id : int = 0
                    ):
    
        self.env = self.start_env(  executable_file=executable_file,
                                    no_graphics=no_graphics,
                                    worker_id=worker_id
                                    )

        self.behavior_names = list(self.env.behavior_specs.keys())
        self.num_agents = len(self.behavior_names)

    def start_env(  self,
                    executable_file: str = None,
                    no_graphics: bool = False,
                    worker_id: int = 0
                    ) -> UnityEnvironment:
        """Starting a unity environment. 

        Args:
            Executable_file (str, optional): Name of the executable file.
        Returns:
            UnityEnvironment: return the unity environment
        """
        env = UnityEnvironment( seed=interface_config['seed'],
                                file_name=executable_file,
                                no_graphics=False,
                                worker_id=worker_id,
                                log_folder='./')
        env.reset()
        return env

    def send_actions_to_unity(self, actions: np.array) -> list:
        
        self.env.reset()
        
        if len(actions) > self.num_agents:
            print(f"Need more agents, training with {self.num_agents} \
                    agents and {len(actions)} actions")
        positions = []
        # Looping over number of movements to evaluate

        # so the crawler falls down to it start position 
        for i in range(30):
            self.env.step()
        for i in range(len(actions[0][0])): # for 10 or 200

            # Looping over all actions (one for each agent) to evaluate
            # - one action is composed of many movements
            for action_i, individual in enumerate(actions): # for 30 agents
                
                # Shape the sequence of movements
                step = np.array([np.array([movement_dir[i] for movement_dir in individual])]) # for 12

                # Creates a datastructure that Unity understands
                action_tuple = ActionTuple()
                action_tuple.add_continuous(step)

                # Sets actions for all agents in a behavior name.
                self.env.set_actions(self.behavior_names[action_i], action_tuple) 

                if i >= len(actions[0][0]) - 1:
                    decision_steps, _ = self.env.get_steps(self.behavior_names[action_i])
                    positions.append(decision_steps.obs[0][0][:3])

            self.env.step()

        return positions

    def stop_env(self) -> None:
        self.env.close() 

    def reset_env(self) -> None:
        self.env.reset()

    def get_agents(self) -> int:
        return self.num_agents
