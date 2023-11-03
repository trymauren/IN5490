# Python modules
import numpy as np

# Unity ML agents modules
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.logging_util import set_log_level, DEBUG

set_log_level(DEBUG)


class UnityInterface():

    def __init__(self, executable_file: str = None, no_graphics: bool = False, worker_id : int = 0):
    
        self.env = self.start_env(executable_file=executable_file, no_graphics=no_graphics, worker_id=worker_id)
        self.behavior_names = list(self.env.behavior_specs.keys())
        self.num_agents = len(self.behavior_names)

    def start_env(self, executable_file: str = None, no_graphics: bool = False, worker_id: int = 0) -> UnityEnvironment:
        """Starting a unity environment. 

        Args:
            executable_file (str, optional): Name of the executable file. Defaults to None, for runing in editor mode
        Returns:
            UnityEnvironment: return the unity environment
        """
        env = UnityEnvironment(file_name=executable_file, no_graphics=no_graphics, worker_id=worker_id, log_folder='./')
        env.reset()
        return env

    def send_actions_to_unity(self, actions: np.array) -> list:
        
        self.env.reset()
        
        if len(actions) > self.num_agents:
            print(f"Need more agents, training with {self.num_agents} agents and {len(actions)} actions")
        positions = []
        # Looping over number of movements to evaluate

        #so the crwaler falls down to it start position 
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
