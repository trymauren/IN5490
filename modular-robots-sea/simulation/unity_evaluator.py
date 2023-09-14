from simulation.unity_interface import RobotConfigChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import time
import os

BUILD_PATH="EmmasModularRobots/Builds/TestBuild.x86_64"

class UnityEvaluator:
    def __init__(self, evaluation_steps, editor_mode=False, headless=False, worker_id=0):
        self.steps = evaluation_steps
        self.channel = RobotConfigChannel()
        if editor_mode:
            self.env = UnityEnvironment(file_name=None, seed=1, side_channels=[self.channel])
        elif headless:
            self.env = UnityEnvironment(file_name=BUILD_PATH, seed=1, side_channels=[self.channel], no_graphics=True, worker_id=worker_id)
        else:
            self.env = UnityEnvironment(file_name=BUILD_PATH, seed=1, side_channels=[self.channel], no_graphics=False, worker_id=worker_id)
        self.times_used = 0

    def evaluate(self, individual):
        # Setup
        genome_length, genome, springyness, rotation = individual.get_unity_data()
        self.channel.send_config(genome_length, genome, springyness, rotation)
        self.env.reset()
        sleep_counter = 0
        while(self.channel.message is None):
            print("Could not connect to unity.")
            os._exit(1)
        realised_genome = self.channel.message
        behavior_names = self.env.behavior_specs.keys()
        behavior_name = list(behavior_names)[0]

        # For data collection
        energy_consumption = None

        # Simulation
        for g in range(self.steps):
            decisionSteps, terminalSteps = self.env.get_steps(behavior_name)
            if len(list(decisionSteps)) == 0:
                break
            if len(decisionSteps.obs) > 0:
                observations = list(decisionSteps.obs[0][0][::2])
                powers = list(decisionSteps.obs[0][0][1::2])
                if energy_consumption is None:
                    energy_consumption = [0.0 for _ in range(len(powers))]
                energy_consumption = [x + abs(y) for x, y in zip(energy_consumption, powers)]
                actions = individual.get_actions(g, observations=observations)
            else:
                actions = individual.get_actions(g)
            decisionSteps[self.times_used].reward
            self.env.set_actions(behavior_name, ActionTuple(actions))
            self.env.step()

        

        # Number of modules
        a = 0
        for val in realised_genome:
            if val != 0:
                a += 1
        # Number of springs
        b = 0
        spring_b = 0
        for val, spr in zip(realised_genome, springyness):
            if val == 3 or val == 4:
                b += 1
                spring_b += spr #*spr*spr*spr
        # Module types used
        modules = []
        c = 0
        for val in realised_genome:
            if val != 0 and val not in modules:
                modules.append(val)
        
        # Behaviour descriptors
        speed = decisionSteps[self.times_used].reward
        if speed > 25.0:
            speed = 25.0
        behaviour_descriptors = [a/(individual.encoding.max_modules+0.00001), spring_b/(b+0.00001)] #, speed/25.00001]

        self.times_used += 1
        return decisionSteps[self.times_used-1].reward, behaviour_descriptors, realised_genome, energy_consumption