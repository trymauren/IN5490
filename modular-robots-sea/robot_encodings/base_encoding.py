from robot_encodings.module_properties import module_properties
import numpy as np

class BaseEncoding:
    def __init__(self, Controller, max_modules, max_depth):
        self.controller_type = Controller
        self.genome = None
        self.controllers = None
        self.max_modules = max_modules
        self.max_depth = max_depth

    def mutate(self, mutation_probability, mutation_sigma, mutation_probability_controller, mutation_sigma_controller, rng):
        for i in range(self.max_modules):
            if rng.random() < mutation_probability_controller:
                self.controllers[i].mutate(mutation_sigma_controller)
            if rng.random() < mutation_probability:
                self.genome[i] = rng.choice(list(module_properties.keys()) + [0])
        self.parent_register = [None]
        for i, val in zip(range(len(self.genome)), self.genome):
            if val != 0:
                for _ in range(module_properties[val][0]):
                    self.parent_register.append(i)

    def init_random_robot(self, rng):
        self.genome = [rng.choice(list(module_properties.keys()) + [0]) for _ in range(self.max_modules)]
        self.controllers = [self.controller_type(rng) for _ in range(self.max_modules)]
        self.parent_register = [None]
        for i, val in zip(range(len(self.genome)), self.genome):
            if val != 0:
                for _ in range(module_properties[val][0]):
                    self.parent_register.append(i)

    def get_base_encoding(self):
        return self.genome

    def get_actions(self, time, observations=None):
        actions = np.ndarray(shape=(1,50),dtype=np.float32)
        i = 0
        for j in range(self.max_modules):
            key = self.genome[j]
            controller =  self.controllers[j]
            if key != 0 and module_properties[key][2]:
                parent_phase = 0.0
                if j != 0:
                    parent_key = self.genome[self.parent_register[j]]
                    if module_properties[parent_key][2]:
                        parent_phase = self.controllers[self.parent_register[j]].previous_phase
                observation = None
                if observations is not None:
                    observation = observations[i]
                actions[0][i] = controller.get_action(time, parent_phase_old=parent_phase, observation=observation)
                i += 1
        return actions
