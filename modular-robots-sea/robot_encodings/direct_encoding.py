from robot_encodings.module_properties import module_properties
import numpy as np
from copy import deepcopy

class DirectEncoding:
    def __init__(self, Controller, max_modules, max_depth, num_symbols, self_adaptive, module_types):
        self.controller_type = Controller
        self.root = None
        self.max_modules = max_modules
        self.max_depth = max_depth
        self.module_types = module_types

    def mutate(self, mutation_probability, mutation_sigma, mutation_probability_controller, mutation_sigma_controller, rng):
        mutation_probability = mutation_probability/len(self.root.get_subtree_nodes())
        nodes = self.root.get_subtree_nodes()
        num_controllers = 0
        for node in nodes:
            if node.controller is not None:
                num_controllers += 1
        if num_controllers != 0:
            mutation_probability_controller = mutation_probability_controller/num_controllers
        for node in nodes:
            for i in range(len(node.children)):
                if node.controller is not None:
                    node.controller.mutate(mutation_sigma_controller, mutation_probability_controller, rng)
                if rng.random() < mutation_probability:
                    node.rotation = rng.choice([0, 90, 180, 270])
                if node.children[i] is None and len(self.root.get_subtree_nodes()) < self.max_modules and node.depth < self.max_depth and rng.random() < mutation_probability:
                    node.add(rng.choice(self.module_types), i, self.controller_type(rng), rng.choice([0, 90, 180, 270]))
                    nodes.append(node.children[i])
                if node.children[i] is None and rng.random() < mutation_probability/2.0:
                    j = rng.choice(range(len(node.children)))
                    if node.children[j] is not None:
                        if len(node.children[j].get_subtree_nodes()) < self.max_modules - len(self.root.get_subtree_nodes()):
                            node.children[i] = deepcopy(node.children[j])
                            for subtree_node in node.children[i].get_subtree_nodes():
                                nodes.append(subtree_node)
                if node.children[i] is not None and rng.random() < mutation_probability/2.0:
                    node.remove(i)
        return True

    def init_random_robot(self, rng):
        self.root = Node(rng.choice(self.module_types), 0, self.controller_type(rng), None, rng.choice([0, 90, 180, 270]))
        for _ in range(40):
            self.mutate(0.2, 0.3, 0.0, 0.3, rng)

    def get_base_encoding(self):
        base_encoding = []
        springyness_encoding = []
        rotation_encoding = []
        base_encoding.append(self.root.module_type)
        rotation_encoding.append(self.root.rotation)
        if self.root.controller is not None:
            springyness_encoding.append(self.root.controller.springyness)
        else:
            springyness_encoding.append(0)
        nodes = self.root.get_subtree_nodes_sorted()
        for node in nodes:
            for child in node.children:
                if child is None:
                    base_encoding.append(0)
                    springyness_encoding.append(0)
                    rotation_encoding.append(0)
                else:
                    base_encoding.append(child.module_type)
                    rotation_encoding.append(child.rotation)
                    if child.controller is not None:
                        springyness_encoding.append(child.controller.springyness)
                    else:
                        springyness_encoding.append(0)
        return base_encoding, springyness_encoding, rotation_encoding
    
    def get_realised_nodes(self, realised_genome):
        genomeified_nodes = []
        genomeified_nodes.append(self.root)
        nodes = self.root.get_subtree_nodes_sorted()
        for node in nodes:
            for child in node.children:
                genomeified_nodes.append(child)
        realised_nodes = []
        for node, val in zip(genomeified_nodes, realised_genome):
            if val != 0:
                realised_nodes.append(node)
        return realised_nodes

    def get_actions(self, time, observations=None):
        actions = np.ndarray(shape=(1,50),dtype=np.float32)
        nodes = self.root.get_subtree_nodes_sorted()
        i = 0
        for node in nodes:
            if node.controller is not None:
                parent_phase = 0.0
                if node.parent is not None and node.parent.controller is not None:
                    parent_phase = node.parent.controller.previous_phase
                observation = None
                if observations is not None:
                    observation = observations[i]
                actions[0][i] = node.controller.get_action(time, parent_phase_old=parent_phase, observation=observation)
                i += 1
        return actions

class Node:
    def __init__(self, module_type, depth, controller, parent, rotation):
        self.module_type = module_type
        self.controller = None
        if module_properties[module_type][2]:
            self.controller = controller
        self.max_children = module_properties[module_type][0]
        self.children = [None for _ in range(self.max_children)]
        self.rotation = rotation
        self.depth = depth
        self.parent = parent

    def add(self, module_type, index, controller, rotation):
        self.children[index] = Node(module_type, self.depth+1, controller, self, rotation)

    def remove(self, index):
        self.children[index] = None

    def get_subtree_nodes(self):
        nodes = []
        nodes.extend([self])
        for child in self.children:
            if child is not None:
                nodes.extend(child.get_subtree_nodes())
        return nodes

    def get_subtree_nodes_sorted(self):
        nodes = self.get_subtree_nodes()
        sorted_nodes = []
        changed = True
        depth = 0
        while(changed):
            changed = False
            for node in nodes:
                if node.depth == depth:
                    changed = True
                    sorted_nodes.append(node)
            depth += 1
        return sorted_nodes
