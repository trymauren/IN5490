import argparse
import numpy as np

from evolution.ea import StandardEvolutionaryAlgorithm
from evolution.map_elites import MapElites

from robot_encodings.individual import Individual
from robot_encodings.base_encoding import BaseEncoding
from robot_encodings.direct_encoding import DirectEncoding

from controllers.sine_controller import SineController

parser = argparse.ArgumentParser()
group_general = parser.add_argument_group("General parameters")
group_general.add_argument('-t', '--threads', type=int, default=16)
group_general.add_argument('-so', '--socket_offset', type=int, default=0)
group_general.add_argument('-s', '--seed', type=int, default=None)
group_general.add_argument('-l', '--load', action='store_true')
group_general.add_argument('-pa', '--path', default='')

group_simulator = parser.add_argument_group("Simulator parameters")
group_simulator_ex = group_simulator.add_mutually_exclusive_group()
group_simulator_ex.add_argument('-hl', '--headless', action='store_true')
group_simulator_ex.add_argument('-em', '--editor_mode', action='store_true')
group_simulator.add_argument('-n', '--evaluation_steps', type=int, default=5000)

group_evolution = parser.add_argument_group("Evolution parameters")
group_evolution.add_argument('-a', '--algorithm', choices=['ea', 'map', 'map_elites'], default='ea')
group_evolution.add_argument('-f', '--fitness', choices=['d', 'distance', 'cot', 'costoftransport'], default='distance')
group_evolution.add_argument('-p', '--population_size', type=int, default=64)
group_evolution.add_argument('-k', '--tournament_size', type=int, default=4)
group_evolution.add_argument('-y', '--symbol_size', type=int, default=16)
group_evolution.add_argument('-g', '--generations', type=int, default=128)
group_evolution.add_argument('-cr', '--crossover_probability', type=float, default=0.0)
group_evolution.add_argument('-sa', '--self_adaptive', action='store_true')
group_evolution.add_argument('-mp', '--mutation_parameters', nargs=4, type=float, default=[0.32, 0.3, 0.64, 0.3])
group_evolution.add_argument('-dim', '--map_dimensions', type=int, default=2)
group_evolution.add_argument('-res', '--map_resolution', type=int, default=20)

group_encoding = parser.add_argument_group("Encoding parameters")
group_encoding.add_argument('-e', '--encoding', choices=['b', 'base', 'd', 'direct'], default='direct')
group_encoding.add_argument('-mt', '--module_types', nargs='+', type=int, default=[3])
group_encoding.add_argument('-m', '--max_modules', type=int, default=20) 
group_encoding.add_argument('-d', '--max_depth', type=int, default=10)

group_controller = parser.add_argument_group("Controller parameters")
group_controller.add_argument('-c', '--controller', choices=['s', 'sine'], default='sine')
args = parser.parse_args()

def run():
    if args.seed is None:
        args.seed = np.random.randint(10000000)

    individual = Individual
    if args.encoding == 'base' or args.encoding == 'b':
        encoding = BaseEncoding
    elif args.encoding == 'direct' or args.encoding == 'd':
        encoding = DirectEncoding
    if args.controller == 'sine' or args.controller == 's':
        controller = SineController

    if args.algorithm == 'ea':
        ea = StandardEvolutionaryAlgorithm(args, individual, encoding, controller)
    elif args.algorithm == 'map_elites' or args.algorithm == 'map':
        ea = MapElites(args, individual, encoding, controller)

    if args.editor_mode or args.load:
        args.threads = 1

    ea.writetime()
    for i in range(args.generations):
        ea.step()
        ea.stat(i)
        ea.save_data(i)
    ea.writetime()

def check_data():
    if args.algorithm == 'ea':
        StandardEvolutionaryAlgorithm.load_data(args)
    elif args.algorithm == 'map_elites' or args.algorithm == 'map':
        MapElites.load_data(args, args.seed)

if __name__ == "__main__":
    if args.load:
        check_data()
    else:
        run()