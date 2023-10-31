import pickle
from datetime import datetime
from utils import basic_deap
from utils import cma_es_deap
from config import ea_config, interface_config
from utils import functions
from utils import unity_stuff
from utils import fitness_evaluation
import numpy as np

import os
import sys

# Set path for data
root = os.path.dirname(os.path.abspath(__file__))
executable_path = os.path.join(root,'../executables/')
runs_path = os.path.join(root,'runs/')

def main():

	path_to_files = ''
	plotted = False
	if 'train' in sys.argv:

		# Create the simulation environment
		exe_path = functions.get_executable(executable_path, ea_config['pop_size'])
		unity_interface = unity_stuff.UnityInterface(executable_file=exe_path, **interface_config)

		# Run EA
		logbooks, halloffame = cma_es_deap.train(unity_interface)

		# Stop simulation environment
		unity_interface.stop_env()

		# Write results to file
		functions.dump_data(logbooks, halloffame, runs_path)


		if 'plot' in sys.argv:
			path_to_files = functions.get_newest_file_paths(runs_path)
			functions.make_plots_from_logbook(path_to_files[0])
			# functions.make_plots_from_halloffame(path_to_files[1])
			plotted = True
	else:

		if 'plot' in sys.argv:
			timestamp = input('Timestamp to plot: (enter to use latest)\n')
			if len(timestamp) > 1:
				path_to_files = functions.get_specific_file_paths(runs_path, timestamp)
			else:
				path_to_files = functions.get_newest_file_paths(runs_path)
				print(path_to_files)
			functions.make_plots_from_logbook(path_to_files[0])
			# functions.make_plots_from_halloffame(path_to_files[1])
		
		if 'sim_best' in sys.argv:
			pass

	if 'view_best' in sys.argv:
		fitness_evaluation.simulate_best(halloffame, unity_interface, 500)

if __name__ == "__main__":
    main()

