# Project
from utils import basic_deap
from utils import cma_es_deap
from utils import functions
from utils import unity_stuff
from utils import fitness_evaluation
from config import ea_config
from config import interface_config

# Python modules
import os
import sys
import numpy as np
from datetime import datetime

def main():

	root = os.path.dirname(os.path.abspath(__file__))
	executable_path = os.path.join(root,'executables/')
	runs_path = os.path.join(root,'runs/')
	path_to_files = ''
	plotted = False
	halloffame = 0

	if 'train' in sys.argv:

		# Create the simulation environment
		exe_path = functions.get_executable(executable_path, ea_config['pop_size'])
		print(exe_path)
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

	if 'sim_best' in sys.argv:
		if not halloffame:
			halloffame = functions.get_newest_file_paths()
		fitness_evaluation.simulate_best(halloffame, unity_interface, 500)

if __name__ == "__main__":
    main()

