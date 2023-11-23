from utils_threaded import functions
from config import ea_config, sim_config, interface_config
import os
import sys
import numpy
def main():
	root = os.path.dirname(os.path.abspath(__file__))
	executable_path = os.path.join(root,'executables/')
	runs_path = os.path.join(root,'runs/')
	print('ea_config:', ea_config)
	print('interface_config', interface_config)
	if 'train' in sys.argv:
		functions.train(ea_config['num_restarts'])
  
		if 'plot' in sys.argv:
			functions.plot_latest(runs_path)
   
	if 'plot' in sys.argv:
		functions.new_plot(runs_path)

	if 'plot_combined' in sys.argv:
		functions.make_combined_plots_from_logbook(runs_path)

	if 'plot_combined_bipop' in sys.argv:
		functions.make_combined_plots_bipop(runs_path)

	if 'sim_best' in sys.argv:
		functions.sim_best(runs_path, executable_path, 10)

if __name__ == "__main__":
    main()

