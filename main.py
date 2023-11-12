from utils_threaded_high import functions
from config import sim_config
import os
import sys

def main():
	root = os.path.dirname(os.path.abspath(__file__))
	executable_path = os.path.join(root,'executables/')
	runs_path = os.path.join(root,'runs/')

	if 'train' in sys.argv:
		functions.train(10)
  
		if 'plot' in sys.argv:
			functions.plot_latest(runs_path)
   
	if 'plot' in sys.argv:
		functions.plot(runs_path)

	if 'sim_best' in sys.argv:
		functions.sim_best(runs_path, executable_path, sim_config['n_agents'])

if __name__ == "__main__":
    main()

