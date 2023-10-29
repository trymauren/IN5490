import pickle
from time import time
from utils import basic_deap
from utils import cma_es_deap
from config import train_config
from utils import functions
from utils import unity_stuff
from utils import fitness_evaluation

import os
# Set path for data
root = os.path.abspath('') # '../' for noobs
executable_path = root + '/executables/'

def main():
	params = {'executable_file':functions.get_executable(executable_path),'no_graphics':False, 'worker_id':1}
	unity_interface = unity_stuff.UnityInterface(**params)
	# best = basic_deap.train(train_config, unity_interface)
	logbooks, halloffame = cma_es_deap.train(unity_interface)
	pickle.dump(logbooks, f'logbook_{time.time()}')
	unity_interface.stop_env() #funker dette ?? kanskje, ja!!!!
	

if __name__ == "__main__":
    main()

