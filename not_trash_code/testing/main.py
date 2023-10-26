from utils import basic_deap
from utils import cma_es_deap
from config import train_config
from utils import functions
from utils import unity_stuff
from utils import fitness_evaluation

import os
# Set path for data
root = os.path.abspath('../')
executable_path = root + '/executables/'

def main():
	params = {'executable_file':functions.get_executable(executable_path),'no_graphics':False, 'worker_id':1}
	unity_interface = unity_stuff.UnityInterface(**params)
	# best = basic_deap.train(train_config, unity_interface)
	halloffame = cma_es_deap.train(unity_interface)
	print(halloffame)
    
	# unity_interface.stop_env() #funker dette ??
	# visualization = False
	# best_res = None
      
	# if visualization:
	# 	vis_unity_interface = unity_stuff.UnityInterface(executable_file=functions.get_executable(executable_path), no_graphics=False, worker_id=1)
	# 	fitness_evaluation.evaluate_population(best_res, vis_unity_interface)
	# 	vis_unity_interface.stop_env()
	# else:
	# 	unity_interface = unity_stuff.UnityInterface(executable_file=functions.get_executable(executable_path), no_graphics=False, worker_id=1)
	# 	best = basic_deap.train(train_config, unity_interface)
	# 	print(best)
	# 	unity_interface.stop_env() #funker dette ??
		

if __name__ == "__main__":
    main()

