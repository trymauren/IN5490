from utils import basic_deap
from config import train_config
from utils import functions
from utils import unity_stuff

import os
# Set path for data
executable_path = os.path.abspath('..') #Working directory path (Where program is run from)
executable_path += '/executables/'

def main():
	params = {'executable_file':functions.get_executable(executable_path),'no_graphics':True, 'worker_id':1}
	unity_interface = unity_stuff.UnityInterface(**params)
	best = basic_deap.train(train_config, unity_interface)
	print(best)
    
	unity_interface.stop_env() #funker dette ??

if __name__ == "__main__":
    main()
