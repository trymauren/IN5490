from utils import basic_deap
from config import train_config
from utils import functions
from utils import unity_stuff

import os
# Set path for data
executable_path = os.path.abspath('..') #Working directory path (Where program is run from)
executable_path += '/executables/'

def main():
	unity_interface = unity_stuff.UnityInterface(executable_file=functions.get_executable(executable_path), no_graphics=False, worker_id=1)
	best = basic_deap.train(train_config, unity_interface)
	print(best)
    
	unity_interface.stop_env() #funker dette ??

if __name__ == "__main__":
    main()

