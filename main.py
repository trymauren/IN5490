# Project
from utils import basic_deap, cma_es_deap, functions, unity_stuff, fitness_evaluation
from config import ea_config
from config import interface_config

# Python modules
import os
import sys
import numpy as np
from datetime import datetime
from tkinter import filedialog as fd

def main():

	root = os.path.dirname(os.path.abspath(__file__))
	executable_path = os.path.join(root,'executables/')
	runs_path = os.path.join(root,'runs/')

	if 'train' in sys.argv:
		functions.train(runs_path, executable_path)

		if 'plot' in sys.argv:
			functions.plot_latest(runs_path)

	if 'plot' in sys.argv:
		functions.plot(runs_path)

	if 'sim_best' in sys.argv:
		functions.sim_best(runs_path, executable_path)

if __name__ == "__main__":
    main()

