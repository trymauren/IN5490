import os
from sys import platform
from datetime import datetime
import pickle

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import shelve

import matplotlib.pyplot as plt

def make_plots_from_logbook(path):
    path_w_out_extension = os.path.splitext(path)[0]
    logbooks = []
    with shelve.open(path_w_out_extension, 'c') as fp: 
        for i, d in enumerate(fp):
            logbook = fp[str(i)]
            logbooks.append(logbook) 

    for logbook in logbooks:
        fitness = logbook.chapters['fitness'].select('avg')
        frequency = logbook.chapters['frequency'].select('avg')
        amplitude = logbook.chapters['amplitude'].select('avg')
        phase_shift = logbook.chapters['phase_shift'].select('avg')

        fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
        # Plotting and styling for Average Fitness
        ax1.plot(fitness, marker='o', linestyle='-', color='b')
        ax1.set_title('Average Fitness', fontsize=14)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.grid(True)

        # Plotting and styling for Average Frequency
        ax2.plot(frequency, marker='s', linestyle='--', color='g')
        ax2.set_title('Average Frequency', fontsize=14)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True)

        # Plotting and styling for Average Amplitude
        ax3.plot(amplitude, marker='x', linestyle='-.', color='r')
        ax3.set_title('Average Amplitude', fontsize=14)
        ax3.set_xlabel('Generation', fontsize=12)
        ax3.set_ylabel('Amplitude', fontsize=12)
        ax3.grid(True)

        # Plotting and styling for Average Phase Shift
        ax4.plot(phase_shift, marker='^', linestyle=':', color='m')
        ax4.set_title('Average Phase Shift', fontsize=14)
        ax4.set_xlabel('Generation', fontsize=12)
        ax4.set_ylabel('Phase Shift', fontsize=12)
        ax4.grid(True)
        plt.show()
    print(' -- Made plots')

def get_executable(executable_path):
    """
    Fetches the executable, automatically depending on the OS.
        Arg: Path to the executable folder
        Ret: Path to the correct executable
    """
    if platform == "linux" or platform == "linux2":
        ret = executable_path + 'CHANGE_THIS'
    elif platform == "darwin":
        ret = executable_path + 'exe_mac_30.app' 
    elif platform == "win32":
        ret = executable_path + 'exe_pc_30/UnityEnvironment.exe'

    return ret

def dump_data(logbooks, halloffame, runs_path):
    """
    Redericts dumping to the two helpers
        Arg: List containing Logbook() objects, halloffame containing individual, path to runs dir
        Ret: None
    """
    timestamp = datetime.today().strftime('%Y-%m-%d|%H:%M:%S')
    logbook_path = os.path.join(runs_path, 'logbook_' + timestamp)
    halloffame_path = os.path.join(runs_path,'halloffame_' + timestamp)
    dump_logbook(logbooks, logbook_path)
    dump_halloffame(halloffame, halloffame_path)

def dump_logbook(data, path):
    print('hrireionreoijfo')
    """
    Dumps the data using pickle
        Arg: List containing Logbook() objects
        Ret: None
    """

    with shelve.open(path, 'c') as fp: 
        for i, d in enumerate(data):
            fp[str(i)] = d
    print(' -- Dumped logbooks')

def dump_halloffame(data, path):
    """
    Dumps the data using pickle
        Arg: List containing Logbook() objects
        Ret: None
    """
    with shelve.open(path, 'c') as fp: 
        for i, d in enumerate(data):
            fp[str(i)] = d
    print(' -- Dumped halloffame')

def get_newest_file_paths(path_to_dir):
    """
    Fetches the latest created logbook and halloffame file paths
        Arg: Path to the directory containing the data
        Ret: Paths to the latest created files containing logbook and halloffame
    """
    files = os.listdir(path_to_dir)
    paths_logbook = [os.path.join(path_to_dir, basename) for basename in files if 'logbook_' in basename]
    paths_halloffame = [os.path.join(path_to_dir, basename) for basename in files if 'halloffame' in basename]
    if len(paths_halloffame) == 0 or len(paths_logbook) == 0:
        print(f' -- Directory {path_to_dir} is empty')
        return None,None
    latest_logbook = max(paths_logbook, key=os.path.getctime)
    latest_halloffame = max(paths_halloffame, key=os.path.getctime)
    print(f' -- Fetched {latest_logbook}')
    print(f' -- Fetched {latest_halloffame}')
    return latest_logbook, latest_halloffame

def get_specific_file_paths(path, timestamp):
    """
    Fetches specific logbook and halloffame file paths
        Arg: Path to the directory containing the data
        Ret: Paths to the files containing logbook and halloffame
    """
    files = os.listdir(path)
    files = [os.path.join(path, basename) for basename in files]

    path_to_logbook = os.path.join(path, 'logbook_') + timestamp
    path_to_halloffame = os.path.join(path, 'halloffame_') + timestamp

    if path_to_logbook in files:
        print(f' -- Fetched {path_to_logbook}')

    else:
        print(f' -- File {path_to_logbook} does not exist')
        path_to_logbook = None

    if path_to_halloffame in files:
        print(f' -- Fetched {path_to_halloffame}')
    
    else:
        print(f' -- File {path_to_halloffame} does not exist')
        path_to_halloffame = None

    return path_to_logbook, path_to_halloffame