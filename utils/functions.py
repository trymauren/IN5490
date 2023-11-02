import os
from sys import platform
from datetime import datetime
import pickle
from config import interface_config
from utils import fitness_evaluation, unity_stuff, cma_es_deap

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import shelve

import matplotlib.pyplot as plt
from config import ea_config
from tkinter import filedialog as fd

def fetch_timestamp():
    """Gets time of when function is called

    Returns:
        datetime: year_month_day_hour_minute_second
    """
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def make_plots_from_logbook(path):
    """Makes plot from data stored in logbook

    Args:
        path (tulpe): path to logbook and halloffame in a tuple
    """
    print(path)
    path_w_out_extension = os.path.splitext(path[0])[0]
    file_name = path_w_out_extension[path_w_out_extension.find('_'):][1:] # Gets the date from 
    logbooks = []
    with shelve.open(path_w_out_extension, 'c') as fp: 
        for i, d in enumerate(fp):
            logbook = fp[str(i)]
            logbooks.append(logbook) 
            
    fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
    for logbook in logbooks:
        fitness = logbook.chapters['fitness'].select('avg')
        max_fitness = logbook.chapters['fitness'].select('max')
        amplitude = logbook.chapters['amplitude'].select('avg')
        phase_shift = logbook.chapters['phase_shift'].select('avg')

        # Plotting and styling for Average Fitness
        ax1.plot(fitness, marker='o', linestyle='-', color='b')
        ax1.set_title('Average Fitness', fontsize=14)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.grid(True)

        # Plotting and styling for Average Frequency
        ax2.plot(max_fitness, marker='s', linestyle='--', color='g')
        ax2.set_title('Max Fitness', fontsize=14)
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
    plt.savefig(f'4xplot_{file_name}.pdf',dpi=300)
    print(' -- Made plots')

def get_executable(executable_path, pop_size=ea_config['pop_size']):
    """
    Fetches the executable, automatically depending on the OS.
        Arg: Path to the executable folder
        Ret: Path to the correct executable
    """
    if platform == "linux" or platform == "linux2":
        ret = executable_path + 'exe_linux_test_m'
    elif platform == "darwin":
        if pop_size == 1:
            ret = executable_path + 'exe_mac_1.app'
            print('-- Training with 1 agents')
        elif pop_size <= 30:
            ret = executable_path + 'exe_mac_30.app' 
            print('-- Training with 30 agents')
        elif pop_size > 30:
            ret = executable_path + 'exe_mac_60.app' 
            print('-- Training with 60 agents')
    elif platform == "win32":
        if pop_size == 1:
            ret = executable_path + 'exe_pc_1/UnityEnvironment.exe'
            print('-- Training with 1 agents') 
        elif pop_size <= 30:
            ret = executable_path + 'exe_pc_30/UnityEnvironment.exe'
            print('-- Training with 30 agents')
        if pop_size > 30:
            ret = executable_path + 'exe_pc_60/UnityEnvironment.exe'
            print('-- Training with 60 agents') 
        
    return ret

def dump_data(logbooks, halloffame, runs_path):
    """
    Redericts dumping to the two helpers
        Arg: List containing Logbook() objects, halloffame containing individual, path to runs dir
        Ret: None
    """
    timestamp = fetch_timestamp()
    logbook_path = os.path.join(runs_path, 'logbook_' + timestamp)
    halloffame_path = os.path.join(runs_path,'halloffame_' + timestamp)
    dump_logbook(logbooks, logbook_path)
    dump_halloffame(halloffame, halloffame_path)

def dump_logbook(data, path):
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

def hof_data(path):
    """Reads HOF data from file and visualizes the agent with highest fitness

    Args:
        path(selve.DbfilenameShelf,selve.DbfilenameShelf): Tuple of shelve.DbfilenameShelf file paths

    Returns:
        data[][]: Nested list of all parameters for amplitude, frequency and phase shift for the best fitness
    """

    path_w_out_extension = os.path.splitext(path[1])[0]
    data = []
    with shelve.open(path_w_out_extension, 'c') as fp: 
        print(type(fp))
        for i, d in enumerate(fp):
            data.append(fp[str(i)])
    return data

def sim_best(runs_path, executable_path):
    """
    Simulate the best solution from the hall of fame.
    
    Args:
        runs_path (str): Path to the directory where simulation runs are stored.
        executable_path (str): Path to the executable file.
        
    Returns:
        None
    """
    timestamp = input('Timestamp to simulate: (enter to use latest), n to navigate\n')
    if timestamp == 'n':
        path_to_file = fd.askopenfilename()
    else:
        path_to_file = get_newest_file_paths(runs_path)

    halloffame = hof_data(path_to_file)

    exe_path = get_executable(executable_path, 1)
    unity_interface = unity_stuff.UnityInterface(executable_file=exe_path, **interface_config)

    fitness_evaluation.simulate_best(halloffame, 500, unity_interface)
    

def plot(runs_path):
    """
    Plot the results based on a given timestamp or the latest results.
    
    Args:
        runs_path (str): Path to the directory where simulation runs are stored.
        
    Returns:
        None
    """
    timestamp = input('Timestamp to plot: \'enter\' to use latest, \'n\' to navigate\n')
    if timestamp == 'n':
        print('Pick logbook file')
        path_to_files = fd.askopenfilename()
        # print('Pick halloffame file')
        # path_to_files[1] = fd.askopenfilename()
        # functions.get_specific_file_paths(runs_path, timestamp)
    else:
        path_to_files = get_newest_file_paths(runs_path)
    make_plots_from_logbook(path_to_files)
    # functions.make_plots_from_halloffame(path_to_files[1])

def train(runs_path, executable_path):
    """
    Train the simulation using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    Args:
        runs_path (str): Path to the directory where simulation runs are stored.
        executable_path (str): Path to the executable file.
        
    Returns:
        None
    """
    # Create the simulation environment
    exe_path = get_executable(executable_path, ea_config['pop_size'])
    unity_interface = unity_stuff.UnityInterface(executable_file=exe_path, **interface_config)

    # Run EA
    logbooks, halloffame = cma_es_deap.train(unity_interface, runs_path)

    # Stop simulation environmentsim
    unity_interface.stop_env()

    # Write results to file
    dump_data(logbooks, halloffame, runs_path)
    
    
def plot_latest(runs_path):
    """
    Plot the latest results from the simulation runs.
    
    Args:
        runs_path (str): Path to the directory where simulation runs are stored.
        
    Returns:
        None
    """
    path_to_files = get_newest_file_paths(runs_path)
    make_plots_from_logbook(path_to_files[0])