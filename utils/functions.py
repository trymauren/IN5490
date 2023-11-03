import os
from sys import platform
from datetime import datetime
import pickle
from config import interface_config, ea_config
from utils import fitness_evaluation, unity_stuff, cma_es_deap, basic_deap

import shelve
import matplotlib.pyplot as plt
from tkinter import filedialog as fd

def make_plots_from_logbook(path, runs_path):
    """Makes plot from data stored in logbook

    Args:
        path (tulpe): path to logbook and halloffame in a tuple
    """
    path_w_out_extension = os.path.splitext(path)[0] # important to make shelve work!
    file_name = path_w_out_extension[path_w_out_extension.find('_2023'):][1:] # Gets the date from path
    logbooks = []
    
    with shelve.open(path_w_out_extension, 'c') as fp: 
        for i, d in enumerate(fp):
            logbook = fp[str(i)]
            logbooks.append(logbook) 
            
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(15,10))
    for logbook in logbooks:
        fitness = logbook.chapters['fitness'].select('avg')
        max_fitness = logbook.chapters['fitness'].select('max')
        avg_freq = logbook.chapters['frequency'].select('avg')
        
        # Plotting and styling for Average Fitness
        ax1.plot(fitness)
        ax1.set_title('Average Fitness', fontsize=14)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.grid(True)

        # Plotting and styling for Average Frequency
        ax2.plot(max_fitness)
        ax2.set_title('Max Fitness', fontsize=14)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Fitness', fontsize=12)
        ax2.grid(True)
        
        ax3.plot(avg_freq)
        ax3.set_title('Average Frequency', fontsize=14)
        ax3.set_xlabel("Generation", fontsize=12)
        ax3.set_ylabel('Frequency',fontsize=12)
        ax3.grid(True)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f'{runs_path}/4xplot_{file_name}.pdf',dpi=300)
    print(' -- Made plots')


def get_halloffame_data(path):
    """Reads HOF data from file and visualizes the agent with highest fitness

    Args:
        path(selve.DbfilenameShelf,selve.DbfilenameShelf): Tuple of shelve.DbfilenameShelf file paths

    Returns:
        data[][]: Nested list of all parameters for amplitude, frequency and phase shift for the best fitness
    """

    path_w_out_extension = os.path.splitext(path)[0]
    data = []
    with shelve.open(path_w_out_extension, 'c') as fp: 
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
    timestamp = input('Timestamp to simulate: \'enter\' to use latest, \'n\' to navigate\n')
    if timestamp == 'n':
        path_to_halloffame = fd.askopenfilename()
    else:
        path_to_halloffame = get_newest_halloffame(runs_path)

    halloffame = get_halloffame_data(path_to_halloffame)
    exe_path = get_executable(executable_path, 1)
    unity_interface = unity_stuff.UnityInterface(executable_file=exe_path, worker_id=(interface_config['worker_id'] + 1))
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
        path_to_logbook = fd.askopenfilename()
    else:
        path_to_logbook = get_newest_logbook(runs_path)
    make_plots_from_logbook(path_to_logbook, runs_path)
    

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
    if ea_config['ea_type'] == 'basic':
        logbooks, halloffame = basic_deap.train(unity_interface, runs_path)
    elif ea_config['ea_type'] == 'cma_es':
        logbooks, halloffame = cma_es_deap.train_normal(unity_interface, runs_path)
    elif ea_config['ea_type'] == 'cma_es_bipop':
        logbooks, halloffame = cma_es_deap.train_bipop(unity_interface, runs_path)
    else:
        print(f' -- Invalid ea_type in config: {ea_config["ea_type"]}')
        unity_interface.stop_env()
        exit()
    
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
    path_to_logbook = get_newest_logbook(runs_path)
    make_plots_from_logbook(path_to_logbook)


### --- Read from file --- ###
def get_newest_logbook(path_to_runs):
    """
    Fetches the latest created logbook file paths
        Arg: Path to the directory containing the data
        Ret: Paths to the latest created files containing logbook
    """
    files = os.listdir(path_to_runs)
    paths_logbook = [os.path.join(path_to_runs, basename) for basename in files if 'logbook_' in basename]
    if len(paths_logbook) == 0:
        print(f' -- Directory {path_to_runs} does not contain any logbook history')
        return None
    latest_logbook = max(paths_logbook, key=os.path.getctime)
    print(f' -- Fetched {latest_logbook}')
    return latest_logbook

def get_newest_halloffame(path_to_runs):
    """
    Fetches the latest created halloffame path
        Arg: Path to the directory containing the data
        Ret: Paths to the latest created files containing halloffame
    """
    files = os.listdir(path_to_runs)
    paths_halloffame = [os.path.join(path_to_runs, basename) for basename in files if 'halloffame_' in basename]
    if len(paths_halloffame) == 0:
        print(f' -- Directory {path_to_runs} does not contain any halloffame history')
        return None
    latest_halloffame = max(paths_halloffame, key=os.path.getctime)
    print(f' -- Fetched {latest_halloffame}')
    return latest_halloffame

### --- Write to file --- ###
def dump_data(logbooks, halloffame, runs_path):
    """
    Redericts dumping to the two helpers
        Arg: List containing Logbook() objects, halloffame containing individual, path to runs dir
        Ret: None
    """
    timestamp = get_timestamp()
    logbook_path = os.path.join(runs_path, 'logbook_' + timestamp)
    halloffame_path = os.path.join(runs_path, 'halloffame_' + timestamp)
    txt_path = os.path.join(runs_path, 'config_' + timestamp)
    dump_logbook(logbooks, logbook_path)
    dump_halloffame(halloffame, halloffame_path)
    config_to_txt(txt_path)

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
    
def config_to_txt(path_to_file):
    """
    Store the dictionaries from config.py in a TXT file within the specified path.
    
    Args:
        path (str): Path including filename of the TXT file to store the data.
        
    Returns:
        None
    """
    with open(path_to_file, 'w') as txtfile:
        
        # Write ea_config entries
        txtfile.write("[ea_config]\n")
        for key, value in ea_config.items():
            txtfile.write(f"{key}: {value}\n")
            
        txtfile.write("\n")  # Adding a newline for separation
        
        # Write interface_config entries
        txtfile.write("[interface_config]\n")
        for key, value in interface_config.items():
            txtfile.write(f"{key}: {value}\n")


### --- Helpers --- ###
def get_timestamp():
    """Gets time of when function is called

    Returns:
        datetime: year_month_day_hour_minute_second
    """
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

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
            print('-- Environment with 1 agent')
        elif pop_size <= 30:
            ret = executable_path + 'exe_mac_30.app' 
            print('-- Environment with 30 agents')
        elif pop_size > 30:
            ret = executable_path + 'exe_mac_60.app' 
            print('-- Environment with 60 agents')
    elif platform == "win32":
        if pop_size == 1:
            ret = executable_path + 'exe_pc_1/UnityEnvironment.exe'
            print('-- Environment with 1 agent') 
        elif pop_size <= 30:
            ret = executable_path + 'exe_pc_30/UnityEnvironment.exe'
            print('-- Environment with 30 agents')
        if pop_size > 30:
            ret = executable_path + 'exe_pc_60/UnityEnvironment.exe'
            print('-- Environment with 60 agents') 
        
    return ret

