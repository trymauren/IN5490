import os
from sys import platform
from datetime import datetime
import numpy as np
from config import interface_config, ea_config, sim_config
from utils_threaded_high import fitness_evaluation, unity_stuff, cma_es_deap, basic_deap

import shelve
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import multiprocessing

def make_plots_from_logbook(paths, runs_path):
    """Makes plot from data stored in logbook

    Args:
        paths (tuple): path to logbook and halloffame in a tuple
    """
    # Initialize the plot outside the loop
    fig_fitness, ax_fitness = plt.subplots(figsize=(3.5,2.6))

    for path in paths:
        path_w_out_extension = os.path.splitext(path)[0]
        file_name = path_w_out_extension[path_w_out_extension.find('_2023'):][1:]
        logbooks = []

        with shelve.open(path_w_out_extension, 'c') as fp:
            for i, d in enumerate(fp):
                logbook = fp[str(i)]
                logbooks.append(logbook)

        all_max_fitness = [logbook.chapters['fitness'].select('max') for logbook in logbooks]
        all_max_fitness = handle_variable_lengths(all_max_fitness)

        avg_of_max_fitness = np.nanmean(all_max_fitness, axis=0)
        std_dev_max_fitness = np.nanstd(all_max_fitness, axis=0)
        generations = range(len(avg_of_max_fitness))

        # Plot each set of fitness data on the same axes
        ax_fitness.plot(generations, avg_of_max_fitness, label=f'{get_ea_strat(path)}')
        ax_fitness.fill_between(generations, avg_of_max_fitness - std_dev_max_fitness, 
                                avg_of_max_fitness + std_dev_max_fitness, alpha=0.3)

    # Set titles, labels, and other properties of the plot
    ax_fitness.set_title(f'Fitness: {get_fitness_func(path)}, Freq: {get_freq(path)}', fontsize=12)
    ax_fitness.set_xlabel(f'Generation', fontsize=11)
    ax_fitness.set_ylabel(f'Fitness', fontsize=11)
    ax_fitness.grid(True)
    ax_fitness.legend(loc='upper left')

        # Save the combined plot to a PDF file for fitness
    fig_fitness.savefig(f'{runs_path}/fitness_performance_metrics_with_error_bands.pdf', dpi=400, bbox_inches='tight')
    plt.close(fig_fitness)

    print('All plots saved')
    
def get_ea_strat(path):
    """Gets ea type from config.txt

    Args:
        path : path to txt

    Returns:
        str: ea_type
    """
    last_slash = path.rfind('/')
    path = path[:last_slash]+'/config.txt'
    print(path)
    with open(path, 'r') as file:
        for line in file:
            if line.startswith("ea_type"):
                # Splitting the line on ':' and stripping any whitespace
                _, value = line.split(":", 1)
                return value.strip()
        return "ea_type not found in the file."
    
def get_fitness_func(path):
    """Gets which fitness function was used from config.txt

    Args:
        path : path to config.txt

    Returns:
        str: fitness_func
    """
    last_slash = path.rfind('/')
    path = path[:last_slash]+'/config.txt'
    with open(path, 'r') as file:
        for line in file:
            if line.startswith("fitness_one_axis"):
                # Splitting the line on ':' and stripping any whitespace
                _, value = line.split(":", 1)
                return 'One dir' if value.strip()=="True" else 'All dir'
        return "Fitness not found in config.tzt."
    
def get_freq(path):
    """Gets which freq setting was used from config.txt

    Args:
        path : path to config.txt

    Returns:
        str: freq_setting
    """
    last_slash = path.rfind('/')
    path = path[:last_slash]+'/config.txt'
    with open(path, 'r') as file:
        for line in file:
            if line.startswith("equal_frequency_all_limbs"):
                # Splitting the line on ':' and stripping any whitespace
                _, value = line.split(":", 1)
                return 'equal freq' if value.strip()=="True" else 'not equal freq'
        return "Fitness not found in config.tzt."
    
def new_plot(runs_path):
    """
    Plot the results of different algorithms.
    
    Args:
        runs_path (str): Path to the directory where simulation runs are stored.
        
    Returns:
        None
    """
    timestamp = input('\'enter\' to use choose, \'q\' to quit\n')
    paths_to_logbooks = []
    while not timestamp:
        paths_to_logbooks.append(fd.askopenfilename())
        timestamp = input('\'enter\' to use choose, \'q\' to quit\n')
    make_plots_from_logbook(paths_to_logbooks, runs_path)
    


def make_combined_plots_from_logbook(runs_path):

    
    # path_w_out_extension = os.path.splitext(path)[0] # important to make shelve work!
    # file_name = path_w_out_extension[path_w_out_extension.find('_2023'):][1:] # Gets the date from path
    
    filenames = []

    cont = 'n'
    while cont == 'n':
        filenames.append(os.path.splitext(fd.askopenfilename())[0])
        cont = input('n to continue')
    fig = plt.figure(figsize=(15,9))
    for filename in filenames:  
        logbooks = []
        with shelve.open(filename, 'c') as fp: 
            for i, d in enumerate(fp):
                logbook = fp[str(i)]
                logbooks.append(logbook) 
                # print(logbook.chapters["frequency"].select("avg"))               
        # Initialize lists to store the data from all logbooks
        all_fitness = []
        all_max_fitness = []
        all_avg_freq = []

        # Extract the data from each logbook and append it to the lists
        for logbook in logbooks:
            fitness = logbook.chapters['fitness'].select('avg')
            max_fitness = logbook.chapters['fitness'].select('max')
            # avg_freq = logbook.chapters['frequency'].select('avg')
            
            all_fitness.append(fitness)
            all_max_fitness.append(max_fitness)
            # all_avg_freq.append(avg_freq)

        # Convert lists to numpy arrays for computation
        all_fitness = np.array(all_fitness)
        all_max_fitness = np.array(all_max_fitness)
        # all_avg_freq = np.array(all_avg_freq)

        # Compute the average and standard deviation across all logbooks
        avg_of_fitness = np.mean(all_fitness, axis=0)
        std_dev_fitness = np.std(all_fitness, axis=0)

        avg_of_max_fitness = np.mean(all_max_fitness, axis=0)
        std_dev_max_fitness = np.std(all_max_fitness, axis=0)

        # # Compute average and standard deviation only for frequency
        # avg_of_avg_freq = np.mean(all_avg_freq, axis=0)
        # std_dev_avg_freq = np.std(all_avg_freq, axis=0)

        sub_label = input('Input label\n')
        # Plotting each average line with error bands for fitness
        # plt.plot(avg_of_fitness, label=f'Average Fitness {sub_label}')
        # plt.fill_between(range(len(avg_of_fitness)), avg_of_fitness - std_dev_fitness, avg_of_fitness + std_dev_fitness, alpha=0.2)

        plt.plot(avg_of_max_fitness, label=f'{sub_label}')
        plt.fill_between(range(len(avg_of_max_fitness)), avg_of_max_fitness - std_dev_max_fitness, avg_of_max_fitness + std_dev_max_fitness, alpha=0.2)

        # Styling the plot for fitness
    title = input('Input title for plot\n')
    plt.title(f'{title}', fontsize=12)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    ax = plt.gca()
    ax.set_ylim(0,50)
    plt.grid(True)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1))

        # Save the combined plot to a PDF file for fitness


        # # Create a single plot for average frequency
        # fig_freq, ax_freq = plt.subplots()

        # # Plotting the average frequency with error bands
        # ax_freq.plot(avg_of_avg_freq, label='Average Frequency', color='green')
        # ax_freq.fill_between(range(len(avg_of_avg_freq)), avg_of_avg_freq - std_dev_avg_freq, avg_of_avg_freq + std_dev_avg_freq, color='green', alpha=0.2)

        # # Styling the plot for frequency
        # ax_freq.set_title('Frequency Performance', fontsize=20)
        # ax_freq.set_xlabel('Generation', fontsize=19)
        # ax_freq.set_ylabel('Frequency', fontsize=19)
        # ax_freq.grid(True)
        # ax_freq.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Save the plot to a PDF file for frequency
    
    plt.savefig(f'{runs_path}/{title}.pdf', dpi=400, bbox_inches='tight')
    print('All plots saved')

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

def sim_best(runs_path, executable_path, n_agents):
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
    exe_path = get_executable(executable_path, n_agents)
    unity_interface = unity_stuff.UnityInterface(executable_file=exe_path, worker_id=(interface_config['worker_id'] + 1))
    fitness_evaluation.simulate_best(halloffame, unity_interface)
    
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


def train(runs):
    """
    Train the simulation using a basic EA, CMA-ES or bi-pop CMA-ES.
    
    Args:
        runs (int): Number of times to run the evolution
        
    Returns:
        None
    """
    # Run EA
    root = os.path.abspath('')
    executable_path = os.path.join(root,'executables/')
    runs_path = os.path.join(root,'runs/')
    start_worker = interface_config['worker_id']
    start_seed = ea_config['seed']
    logbooks, halloffames = 0,0
    worker_ids = range(start_worker, start_worker + runs)
    seeds = range(start_seed, start_seed + runs)
    args = zip(worker_ids, seeds)
    

    if ea_config['ea_type'] == 'basic':
        logbooks, halloffames = basic_deap.train()

    elif ea_config['ea_type'] == 'cma_es_bipop':
        # logbooks, halloffames = cma_es_deap.train_bipop(interface_config['worker_id'],ea_config['seed'])
        with multiprocessing.Pool(runs) as pool:
            rets = pool.starmap(cma_es_deap.train_bipop, args)

    elif ea_config['ea_type'] == 'basic_parallel':
        with multiprocessing.Pool(runs) as pool:
            rets = pool.starmap(basic_deap.train_parallel, args)

    elif ea_config['ea_type'] == 'cma_es_parallel':
        with multiprocessing.Pool(runs) as pool:
            rets = pool.starmap(cma_es_deap.train_parallel, args)

    else:
        print(f' -- Invalid ea_type in config: {ea_config["ea_type"]}')
        print(f' -- Exiting')
        exit()
    
    if not logbooks or not halloffames:
        logbooks, halloffames = map(list, zip(*rets))

    dump_data(logbooks, halloffames, runs_path)
    
 
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
    dir_path = os.path.join(runs_path, timestamp)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    logbook_path = os.path.join(dir_path, 'logbook')
    halloffame_path = os.path.join(dir_path, 'halloffame')
    txt_path = os.path.join(dir_path, 'config.txt')
    dump_logbook(logbooks, logbook_path)
    dump_halloffame(halloffame, halloffame_path)
    config_to_txt(txt_path, timestamp)

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
    print(' -- Dumped halloffames')
    
def config_to_txt(path, timestamp):
    """
    Store the dictionaries from config.py in a TXT file within the specified path.
    
    Args:
        path (str): Path including filename of the TXT file to store the data.
        
    Returns:
        None
    """
    with open(path, 'w') as txtfile:
        #write timestamp to config so it can be accessed later
        txtfile.write(f"Timestamp: {timestamp}\n")
        txtfile.write("[Timestamp]\n")
        # Write ea_config entries
        txtfile.write("[ea_config]\n")
        for key, value in ea_config.items():
            txtfile.write(f"{key}: {value}\n")
            
        txtfile.write("\n")  # Adding a newline for separation
        
        # Write interface_config entries
        txtfile.write("[interface_config]\n")
        for key, value in interface_config.items():
            txtfile.write(f"{key}: {value}\n")

        txtfile.write("\n")  # Adding a newline for separation
        
        # Write interface_config entries
        txtfile.write("[sim_config]\n")
        for key, value in sim_config.items():
            txtfile.write(f"{key}: {value}\n")


### --- Helpers --- ###
def uniform_length_check(lst):
    return all(len(item) == len(lst[0]) for item in lst)

def handle_variable_lengths(lst):
    if uniform_length_check(lst):
        return np.array(lst)
    else:
        # Handling variable lengths, option 1: Use dtype='object'
         return np.array(lst, dtype='object')

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
        ret = executable_path + 'exe_linux_60/linux_60.x86_64'
        print(' -- Using environment with 60 agents')
    elif platform == "darwin":
        if pop_size == 1:
            ret = executable_path + 'exe_mac_1.app'
            print(' -- Using environment with 1 agent')
        elif pop_size <= 30:
            ret = executable_path + 'exe_mac_30.app' 
            print(' -- Using environment with 30 agents')
        elif pop_size > 30:
            ret = executable_path + 'exe_mac_60.app' 
            print(' -- Using environment with 60 agents')
    elif platform == "win32":
        if pop_size == 1:
            ret = executable_path + 'exe_pc_1/UnityEnvironment.exe'
            print(' -- Using environment with 1 agent')
        elif pop_size <= 30:
            ret = executable_path + 'exe_pc_30/UnityEnvironment.exe'
            print(' -- Using environment with 30 agents')
        if pop_size > 30:
            ret = executable_path + 'exe_pc_60/UnityEnvironment.exe'
            print(' -- Using environment with 60 agents')
        
    return ret
