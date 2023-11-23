# IN5490 project - Evolutionary Robotics, Reality Gap. 

The initial goal of this project was to try different physics simulators and other noise to create more robust robots,
reducing the reality gap. However, due to time constraints, this was not possible, and the focus changed to comparing different
evolutionary algorithms, fitness functions and frequency configurations.

The code became a little messy, but anyways: The code runs evolutionary algorithms implemented with Python and the DEAP library.
The individuals are evaluated in Unity.

### Run ###

#### 1. Adjust config.py ####
- To run an EA, change the config files 'ea_type' to the desired evolutionary algorithm (ea). The types can be 'basic',
  'basic_parallel', 'cma_es_parallel' and 'cma_es_bipop'.
- Population size can be set to whatever.
  Note that the executables running the evaluation of each individual contains 1, 30 and 60 Unity crawlers for Mac and Windows.
  This means that for a population of < 60 individuals, each individual will be evaluated in parallel.
- If you want to adjust the fitness function, swap the 'fitness_one_axis' to True or False. True will make the fitness function
   assign a fitness value based on how far away from the starting point the crawler reaches, measured along the axis pointing
   forward when the crawlers are in their default position. False will make the fitness function assign a fitness value based on
   how far away from the starting point the crawler reaches, measured as euclidean, horizontal distance.
- 'std_dev', 'lower_start_limit', 'upper_start_limit' all control the initial population. Setting these values to 2,-1,1 means
  initiating the population where each genotype is in the range -1,1 with a standard deviation of 2.
- 'equal_frequency_all_limbs' controls whether evolution evolves a different frequency for all movement directions, or if it
  uses the same frequency for all limbs.

- 'no_graphics' should be set to 'False' to run in visual mode, and 'True' when running on ML nodes or to save computation when
  running local. If several instances of the python program main.py are ran on the same computer, worker_id should be increased
  by 1 manually to avoid error.

- The sim_config can be ignored.


#### 2. Multiprocessing ####
- The project currently uses the Python multiprocessing library to enable true parallell execution. Currently, the 'num_restarts'
  in the 'ea_config' in the config.py file controls how many instances of the Python program is launched. You do not have to change
  the 'worker_id' when running "python3 main.py", only if this command is repeated several times on the same computer without
  closing previous instances.

### Article ###
Article can be found as article.pdf

### YouTube ###
Have a look at our YouTube channel for videos of some of the best solutions found: https://www.youtube.com/watch?v=A46DHS6cHPM

### Thanks ###
Thanks to Kyrre Glette for the supervision of the project.
Thanks to Frank Veenstra for code inspiration and Unity help.

### Disclaimer ###
The code is unfinished, contains stuff that is not necessary and could be written more readable. Also, the docstrings are not correct.
