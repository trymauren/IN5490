ea_config = {
# 'ea_type'						: 'cma_es_deap_without_restarts',
'ea_type'						: 'basic_deap',
'pop_size'						: 60,
'num_generations'				: 2, # only appliccable for basic_ea
'std_dev'						: 3,
'genome_len'					: 25,
'num_mov_repeat'				: 50,
'fitness_one_axis'				: False,
'lower_start_limit'				: -10,
'upper_start_limit'				: 10,
'num_restarts'					: 10,
'seed'							: 128,
'equal_frequency_all_limbs'		: True,
'basic_ea'						: False,
}

interface_config = {
'no_graphics'		:False,
'worker_id'			:6
}
