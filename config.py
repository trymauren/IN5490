ea_config = {
'ea_type'						: 'cma_es_bipop',
# 'ea_type'						: 'cma_es',
# 'ea_type'						: 'basic',
'pop_size'						: 180,
'num_generations'				: 100,
'std_dev'						: 3,
'genome_len'					: 36,
'num_mov_repeat'				: 50,
'fitness_one_axis'				: True,
'lower_start_limit'				: -10,
'upper_start_limit'				: 10,
'num_restarts'					: 10,
'seed'							: 128,
'equal_frequency_all_limbs'		: False,
'n_cores'						: 12,
}

interface_config = {
'no_graphics'		:False,
'worker_id'			:5
}
