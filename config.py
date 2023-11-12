ea_config = {
# 'ea_type'						: 'cma_es_bipop',
 'ea_type'						: 'cma_es',
# 'ea_type'						: 'basic',
'pop_size'						: 60,
'num_generations'				: 100,
'std_dev'						: 3,
'genome_len'					: 36,
'num_mov_repeat'				: 50, 		# 50 is fine
'fitness_one_axis'				: False,
'lower_start_limit'				: -10,
'upper_start_limit'				: 10,
'num_restarts'					: 10,		# not appliccable for bipop
'seed'							: 128,		# should really not be necessary to adjust
'equal_frequency_all_limbs'		: False,
'n_cores'						: 8,		# 
}

interface_config = {
'no_graphics'					:False,
'worker_id'						:1,			# should be 1
'seed'							:13,
}

sim_config = {
'n_agents'						:1,
}