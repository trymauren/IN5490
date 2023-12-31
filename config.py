ea_config = {
'ea_type'						: 'basic',
# 'ea_type'						: 'basic_parallel',
# 'ea_type'						: 'cma_es_parallel',
# 'ea_type'						: 'cma_es_bipop',
'pop_size'						: 30, 		# not appliccable for bipop
'num_generations'				: 100, 		# not appliccable for bipop
'std_dev'						: 1, 		# only appliccable for cma
'genome_len'					: 36,
'num_mov_repeat'				: 50, 		# 50 is fine
'fitness_one_axis'				: False,
'lower_start_limit'				: 1,
'upper_start_limit'				: 4,
'num_restarts'					: 10,		# not appliccable for bipop
'seed'							: 128,		# should really not be necessary to adjust
'equal_frequency_all_limbs'		: False,
'n_cores'						: 10,		# 
}

interface_config = {
'no_graphics'					:False,
'worker_id'						:1,			# should be 1
'seed'							:13,
}

sim_config = {
'n_agents'						:1,
}
