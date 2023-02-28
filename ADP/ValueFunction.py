from copy import deepcopy, copy

class ADP():
	"""
	ADP value function

	Attributes
	----------
	V : dict / dict{dict}
		Single V dictionary representing the value funciton if no aggregation used, 2D dictionary if aggregation used
	BAKF_Vs : dict / dict{dict}
		Dictionary or 2D dictionary with values relating to each post-state and its BAKF values
	stored_values : dict{dict}
		Storing the relevent information for each post-decision state (such as times visited, previous values, etc.)
	n : int
		Total number of iterations (days) trained
	alpha : float
		Learning rate / step size
	lamb : float
		Value used for defining Harmonic learning rate
	stepsize_type : str
		Type of step size to consider
	levels : int
		Number of aggregation levels
	zones : dict{dict}
		For each aggregation level, tells us which zone each node belongs to
	"""
	def __init__(self, stepsize_type, levels, zones):
		self.V = {} if levels == 1 else {i : {} for i in range(levels)}
		self.BAKF_Vs = {} if levels == 1 else {i : {} for i in range(levels)}
		self.stored_values = {i : {} for i in range(levels)}
		self.n = 1
		self.alpha = 0.05 if stepsize_type == 'F' else 1
		self.lamb = 25
		self.stepsize_type = stepsize_type
		self.levels = levels
		self.zones = zones

	def update_v(self, state_post_pairs, duals):
		"""
		Updating non-aggregated value function

		Parameters
		----------
		state_post_pairs : list[tuple]
			Pairing of post-decision states and their corresponding state values in the next time step
		duals : float
			Dual values associated with each unique state value
		"""
		# For every state and its corresponding post-decision state from the previous time-step
		for state,post_state in state_post_pairs:
			# Get the dual value corresponding to the state from the current time steps ILP
			dual_value = duals[state]
			if self.stepsize_type != 'B':
				# Get the updated value
				update = (1 - self.alpha) * (self.V.get(str(post_state),0)) + (self.alpha) * (dual_value)
			else:
				bakf_values = self.BAKF_Vs.get(str(post_state),{'alpha': 0.05, 'step': 0.01, 'step_bar': 0.2, 'lamb': 25, 'beta': 0, 'delta': 0.001, 'sigma_s': 0})
				bakf_values['step'] = bakf_values['step'] / (1 + bakf_values['step'] - bakf_values['step_bar'])
				bakf_values['beta'] = (1 - bakf_values['step']) * bakf_values['beta'] + bakf_values['step'] * (dual_value - self.V.get(str(post_state),0))
				bakf_values['delta'] = (1 - bakf_values['step']) * bakf_values['delta'] + bakf_values['step'] * (dual_value - self.V.get(str(post_state),0))**2
				if self.n == 1:
					bakf_values['alpha'] = 1
					bakf_values['lamb'] = bakf_values['alpha']**2
				else:
					bakf_values['sigma_s'] = (bakf_values['delta'] - (bakf_values['beta'])**2) / (1 + bakf_values['lamb'])
					bakf_values['alpha'] = 1 - (bakf_values['sigma_s'] / bakf_values['delta'])
					bakf_values['lamb'] = ((1 - bakf_values['alpha'])**2) * bakf_values['lamb'] + (bakf_values['alpha'])**2
				self.BAKF_Vs[str(post_state)] = deepcopy(bakf_values)

				update = (1 - bakf_values['alpha']) * (self.V.get(str(post_state),0)) + (bakf_values['alpha']) * (dual_value)
			# If the post-decision state value in the V matrix is currently 0 (default)
			if not self.V.get(str(post_state),0):
				# And the update is not 0, update the value
				if update != 0:
					self.V[str(post_state)] = update
			# Otherwise, if there exists a non-zero value of the post-decision state in the V matrix, update it
			else:
				self.V[str(post_state)] = update

	def update_agg_v(self, state_post_pairs, duals):
		"""
		Updating aggregated value function

		Parameters
		----------
		state_post_pairs : list[tuple]
			Pairing of post-decision states and their corresponding state values in the next time step
		duals : float
			Dual values associated with each unique state value
		"""
		for state, original_post_state in state_post_pairs:
			for level in range(self.levels):
				post_state = self.get_agg_state(copy(original_post_state),level)
				dual_value = duals[state]
				if self.stepsize_type != 'B':
					# Get the updated value
					update = (1 - self.alpha) * (self.V[level].get(str(post_state),0)) + (self.alpha) * (dual_value)
				else:
					bakf_values = self.BAKF_Vs[level].get(str(post_state),{'alpha': 0.05, 'step': 0.01, 'step_bar': 0.2, 'lamb': 25, 'beta': 0, 'delta': 0.001, 'sigma_s': 0})
					bakf_values['step'] = bakf_values['step'] / (1 + bakf_values['step'] - bakf_values['step_bar'])
					bakf_values['beta'] = (1 - bakf_values['step']) * bakf_values['beta'] + bakf_values['step'] * (dual_value - self.V[level].get(str(post_state),0))
					bakf_values['delta'] = (1 - bakf_values['step']) * bakf_values['delta'] + bakf_values['step'] * (dual_value - self.V[level].get(str(post_state),0))**2
					if self.n == 1:
						bakf_values['alpha'] = 1
						bakf_values['lamb'] = bakf_values['alpha']**2
					else:
						bakf_values['sigma_s'] = (bakf_values['delta'] - (bakf_values['beta'])**2) / (1 + bakf_values['lamb'])
						bakf_values['alpha'] = 1 - (bakf_values['sigma_s'] / bakf_values['delta'])
						bakf_values['lamb'] = ((1 - bakf_values['alpha'])**2) * bakf_values['lamb'] + (bakf_values['alpha'])**2
					self.BAKF_Vs[level][str(post_state)] = deepcopy(bakf_values)
					update = (1 - bakf_values['alpha']) * (self.V[level].get(str(post_state),0)) + (bakf_values['alpha']) * (dual_value)

				# If the post-decision state value in the V matrix is currently 0 (default)
				if self.V[level].get(str(post_state),0) == 0:
					# And the update is not 0, update the value
					if update != 0:
						self.V[level][str(post_state)] = update
				# Otherwise, if there exists a non-zero value of the post-decision state in the V matrix, update it
				else:
					self.V[level][str(post_state)] = update
				
				if self.stored_values[level].get(str(post_state),None) is None:
					self.stored_values[level][str(post_state)] = {'Value': self.V[level].get(str(post_state),0), 'Values': [self.V[level].get(str(post_state),0)]}
				else:
					self.stored_values[level][str(post_state)]['Value'] = self.V[level].get(str(post_state),0)
					self.stored_values[level][str(post_state)]['Values'].append(self.V[level].get(str(post_state),0))

	def get_agg_state(self,state,level):
		"""
		Setting attributes and aggregating agent and request types

		Parameters
		----------
		state : list
			Post-decision state
		level : int
			Level of aggregation

		Returns
		-------
		list
			Aggregated version of the post-decision state
		"""
		# Update current location
		state[0] = (self.zones[level][state[0][0]],state[0][1])
		# Update upcoming locations
		state[1] = [(self.zones[level][old_destination[0]],old_destination[1],old_destination[2]) for old_destination in state[1]]
		return state

	def update_alpha(self):
		"""
		Updating of the step-size for each step-size type
		"""
		if self.stepsize_type == 'H':
			self.alpha = max(self.lamb / (self.lamb + self.n - 1), 0.05)
		elif self.stepsize_type == 'P':
			self.alpha = 1 / self.n
		elif self.stepsize_type == 'F':
			pass
		elif self.stepsize_type == 'B':
			pass
