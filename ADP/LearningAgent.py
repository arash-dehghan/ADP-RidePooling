class LearningAgent(object):
	"""
	The LearningAgent represents each individual vehicle in the system

	Attributes
	----------
	id : int
		Unique ID of agent
	next_location : int
		Location ID of the next location agent will visit
	time_to_next_location : float
		Time (in seconds) until the agent reaches the next location
	capacity : int
		Current vehicle capacity
	car_capacity : int
		The maximum car capacity allowed per vehicle
	dropoff_info : list[(location, deadline, capacity)]
		List of tuples including each requests' drop-off location, deadline by which they must be dropped off, and their capacity
	pickup_info : list[(location, deadline, -capacity)]
		List of tuple including the requests'pickup location, deadline by which they must be picked up, and - of their capacity
	state : list[tuple(int,float), list[tuple(int, float, int)], float]
		State of the agent
	state_str : str
		String version of state of the agent
	"""
	def __init__(self, agent_id, initial_location, current_time):
		self.id = agent_id
		self.next_location = initial_location
		self.time_to_next_location = 0.0
		self.capacity = 0
		self.dropoff_info = []
		self.pickup_info = []
		self.update_state(current_time)

	def __str__(self):
		return(f'Agent {self.id} ({self.pickup_info + self.dropoff_info})')

	def __repr__(self):
		return str(self)

	def update_state(self, current_time):
		"""
		Updating the state of the agent

		Parameters
		----------
		current_time : float
			Current time in the system
		"""
		assert sum([r[2] for r in self.dropoff_info]) == self.capacity
		# Empty list of locations vehicle needs to visit and by what time it needs to be there
		locations = []
		# If it has passengers it's assigned to
		if len(self.dropoff_info) > 0:
			# If it does not need to pick anyone up (none of its upcoming locations are pick-up points)
			if len(self.pickup_info) == 0:
				# For each passenger that needs to be dropped off, include their dropoff locations, deadline to get there, and capacity
				locations = self.dropoff_info
			# If it needs to pick someone up and also drop people off
			else:
				# Get the pickup location of the request that needs picking up and the deadline to pick it up + the list of dropoff locations and the corresponding dropoff deadlines
				locations = self.pickup_info + self.dropoff_info

		# Define state
		state = [(self.next_location, self.time_to_next_location), locations, current_time]
		
		# Set string of state and the state itself
		self.state_str = str(state)
		self.state = state
