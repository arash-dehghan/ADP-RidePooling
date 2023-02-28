from pandas import read_csv

class Environment():
	"""
	The Environment creates framework for keeping track of time and moving agents to their next locations

	Attributes
	----------
	num_agents : int
		Number of agents in the system
	start_epoch : float
		The starting time (in seconds) for the simulation
	stop_epoch : float
		The ending time (in seconds) for the simulation
	epoch_length : float
		The interval between decision epochs (in seconds)
	car_capacity : int
		The maximum car capacity allowed per vehicle
	locs_to_visit : int
		The maximum number of passengers groups allowed at one time per vehicle
	current_time : float
		Current time in simulation (in seconds)
	travel_time : list[list]
		The travel time between each location in the area
	shortest_path : list[list]
		The next location for the desired destination
	"""
	def __init__(self, num_agents, start_epoch, stop_epoch, car_cap, locs_to_visit, data_type, name):
		self.num_agents = num_agents
		self.start_epoch = start_epoch * 3600.0
		self.stop_epoch = stop_epoch * 3600.0
		self.epoch_length = 60.0
		self.car_capacity = car_cap
		self.locs_to_visit = locs_to_visit
		self.current_time = start_epoch * 3600.0
		self.travel_time = read_csv(f'../data/{data_type}_dataset/travel_data/zone_traveltime{"" if data_type == "real" else f"_{name}"}.csv', header=None).values
		self.shortest_path = read_csv(f'../data/{data_type}_dataset/travel_data/zone_path{"" if data_type == "real" else f"_{name}"}.csv', header=None).values

	def get_travel_time(self, source, destination):
		"""
		The travel time between each location in the area

		Parameters
		----------
		source : int
			Node ID of beginning point
		destination : int
			Node ID of destination point

		Returns
		-------
		float
			The time it will take to go from source to the destination
		"""
		return self.travel_time[source, destination]

	def get_next_location(self, source, destination):
		"""
		The next location for the desired destination

		Parameters
		----------
		source : int
			Node ID of beginning point
		destination : int
			Node ID of destination point

		Returns
		-------
		float
			The next location to visit on the path from the source to the destination
		"""
		return self.shortest_path[source, destination]

	def simulate_vehicle_motion(self,agent):
		"""
		Simulate the motion of the agent

		Parameters
		----------
		agent : Agent
			Agent object to simulate motion for
		"""
		# If the vehicle has nowhere to go (is empty)
		if not len(agent.dropoff_info):
			# Keep the agent in the same location but update the state and subsequent agent state
			agent.update_state(self.current_time + self.epoch_length)
		# If the vehicle is non-empty
		else:
			# Move the agent towards the next location
			self._move_agent(agent)

	def _move_agent(self,agent):
		"""
		Move agent for the next decision epoch time

		Parameters
		----------
		agent : Agent
			Agent object to simulate motion for
		"""
		# Set the amount of time left in the current epoch to move the vehicle (Each epoch is 60 seconds)
		time_remaining = self.epoch_length
		# While there's still time left in the epoch and/or the vehicle hasn't dropped off every passenger
		while True:
			# If the time remaining in the epoch is less than the time it would take the agent to get to its next location
			if time_remaining < agent.time_to_next_location:
				# Subtract the remaining time from the time until the vehicle's next location
				agent.time_to_next_location -= time_remaining
				# Set the vehicle's next state and exit the loop
				agent.update_state(self.current_time + self.epoch_length)
				break
			# If there is at least enough time for the vehicle to get to its next location
			else:
				# Subtract the time until its next location from the time remaining in the epoch
				time_remaining -= agent.time_to_next_location
				# If the vehicle's next location is the next pick-up or drop-off location
				if agent.next_location == (agent.pickup_info[0][0] if len(agent.pickup_info) else agent.dropoff_info[0][0]):
					# If the vehicle's next location is a pick-up location
					if len(agent.pickup_info) and (agent.next_location == agent.pickup_info[0][0]):
						# Remove the pickup location (making it an empty list)
						agent.pickup_info = []
					# If the vehicle's next location is a drop-off location
					elif not len(agent.pickup_info) and (agent.next_location == agent.dropoff_info[0][0]):
						# Remove the number of passengers in the in request from the vehicle capacity
						agent.capacity -= agent.dropoff_info[0][2]
						# Remove the passenger from the vehicle altogether
						agent.dropoff_info.pop(0)
					else:
						print('SOMETHING IS WRONG')
						exit()
					# If the vehicle has no more locations it needs to visit (i.e. it's now empty)
					if not len(agent.dropoff_info):
						assert not len(agent.pickup_info)
						if agent.capacity != 0:
							print(agent)
							print(agent.capacity)
							print(agent.pickup_info)
							print(agent.dropoff_info)
							exit()
						# assert not agent.capacity
						# Set the amount of time it has until its next location to 0
						agent.time_to_next_location = 0.0
						# Set the vehicle's next state and exit the loop
						agent.update_state(self.current_time + self.epoch_length)
						break
					# If the vehicle still has more locations it needs to visit (i.e. it has more passengers)
					else:
						assert not len(agent.pickup_info)
						# Keep track of its current location/position
						current_pos = agent.next_location
						# Find and set the next location it will need to visit in order to get to the next location on its path
						agent.next_location = self.get_next_location(current_pos, agent.dropoff_info[0][0])
						# Find and set the time it will take to get to said next location
						agent.time_to_next_location = self.get_travel_time(current_pos,agent.next_location)
				# If the next location is not a pickup of drop-off point
				else:
					# Keep track of its current location/position
					current_pos = agent.next_location
					# Find and set the next location it will need to visit in order to get to the next location on its path
					next_loc = agent.pickup_info[0][0] if len(agent.pickup_info) else agent.dropoff_info[0][0]
					agent.next_location = self.get_next_location(current_pos, next_loc)
					# Find and set the time it will take to get to said next location
					agent.time_to_next_location = self.get_travel_time(current_pos,agent.next_location)

	def get_rebalancing_next_location(self,current_location, rebalancing_location):
		"""
		Get the location the agent will be after the decision epoch if they are moving towards the rebalancing_location

		Parameters
		----------
		current_location : int
			Current location of the agent
		rebalancing_location : int
			The rebalancing location the agent is moving towards

		Returns
		-------
		int
			The location agent will be at the end of the epoch
		"""
		# Current location
		cur_loc = current_location
		# Time remaining in the epoch
		time_remaining = self.epoch_length
		# While we either haven't reached the rebalancing location or there is time remaining in the epoch
		while (cur_loc != rebalancing_location) and (time_remaining > 0):
			# Next location to visit on route to the rebalancing location
			next_loc = self.get_next_location(cur_loc,rebalancing_location)
			# Time to next location to visit from current location
			time_til_next_loc = self.get_travel_time(cur_loc,next_loc)
			# Subtract the time it would take to get to that location from the amount of time remaining in the epoch
			time_remaining -= time_til_next_loc
			# If there is enough time in the epoch to reach that location, set that location as our current location
			if time_remaining > 0:
				cur_loc = next_loc
		return cur_loc









					