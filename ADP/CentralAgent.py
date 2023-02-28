from collections import Counter
import itertools 
import docplex.mp.model as cpx
import cplex
from copy import deepcopy, copy
import numpy as np

class CentralAgent(object):
	"""
	The Central Agent which decides on feasability of matches and matching agents and requests

	Attributes
	----------
	envt : Environment
		Environment of the system
	gamma : float
		Discounting rate/factor
	num_agents : int
		Number of agents in the system
	acceptable_wait_time : float
		Number of seconds agent has to pickup passenger(s)
	acceptable_delay_time : float
		Number of seconds agent has on top of wait time and trip length to drop off passenger(s)
	rebalancing : bool
		Whether to do rebalancing of agents or not
	reward_type : str
		Type of reward type to consider
	levels : int
		Number of aggregation levels
	zones : dict{dict}
		For each aggregation level, tells us which zone each node belongs to
	rebalancing_locations : list[int]
		Rebalancing locations to consider across the map
	"""
	def __init__(self, envt, num_agents, acceptable_wait_time, acceptable_delay_time, levels, zones, rebalancing, reward_type, rebalancing_locations):
		self.envt = envt
		self.gamma = 0.9
		self.num_agents = num_agents
		self.acceptable_wait_time = acceptable_wait_time
		self.acceptable_delay_time = acceptable_delay_time
		self.rebalancing = rebalancing
		self.reward_type = reward_type
		self.levels = levels
		self.zones = zones
		self.rebalancing_locations = rebalancing_locations

	def set_request_deadlines(self, requests):
		"""
		Setting pickup and dropoff deadline for each request

		Parameters
		----------
		requests : list[Request]
			List of Request objects to set deadlines to
		"""
		# For each request set the time by which they need to be picked up and dropped off
		for request in requests:
			request.pickup_deadline = request.origin_time + self.acceptable_wait_time
			request.dropoff_deadline = request.origin_time + self.acceptable_wait_time + request.original_travel_time + self.acceptable_delay_time

	def get_attributes(self, agents, requests, current_time, is_training):
		"""
		Setting attributes and aggregating agent and request types

		Parameters
		----------
		agents : list[Agent]
			List of Agent objects to aggregate
		requests : list[Request]
			List of Request objects to aggregate
		current_time : float
			Current time in the system
		is_training : bool
			Whether the current iteration is a training iteration or not

		Returns
		-------
		float
			The average number of feasible actions for each agent feasible to be paired with requests
		"""
		# Set total number of requests in current time-step
		self.current_total_requests = len(requests)

		# Create a dictionary with each agent state string as its key and the agent object as the value
		self.agent_dict = {agent.state_str: agent for agent in agents}

		# Create a dictionary with each request state string as its key and the request object as the value
		self.passenger_dict = {request.state_str: request for request in requests}

		# Get a list of all the unique string agent states
		self.A = [state for state in self.agent_dict.keys()]

		# Get the number of each type of agent state
		self.R = dict(Counter([agent.state_str for agent in agents]))

		# Get a list of all the unique string passenger state names
		self.B = [passenger for passenger in self.passenger_dict.keys()]

		# Get the number of each type of passenger state
		self.D = dict(Counter([request.state_str for request in requests]))

		# Set information about which agent and request state types can be matched to one another
		self._get_available_matches()

		# Get number of feasible actions per agent
		feasible_actions_data = self._get_feasible_actions_data() if (not is_training) else None

		# Get all possible available agents in the next time-step
		self._set_nearby_agents()

		# If rebalncing is set to True, add rebalancing actions for eligible agents 
		if self.rebalancing:
			self._set_default_actions()

		return feasible_actions_data

	def _get_available_matches(self):
		"""
		Check feasibility of matching each agent and request type based on the agent and request constraints
		"""
		# Get all possible pairings of vehicle types and request types
		possible_pairings = list(itertools.product(self.agent_dict.values(), self.passenger_dict.values()))

		# Create empty dictionary for pairable matches (contains ordering and delay of matching each feasible agent and request type)
		self.pairable_matchings = {}
		# Create dictionary of feasible actions for each agent state type (None as in continue on its current path)
		self.agent_feasible_actions = {agent_state : [None] for agent_state in self.A}

		for agent,request in possible_pairings:
			# Check if agent is able to pick up a new request (if number of passengers it has to pickup is 0)
			if len(agent.pickup_info) == 0:
				# Check if pairing fulfills number of locations to visit constraint
				if self._check_locations_constraint(agent):
					# Check if pairing fulfills capacity constraints
					if self._check_capacity_constraint(agent,request):
						# Check if pairing fulfills wait time constraints
						if self._check_waiting_constraint(agent,request):
							# Check if pairing fulfills delay time constraints
							delay_output = self._check_delay_constraint(agent,request)
							# If there exists a feasible pairing between the agent and request type
							if len(delay_output[0]) > 0:
								# Add pairing to the dictionary of feasible pairings/matchings
								self.pairable_matchings[(agent.state_str,request.state_str)] = delay_output
								self.agent_feasible_actions[agent.state_str].append(request.state_str)

	def _check_locations_constraint(self,agent):
		"""
		Check if adding the new set of passengers would exceed the maximum number of groups constraint

		Parameters
		----------
		agent : Agent
			Agent object to consider

		Returns
		-------
		bool
			Whether agent passes number of passenger groups/locations constraint
		"""
		return True if ((len(agent.dropoff_info) + 1) <= self.envt.locs_to_visit) else False

	def _check_capacity_constraint(self,agent,request):
		"""
		Check if adding the new set of passengers would exceed vehicle capacity constraint

		Parameters
		----------
		agent : Agent
			Agent object to consider
		request : Request
			Request object to consider
		
		Returns
		-------
		bool
			Whether agent-request pair passes the capacity constraint
		"""
		return True if ((agent.capacity + request.value) <= self.envt.car_capacity) else False
		
	def _check_waiting_constraint(self,agent,request):
		"""
		Check if agent would be able to pickup new set of passengers wihtin wait time constraints

		Parameters
		----------
		agent : Agent
			Agent object to consider
		request : Request
			Request object to consider
		
		Returns
		-------
		bool
			Whether agent-request pair passes the waiting time constraint
		"""
		return True if (self._get_wait_time(agent,request) <= self.acceptable_wait_time) else False

	def _get_wait_time(self,agent,request):
		"""
		Return the amount of time it will take for the agent from its current position to reach the passengers' pickup location

		Parameters
		----------
		agent : Agent
			Agent object to consider
		request : Request
			Request object to consider
		
		Returns
		-------
		float
			Time it would take for agent to reach requests' pickup location
		"""
		return agent.time_to_next_location + self.envt.get_travel_time(agent.next_location,request.pickup)

	def _check_delay_constraint(self,agent,request):
		"""
		Check if agent can be inserted into the agents' dropoff schedule without making any of the requests agent has late

		Parameters
		----------
		agent : Agent
			Agent object to consider
		request : Request
			Request object to consider
		Returns
		-------
		tuple(list,float)
			Tuple of (i) a list containing the best ordering in which to put the requests the agent is assigned (including this new potential request)
					 (ii) a value of the delay incurred from this potential ordering
		"""
		# Add the passenger to the passengers already in the agents' vehicle
		possible_passengers = agent.dropoff_info + [(request.dropoff, request.dropoff_deadline, request.value)]
		# Get all possible drop-off orderings for these passengers
		orderings = list(itertools.permutations(possible_passengers,len(possible_passengers)))
		# Mark time at which vehicle would be after picking up new passenger
		time_after_pickup = self.envt.current_time + self._get_wait_time(agent,request)
		# Create dummy best_ordering
		best_ordering = ([],float('-inf'))

		# For each drop-off ordering
		for ordering in orderings:
			# Pre-set that the ordering is possible
			possible_ordering = True
			# Set the current time to the time at which the passenger was picked up
			cur_time = time_after_pickup
			# Set the current location to the location at which agent would be after picking up the passenger
			cur_location = request.pickup
			# Set total drop-off deadline time saved to 0
			total_time_ahead_schedule = 0
			# For each passenger in the given drop-off ordering
			for passenger in ordering:
				# Get what the current time would be when the passenger is actually dropped off
				cur_time += self.envt.get_travel_time(cur_location,passenger[0])
				# Get what the location would be once this passenger is dropped off
				cur_location = passenger[0]
				# Set the time ahead of schedule for the dropoff
				time_ahead_schedule = passenger[1] - cur_time
				# Check if the passenger is dropped off before its deadline, if not break
				if time_ahead_schedule < 0:
					possible_ordering = False
					break
				# If dropoff is valid, add the time ahead of schedule to total time ahead
				total_time_ahead_schedule += time_ahead_schedule
			# If ordering is valid, update best ordering if current order is better than what we have
			if possible_ordering and (total_time_ahead_schedule > best_ordering[1]):
				best_ordering = (list(ordering), total_time_ahead_schedule)
		return best_ordering

	def _get_feasible_actions_data(self):
		"""
		Get data on the average number of feasible actions each available agent has (just for statistics reasons)

		Returns
		-------
		float
			The average number of feasible actions for available agents
		"""
		# Empty list for feasible actions
		feasible_acts = []
		# For each unique agent and the number of feasible actions that are available to them
		for agent_id, acts in self.agent_feasible_actions.items():
			# Get the actual agent
			agent  = self.agent_dict[agent_id]
			# Check that they're actually available, and if so, add the number of feasible actions they have for as many as there are of that type to the feasible acts list
			if (agent.capacity < self.envt.car_capacity) and (len(agent.dropoff_info) < self.envt.locs_to_visit) and (not len(agent.pickup_info)):
				for _ in range(self.R[agent_id]):
					feasible_acts.append(len(acts) - 1)
		return np.average(feasible_acts)

	def _set_nearby_agents(self):
		"""
		Set the number of nearby agents each unique agent state type has near them in the pre-decision state
		"""
		# Empty nearby agents dictionary that takes key as agent state and value as number of nearby agents
		self.nearby_agents = {}
		# For every unique agent state type
		for agent in self.agent_dict.values():
			# Pre-set number of nearby agents to 0
			nearby_agents = 0
			# For every agent state type
			for other_agent in self.agent_dict.values():
				# Check the time it would take going to and/or from the two locations
				going_distance = self.envt.get_travel_time(agent.next_location,other_agent.next_location)
				coming_distance = self.envt.get_travel_time(other_agent.next_location,agent.next_location)
				# If either of them are within the acceptable wait time, then they are a nearby competitor and we add them to the nearby agents
				if (going_distance <= self.acceptable_wait_time) or (coming_distance <= self.acceptable_wait_time):
					nearby_agents += self.R[other_agent.state_str]
			# Assume one of the agents accounted for is the agent themselves, so do -1 to nearby agents and add to nearby dict
			self.nearby_agents[agent.state_str] = nearby_agents - 1

	def _set_default_actions(self):
		"""
		Set the default rebalancing actions for the agent types which are empty 
		"""
		# For every unique agent state and their feasible actions
		for agent_state,feasible_actions in  self.agent_feasible_actions.items():
			# Grab the agent
			agent = self.agent_dict[agent_state]
			# Check if they are empty, if so, add rebalancing actions to their feasible actions
			if not len(agent.dropoff_info):
				self.agent_feasible_actions[agent_state] += self.rebalancing_locations

	def choose_actions(self, value_function):
		"""
		Choose which agents and requests to match together

		Parameters
		----------
		value_function : ValueFunction
			ADP based value function

		Returns
		-------
		dict{str : int}
			Dictionary of unique state type as key and dual value for state type as the value
		list[tuple((str, str), int)]
			List containing tuples which are (i) a tuple with the matching agent and request type and (ii) the number of these to match together
		"""
		# Solve the LP/ILP model
		flow_driver_conservation_const, model, variables, solution = self._solve_model(value_function)

		### Get Dual Values ###
		driver_cons = [const for const in flow_driver_conservation_const.values()]
		driver_duals  = model.dual_values(driver_cons)
		driver_dual_values = {const: dual_val for const,dual_val in zip(flow_driver_conservation_const.keys(),driver_duals)}

		# Ensure that the solution is integer
		for variable in variables:
			assert solution.get_value(variable) == int(solution.get_value(variable))

		# Get the matchings of vehicle types to the request types
		matchings = [((variable.split('_')[1],variable.split('_')[2]),int(solution.get_value(variable))) for variable in variables if solution.get_value(variable) > 0]

		return driver_dual_values, matchings

	def _solve_model(self,value_function):
		"""
		Solve the model for max R + gamma * V

		Parameters
		----------
		value_function : ValueFunction
			ADP based value function

		Returns
		-------
		dict{str : docplex.const}
			Dictionary of unique state type for drivers and the corresponding constraint for it

		docplex.Model
			The model at hand

		list[str]
			List of string decision variable names
		
		docplex.solution
			The solution to the model
		"""
		#############
		### MODEL ###
		model = cpx.Model(name="ADP Matching Model")

		### SETUP ###
		p_values = [None] + self.rebalancing_locations if self.rebalancing else [None]

		### VARIABLES ###
		variables = [f'x_{a}_{b}' for a in self.agent_feasible_actions.keys() for b in self.agent_feasible_actions[a]]
		x_a_b = {(a,b): model.continuous_var(name=f'x_{a}_{b}') for a in self.agent_feasible_actions.keys() for b in self.agent_feasible_actions[a] if (b is not None) and (type(b) != int)}
		x_a_p = {(a,b): model.continuous_var(name=f'x_{a}_{b}') for a in self.agent_feasible_actions.keys() for b in self.agent_feasible_actions[a] if (b is None) or (type(b) == int)}
		x_a_d = {**x_a_b,**x_a_p}

		### CONSTRAINTS ###
		flow_driver_conservation_const = {a: model.add_constraint( ct=( model.sum(x_a_b.get((a,b),0) for b in self.B) + model.sum(x_a_p.get((a,p),0) for p in p_values)  == self.R[a]), ctname=f'constraint_a_{a}') for a in self.A}
		flow_passenger_conservation_const = {b: model.add_constraint( ct=( model.sum(x_a_b.get((a,b),0) for a in self.A) <= self.D[b]), ctname=f'constraint_b_{b}') for b in self.B}

		### OBJECTIVE ###
		direct_reward = model.sum(x_a_b[(a,b)] * self._get_reward(delay) for (a,b),(_,delay) in self.pairable_matchings.items())
		downstream_reward = model.sum(self._get_V_value(action,value_function) * x_ad for action,x_ad in x_a_d.items())
		model.set_objective('max', direct_reward + self.gamma * downstream_reward)

		### SOLVE ###
		solution = model.solve()
		assert solution
		#############

		return flow_driver_conservation_const, model, variables, solution

	def _get_reward(self, reward):
		"""
		If reward set to basic then every request served is equal value, otherwise, it is worth how much delay time is saved from all deadlines

		Parameters
		----------
		reward : float
			The amount of delay incurred from the matching

		Returns
		-------
		float
			The immediate reward value
		"""
		return 1 if self.reward_type == 'basic' else reward

	def _get_V_value(self, action, value_function):
		"""
		Getting the downstream value of having a particular agent take a given action

		Parameters
		----------
		action : str / None / int
			The action to take for the agent

		value_function : ValueFunction
			ADP based value function

		Returns
		-------
		float
			The downstream reward
		"""
		# Create copy of the agent object
		agent = deepcopy(self.agent_dict[action[0]])
		# Get number of nearby agents
		nearby_agents = self.nearby_agents[agent.state_str]
		# If the action is to fulfill a request
		if isinstance(action[1],str):
			# Get the request object
			request = deepcopy(self.passenger_dict[action[1]])
			# Place the passenger in the request into the agents' vehicle
			self._place_passenger_in_vehicle(agent,request)
			# Simulate forward in time to the next time-step
			self.envt.simulate_vehicle_motion(agent)
		# If the action is to continue on the path the agent is currently on
		elif isinstance(action[1],type(None)):
			# Simulate forward in time to the next time-step
			self.envt.simulate_vehicle_motion(agent)
		# If the action is to rebalance an empty vehicle to a given location
		elif isinstance(action[1],int):
			# Relocate the agent and move forward in time
			self._relocate_agent(agent,action[1])
		else:
			print('Something is terribly wrong')
			print(action[1])
			print(type(action[1]))
			exit()

		# Set the agents' post-decision state
		agent.post_state = agent.state + [nearby_agents, self.current_total_requests]
		agent.post_state_str = str(agent.state + [nearby_agents, self.current_total_requests])

		# Get value of post-decision state
		V = value_function.V.get(agent.post_state_str,0) if self.levels == 1 else self.get_aggregated_V(agent,value_function)

		return V

	def get_aggregated_V(self,agent,value_function):
		"""
		Get the aggregated V value for the combined aggregation levels

		Parameters
		----------
		agent : LearningAgent
			The vehicle

		value_function : ValueFunction
			ADP based value function

		Returns
		-------
		float
			The aggregated downstream reward for taking a given a given action and being in a given post-decision state
		"""
		# Create empty distionary to store level data
		V_dicts = {}
		# For each level of aggregation
		for level in range(self.levels):
			# Get the post-decision state relating to that level of aggregation
			post_state = self.get_agg_state(copy(agent.post_state),level)
			# Set the key and value of that level to the data relating to that post-state value
			V_dicts[level] = value_function.stored_values[level].get(str(post_state),{})
		# Get the associated weights for each level of aggregation based on the data
		weights = self.get_aggregated_weights(V_dicts)
		# Sum up and return the aggregated V value
		V = sum([value.get('Value',0)*weight for value, weight in zip(V_dicts.values(),weights)])
		return V

	def get_agg_state(self,state,level):
		"""
		Get the aggregated post-decision state for the given level

		Parameters
		----------
		state : list
			The post-decision state of the agent at hand

		level : int
			The level of aggregation

		Returns
		-------
		list
			The post-decision state at the level of aggregation
		"""
		# Update current location
		state[0] = (self.zones[level][state[0][0]],state[0][1])
		# Update upcoming locations
		state[1] = [(self.zones[level][old_destination[0]],old_destination[1],old_destination[2]) for old_destination in state[1]]
		return state

	def get_aggregated_weights(self,V_dicts):
		"""
		Get the aggregated post-decision state for the given level

		Parameters
		----------
		V_dicts : dict
			Dictionary of data for each given level of aggregation for the post-decision state

		Returns
		-------
		list
			Normalized values for each level
		"""
		# Empty list for weights associated with each level of aggregation
		weights = []
		# For each level and corresponding set of data for that level	
		for level, V in V_dicts.items():
			# If there doesn't exist any prior data, use a weight of 0
			if not len(V):
				weights.append(0)
			# Otherwise, get the weight
			else:
				denominator = ((np.var(V['Values']) / len(V['Values'])) + (V['Value'] - V_dicts[0].get('Value',0))**2)
				weight = 1 / denominator if denominator != 0 else  1e-6
				weights.append(weight)
		return self._normalize_weights(weights)

	def _normalize_weights(self, weights):
		"""
		Normalize the weights of the levels

		Parameters
		----------
		weights : list
			List of weights for each level of aggregation

		Returns
		-------
		list
			Normalized values for each level
		"""
		return [1/len(weights) for _ in range(len(weights))] if not sum(weights) else [weight / sum(weights) for weight in weights]

	def _place_passenger_in_vehicle(self,agent,passenger):
		"""
		Placing the passenger inside the agents' vehicle

		Parameters
		----------
		agent : LearningAgent
			The vehicle

		passenger : RequestOrder
			The passenger to place in the vehicle
		"""
		# Get the ordering of the passengers to drop-off
		ordering,_ = deepcopy(self.pairable_matchings[agent.state_str,passenger.state_str])
		# Update the passenger ordering in the agent object
		agent.dropoff_info = ordering
		# Update the agents' path
		agent.pickup_info = [(passenger.pickup, passenger.pickup_deadline, -passenger.value)]
		# Update the vehicle capacity now that the request has been added to the vehicle
		agent.capacity += passenger.value

	def _relocate_agent(self,agent,location):
		"""
		Move the agent to the rebalancing location

		Parameters
		----------
		agent : LearningAgent
			The vehicle

		location : int
			The location for the agent to move towards when rebalancing
		"""
		assert not agent.time_to_next_location
		# Get the location it would be at the next epoch time if it moved towards the rebalancing location
		next_location = self.envt.get_rebalancing_next_location(agent.next_location, location)
		# Set this new location as its current location
		agent.next_location = next_location
		# Update the state and time for the agent
		agent.update_state(self.envt.current_time + self.envt.epoch_length)
				
	def set_new_paths(self, agents, matchings):
		"""
		Set the new path for each of the agents to be on for the next decision epoch

		Parameters
		----------
		agents : list[LearningAgent]
			List of the agents in the system

		matchings : list[tuple]
			The matchings between the agent types to the request types, rebalancing locations, and null actions

		Returns
		-------
		int
			The total number of requests that have been matched in this current time-step
		int
			The total number of agents which were rebalanced
		"""
		# Initialize an empty list for the agents updated and the total requests served to 0
		agents_updated, total_requests_matched, total_agents_rebalanced = [], 0, 0

		# For each matching of agent type and request type
		for match in matchings:
			# Check (in a very dumb way, I agree) whether the matching is to an actual request, a None, or a rebalancing location
			match_type = self._get_match_type(match[0][1][0])
			# If the agent type is matched to a request
			if match_type == list:
				# Get the request
				passenger = deepcopy(self.passenger_dict[match[0][1]])
				# Get all of the agents who match the agent state and who have not already been matched
				matching_agents = [agent for agent in agents if (agent.state_str == match[0][0]) and (agent.id not in agents_updated)]
				# Get how many of this agent type you need to match to the given request type
				number_to_match = match[1]
				# For the number of agent types to match to the request type
				for i in range(number_to_match):
					# Get the agent
					agent = matching_agents[i]
					# Note that they have been matched
					agents_updated.append(agent.id)
					# Match the agent type with the passenger type and place the passenger into the vehicle
					self._place_passenger_in_vehicle(agent,passenger)
					# Update the total number of requests served
					total_requests_matched += 1
			# If the agent type is matched to a rebalancing location
			elif match_type == int:
				# Get the integer value of the location
				location = int(match[0][1])
				# Get all of the agents who match the agent state and who have not already been matched
				matching_agents = [agent for agent in agents if (agent.state_str == match[0][0]) and (agent.id not in agents_updated)]
				# Get how many of this agent type you need to match to the given rebalancing location
				number_to_match = match[1]
				# For the number of agent types to rebalance to said location
				for i in range(number_to_match):
					# Get the agent
					agent = matching_agents[i]
					# Note that they have been matched
					agents_updated.append(agent.id)
					# Get the location they would be in the next epoch time if they were rebalanced to the rebalancing location and update their current location
					agent.next_location = self.envt.get_rebalancing_next_location(agent.next_location, location)
					# Update the total number of agents rebalanced
					total_agents_rebalanced += 1
		return total_requests_matched, total_agents_rebalanced

	def _get_match_type(self,character):
		"""
		Dumb way to check whehter the matching is a request, a None, or a rebalancing location

		Parameters
		----------
		character : str
			Used to check whether we have a request, rebalancing, or null action

		Returns
		-------
		int / list / None
			Signify what time of action it is
		"""
		if character == '[':
			return list  
		elif character == 'N':
			return None
		else:
			return int




		