import numpy as np

class ResultCollector(object):
	"""
	Collect simulation results used for analyzing and plotting later

	Attributes
	----------
	empty_vehicles : dict
		Dicionary of the number of empty vehicles at every time-step
	capacity_results : dict
		Dicionary of the average capacity at every time-step
	locations_to_visit_results : dict
		Dicionary of the average number of locations agents have to visit at every time-step
	requests_seen_day_results : dict
		Dicionary of the average number of requests seen at every time-step
	requests_served_day_results : dict
		Dicionary of the average number of requests served at every time-step
	requests_seen : dict
		Dicionary of the total requests seen each day
	requests_served : dict
		Dicionary of the total requests served each day
	non_empty_vehicles_groups : dict
		Dicionary of the average number of empty vehicles at every time-step
	feasible_actions : dict
		Dicionary of the average number of feasible actions available to each available agent at every time-step
	num_rebalanced_agents : dict
		Dicionary of the average number of vehicles rebalanced at every time-step
	"""
	def __init__(self):
		self.empty_vehicles = {}
		self.capacity_results = {}
		self.locations_to_visit_results = {}
		self.requests_seen_day_results = {}
		self.requests_served_day_results = {}
		self.requests_seen = {}
		self.requests_served = {}
		self.non_empty_vehicles_groups = {}
		self.feasible_actions = {}
		self.num_rebalanced_agents = {}

	def update_results(self, iteration, results):
		"""
		Update the results of the original dictionaries after running another set of sample paths (n's)

		Parameters
		----------
		iteration : int
			The iteration number (n)
		results : list[list]
			2D list of relevent results
		"""
		rebalanced_agents = np.sum([days_result[0] for days_result in results],0) / len(results)
		feasible_acts = np.sum([days_result[1] for days_result in results],0) / len(results)
		non_empty_cars_locations = np.sum([days_result[2] for days_result in results],0) / len(results)
		empty_cars = np.sum([days_result[3] for days_result in results],0) / len(results)
		capacities = np.sum([days_result[4] for days_result in results],0) / len(results)
		locations = np.sum([days_result[5] for days_result in results],0) / len(results)
		seen = np.sum([days_result[6] for days_result in results],0) / len(results)
		handled = np.sum([days_result[7] for days_result in results],0) / len(results)

		self.num_rebalanced_agents[iteration] = rebalanced_agents
		self.feasible_actions[iteration] = feasible_acts
		self.non_empty_vehicles_groups[iteration] = non_empty_cars_locations
		self.empty_vehicles[iteration] = empty_cars
		self.capacity_results[iteration] = capacities
		self.locations_to_visit_results[iteration] = locations
		self.requests_seen_day_results[iteration] = seen 
		self.requests_served_day_results[iteration] = handled
		self.requests_seen[iteration] = sum(np.sum([days_result[6] for days_result in results],0))
		self.requests_served[iteration] = sum(np.sum([days_result[7] for days_result in results],0))