from RequestOrder import RequestOrder
import numpy

class DataGenerator(object):
	"""
	Generator of request orders

	Attributes
	----------
	start_hour : int
		The starting hour we consider for the simulaiton
	end_hour : int
		The ending hour we consider for the simulation
	nodes : list
		The list of unique nodes considered in the area
	passenger_distribution : dict
		The distribution of number of passengers per request
	request_distribution : dict
		The distribution of orders seen throughout the start-end hour
	top_locations : list
		List of most popular locations in each zone for rebalancing
	unique_requests : list
		List of unique pick-up and drop-off pairs
	prev_of_requests : dict
		How prevelant each unique request is (used for distribution)
	zones : dict{dict}
		Dictionary containing which zone each node belongs in for each level of aggregation
	"""
	def __init__(self, start_hour, end_hour, nodes, passenger_distribution, request_distribution, top_locations, unique_requests, prev_of_requests, zones, real, temporal, pickups, dropoffs, relabeling, G, seed):
		self.start_hour = start_hour
		self.end_hour = end_hour
		self.nodes = nodes
		self.passenger_distribution = passenger_distribution
		self.request_distribution = request_distribution
		self.top_locations = top_locations
		self.unique_requests = unique_requests
		self.prev_of_requests = prev_of_requests
		self.zones = zones
		self.data = real
		self.temporal = temporal
		self.pickups = pickups
		self.dropoffs = dropoffs
		self.relabeling = relabeling
		self.G = G
		self.seed = seed
		self.np = numpy.random.RandomState(seed)

	def get_requests(self, time):
		"""
		Getting the requests at the inputted time

		Parameters
		----------
		time : float
			Time to get requests for (in seconds)

		Returns
		-------
		list[Request]
			List of requests sampled at the given time
		"""
		# Get number of requests to sample at time
		number_of_requests = self.get_number_requests(time)
		# Get which pickup-dropoff pairs to use
		location_pairs = self.get_location_pairs(number_of_requests, time)
		# Get number of passengers for each order
		passengers_added = self.add_passengers(location_pairs)
		# Create the request object
		requests = self.create_requests(time,passengers_added)
		return requests

	def get_number_requests(self,time):
		"""
		Number of requests at the given time

		Parameters
		----------
		time : float
			Time to get requests for (in seconds)

		Returns
		-------
		int
			Number of requests at the time
		"""
		if self.data:
			requests, probs = zip(*self.request_distribution[time].items())
			return int(self.np.choice(requests,p=probs))
		else:
			avg = self.request_distribution[time]
			num_requests = -1
			while num_requests < 0:
				num_requests = int(self.np.normal(loc=avg, scale=1))
			return num_requests

	def get_location_pairs(self,n, time):
		"""
		Getting the requests at the inputted time

		Parameters
		----------
		n : int
			Number of request pairs to get
		time : float
			Time of the requests

		Returns
		-------
		list
			List of pickup-dropoff pairs
		"""
		if self.data and self.temporal:
			locs = list(self.prev_of_requests[time].keys())
			loc_pair_indices = self.np.choice(range(len(locs)),n,p=list(self.prev_of_requests[time].values()))
			return [locs[index] for index in loc_pair_indices]
		else:
			locs = list(self.prev_of_requests.keys())
			loc_pair_indices = self.np.choice(range(len(locs)),n,p=list(self.prev_of_requests.values()))
			return [locs[index] for index in loc_pair_indices]

	def add_passengers(self,location_pairs):
		"""
		Getting the number of passengers in each request

		Parameters
		----------
		location_pairs : list
			List of pickup-dropoff pairs

		Returns
		-------
		list
			2D list of potential requests with number of passengers in request added
		"""
		passenger_counts = self.np.choice(list(self.passenger_distribution.keys()),len(location_pairs),p=list(self.passenger_distribution.values()))
		return [[requests[0],requests[1],passengers,requests[2]] for passengers,requests in zip(passenger_counts,location_pairs)]

	def create_requests(self,t,rs):
		"""
		Create request object

		Parameters
		----------
		time : float
			Time to get requests for (in seconds)
		rs : list
			2D list of potential requests with number of passengers in request added

		Returns
		-------
		list[Request]
			List of requests sampled at the given time
		"""
		return [RequestOrder(r[0],r[1],t,r[3],r[2]) for r in rs]

	def create_test_scenarios(self,num_days):
		"""
		Creating test scenarios for a set number of days for the given start to end times

		Parameters
		----------
		num_days : int
			Number of days' worth of request order data to create

		Returns
		-------
		dict
			Dictionary of test scenarios for the given number of days over the set time horizon
		"""
		test_scenarios = {day: {} for day in range(num_days)}
		for day in range(num_days):
			for time in self.request_distribution.keys():
				test_scenarios[day][time] = self.get_requests(time)
		return test_scenarios

	def create_locations(self, num_locations):
		"""
		Create list of random starting locations

		Parameters
		----------
		num_locations : int
			Number of random locations to create 

		Returns
		-------
		list
			List of location IDs
		"""
		return [self.np.choice(self.nodes) for _ in range(num_locations)]
			





