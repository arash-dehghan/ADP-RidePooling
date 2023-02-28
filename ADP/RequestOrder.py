class RequestOrder(object):
	"""
	The RequestOrder represents the order

	Attributes
	----------
	origin_time : float
		The origin time of the order in the system (in seconds)
	pickup : int
		Location ID of the pick-up location
	dropoff : int
		Location ID of the drop-off location
	value : int
		Number of passengers in the request
	original_travel_time : float
		Amount of time it takes to go from the pick-up to drop-off location (in seconds)
	state : list[int, int, int, float]
		State representation of the order
	state_str : str(list[int, int, int, float])
		String representation of the state of the order
	"""
	def __init__(self, source, destination, current_time, travel_time, value=1):
		self.origin_time = current_time
		self.pickup = source
		self.dropoff = destination
		self.value = value
		self.original_travel_time = travel_time
		self.state = [self.pickup, self.dropoff, self.value, self.origin_time]
		self.state_str = str(self.state)

	def __str__(self):
		return(f'{self.pickup}->{self.dropoff} ({self.original_travel_time})')

	def __repr__(self):
		return str(self)