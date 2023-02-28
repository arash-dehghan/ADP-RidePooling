import sys
sys.dont_write_bytecode = True
import argparse
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import ast
from Environment import Environment
from CentralAgent import CentralAgent
from ValueFunction import ADP
from ResultCollector import ResultCollector
from LearningAgent import LearningAgent

def run_epoch(envt, central_agent, value_function, requests, request_generator, agents_predefined, is_training=True):
	envt.current_time = envt.start_epoch
	ts = int((envt.stop_epoch - envt.start_epoch) / envt.epoch_length)
	agents = deepcopy(agents_predefined)
	vehicle_caps, vehicle_groups, requests_seen, requests_handled, empty_vehicles, non_empty_vehicles_groups, feasible_acts, total_agents_rebalanced = ([] for _ in range(8))

	for t in range(ts):
		# Load current times requests
		current_requests = request_generator.get_requests(envt.current_time) if is_training else requests[envt.current_time]

		# Set request deadlines
		central_agent.set_request_deadlines(current_requests)
		
		# Set all of the driver and passengers states and attributes
		feasible_actions_data = central_agent.get_attributes(agents, current_requests, envt.current_time, is_training)

		# Get the actions of each agent
		dual_values, matchings = central_agent.choose_actions(value_function)

		# Update v_bar values if training
		if is_training and t != 0:
			state_post_pairs = list(set([(agent.state_str,agent.post_state_str) for agent in agents]))
			state_post_pairs = [(state,ast.literal_eval(post_state))for state,post_state in state_post_pairs]
			value_function.update_v(state_post_pairs, dual_values) if value_function.levels == 1 else value_function.update_agg_v(state_post_pairs, dual_values)

		# Applying the matchings provided by the LP/ILP
		requests_served, agents_rebalanced = central_agent.set_new_paths(agents,matchings)

		if not is_training:
			requests_seen.append(len(current_requests))
			requests_handled.append(requests_served)
			vehicle_caps.append(np.average([agent.capacity for agent in agents]))
			vehicle_groups.append(np.average([len(agent.dropoff_info) for agent in agents]))
			empty_vehicles.append(sum([1 for agent in agents if not agent.capacity]))
			non_empty = [len(agent.dropoff_info) for agent in agents if len(agent.dropoff_info) > 0]
			non_empty_vehicles_groups.append(0 if not len(non_empty) else np.average(non_empty))
			feasible_acts.append(feasible_actions_data)
			total_agents_rebalanced.append(agents_rebalanced)

		# Make sure capacities aren't exceeded after matchings
		for agent in agents:
			assert agent.capacity == sum([cap[2] for cap in agent.dropoff_info])
			assert len(agent.pickup_info) <= envt.locs_to_visit
			assert agent.capacity <= envt.car_capacity

		# Simulate the movement of vehicles on the road
		for agent in agents:
			nearby_agents = central_agent.nearby_agents[agent.state_str]
			envt.simulate_vehicle_motion(agent)
			agent.post_state = agent.state + [nearby_agents, central_agent.current_total_requests]
			agent.post_state_str = str(agent.state + [nearby_agents, central_agent.current_total_requests])

		# Move up in time
		envt.current_time += envt.epoch_length

	if is_training:
		value_function.n += 1
		value_function.update_alpha()
		
	return [np.array(total_agents_rebalanced), np.array(feasible_acts), np.array(non_empty_vehicles_groups), np.array(empty_vehicles), np.array(vehicle_caps), np.array(vehicle_groups), np.array(requests_seen), np.array(requests_handled)]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--numagents', type=int, default=20)
	parser.add_argument('-wt', '--waittime', type=float, default=90.0)
	parser.add_argument('-dt', '--delaytime', type=float, default=90.0)
	parser.add_argument('-vehicle_cap', '--capacity', type=int, default=6)
	parser.add_argument('-locs_to_visit', '--locs_to_visit', type=int, default=3)
	parser.add_argument('-start', '--start', type=int, default=11)
	parser.add_argument('-end', '--end', type=int, default=12)
	parser.add_argument('-train_days', '--train_days', type=int, default=1000)
	parser.add_argument('-test_days', '--test_days', type=int, default=20)
	parser.add_argument('-test_every', '--test_every', type=int, default=100)
	parser.add_argument('-stepsize', '--stepsize', type=str, choices=['H','P','F','B'], default='B')
	parser.add_argument('-levels', '--levels', type=int, default = 4)
	parser.add_argument('-rebalancing', '--rebalancing', type=int, choices=[0,1], default=1)
	parser.add_argument('-reward_type', '--reward_type', type=str, choices=['basic','delay'], default='basic')
	parser.add_argument('-data_type', '--data_type', type=str, choices=['real','synthetic'], default='real')
	parser.add_argument('-generation_file', '--generation_file', type=str, default='100_150_1_5')
	args = parser.parse_args()

	request_generator = pickle.load(open(f'../data/generations/{args.data_type}_{args.generation_file}/data_{args.generation_file}.pickle','rb'))
	envt = Environment(args.numagents, args.start, args.end, args.capacity, args.locs_to_visit, args.data_type, args.generation_file)
	central_agent = CentralAgent(envt, args.numagents, args.waittime, args.delaytime, args.levels, request_generator.zones, args.rebalancing, args.reward_type, request_generator.top_locations)
	value_function = ADP(args.stepsize, args.levels, request_generator.zones)

	test_location_data = [request_generator.create_locations(40)[:args.numagents] for _ in range(args.test_days)]
	test_data = request_generator.create_test_scenarios(args.test_days)

	result_collector = ResultCollector()
	stops = [i for i in range(args.test_every,args.train_days + args.test_every,args.test_every)]
	final_results = []
	i = 0

	# Initial myopic results
	for test_day in tqdm(range(args.test_days)):
		initial_states = test_location_data[test_day]
		requests = test_data[test_day]
		agents = [LearningAgent(agent_id, initial_state, envt.start_epoch) for agent_id, initial_state in enumerate(initial_states)]
		results = run_epoch(envt, central_agent, value_function, requests, request_generator, agents, False)
		final_results.append(results)
	result_collector.update_results(i, final_results)

	# Train the model
	for train_day in tqdm(range(args.train_days)):
		initial_states = request_generator.create_locations(args.numagents)
		agents = [LearningAgent(agent_idx, initial_state, envt.start_epoch) for agent_idx, initial_state in enumerate(initial_states)]
		run_epoch(envt, central_agent, value_function, None, request_generator, agents_predefined=agents, is_training=True)
		i += 1
		if i in stops:
			final_results = []
			# Get the test results
			for test_day in range(args.test_days):
				initial_states = test_location_data[test_day]
				requests = test_data[test_day]
				agents = [LearningAgent(agent_idx, initial_state,envt.start_epoch) for agent_idx, initial_state in enumerate(initial_states)]
				results = run_epoch(envt, central_agent, value_function, requests, None, agents_predefined=agents, is_training=False)
				final_results.append(results)
			result_collector.update_results(i, final_results)

	result_collector.V = value_function.V
	result_collector.stored_values = value_function.stored_values

	with open(f'../Results/ADP/results_{args.generation_file}_{args.data_type}_{args.numagents}_{args.waittime}_{args.delaytime}_{args.capacity}_{args.locs_to_visit}_{args.rebalancing}_{args.stepsize}_{args.levels}.pickle', 'wb') as handle:
			pickle.dump(result_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)

