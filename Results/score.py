import sys
sys.dont_write_bytecode = True
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def get_all_filenames(methodology):
	files = os.listdir(f'{methodology}/{directory}/')
	return [file for file in files if os.path.splitext(file)[1] == '.pickle']

def get_learned_results(files, methodology):
	all_seen, all_served = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		seen, served = results.requests_seen[0], list(results.requests_served.values())
		all_seen.append(seen)
		all_served.append(served)
	return [np.average(all_seen) for _ in range(len(all_seen))], np.mean(all_served, axis=0)

def plot_progress(methodology):
	files = get_all_filenames(methodology)
	seen, served = get_learned_results(files, methodology)
	print(f'The myopic version satisfied {served[0]} requests (out of {seen[0]}) and the trained after n={50 if methodology == "NeurADP" else 1000} iterations satisfies {served[-1]}')
	print(f'That is an increase of {round(((served[-1] - served[0]) / seen[0]) * 100,4)}%')
	print(served)
	plt.plot([i*(10 if methodology == 'NeurADP' else 100) for i in range(len(served))], served)
	plt.xlabel('Iterations (n)')
	plt.ylabel('Requests Served')
	plt.title(f'{methodology} Requests Served Over Training : {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.show()

def get_day_results(files, methodology):
	originals, updateds, seens = [], [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.requests_served_day_results.keys())[-1]
		original = results.requests_served_day_results[0]
		updated = results.requests_served_day_results[last_index]
		seen = results.requests_seen_day_results[0]
		originals.append(original)
		updateds.append(updated)
		seens.append(seen)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)
	seen = np.mean(seens, axis=0)

	return original, updated, seen

def plot_day_results_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated, seen = get_day_results(files, methodology)
		plt.plot(updated, label = methodology)
	plt.plot(original, label = 'Myopic')
	plt.plot(seen, label = 'Requests Seen')
	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Number of Requests')
	plt.title(f'Number of Requests Served Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()

def get_rebalanced_agents(files, methodology):
	originals, updateds = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.num_rebalanced_agents.keys())[-1]
		original = results.num_rebalanced_agents[0]
		updated = results.num_rebalanced_agents[last_index]
		originals.append(original)
		updateds.append(updated)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)

	return original, updated


def plot_rebalanced_agents_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated = get_rebalanced_agents(files, methodology)
		plt.plot(original, label = f'{methodology} Original')
		plt.plot(updated, label = f'{methodology} Updated')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Number of Agents Rebalanced')
	plt.title(f'Number of Agents Rebalanced Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()

def get_feasible_acts(files, methodology):
	originals, updateds = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.feasible_actions.keys())[-1]
		original = results.feasible_actions[0]
		updated = results.feasible_actions[last_index]
		originals.append(original)
		updateds.append(updated)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)

	return original, updated

def plot_feasible_acts_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated = get_feasible_acts(files, methodology)
		plt.plot(original, label = f'{methodology} Original')
		plt.plot(updated, label = f'{methodology} Updated')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Average Number of Feasible Actions Per Vehicle')
	plt.title(f'Average Number of Feasible Actions Per Vehicle Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()

def get_empty_vehicles(files, methodology):
	originals, updateds = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.empty_vehicles.keys())[-1]
		original = results.empty_vehicles[0]
		updated = results.empty_vehicles[last_index]
		originals.append(original)
		updateds.append(updated)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)

	return original, updated

def plot_empty_vehicles_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated = get_empty_vehicles(files, methodology)
		plt.plot(original, label = f'{methodology} Original')
		plt.plot(updated, label = f'{methodology} Updated')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Average Number of Empty Vehicles')
	plt.title(f'Average Number of Empty Vehicles Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()

def get_non_empty_vehicle_groups(files, methodology):
	originals, updateds = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.non_empty_vehicles_groups.keys())[-1]
		original = results.non_empty_vehicles_groups[0]
		updated = results.non_empty_vehicles_groups[last_index]
		originals.append(original)
		updateds.append(updated)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)

	return original, updated

def plot_non_empty_vehicle_groups_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated = get_non_empty_vehicle_groups(files, methodology)
		plt.plot(original, label = f'{methodology} Original')
		plt.plot(updated, label = f'{methodology} Updated')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Average Groups of Passengers for Non-Empty Vehicles')
	plt.title(f'Average Groups of Passengers for Non-Empty Vehicles Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()

def get_capacities(files, methodology):
	originals, updateds = [], []
	for file in files:
		with open(f'{methodology}/{directory}/{file}', 'rb') as handle: results = pickle.load(handle)
		last_index = list(results.capacity_results.keys())[-1]
		original = results.capacity_results[0]
		updated = results.capacity_results[last_index]
		originals.append(original)
		updateds.append(updated)

	original = np.mean(originals, axis=0)
	updated = np.mean(updateds, axis=0)

	return original, updated

def plot_capacity_compared(kind = 'both'):
	methods = ['NeurADP', 'ADP'] if kind == 'both' else [kind]
	for methodology in methods:
		files = get_all_filenames(methodology)
		original, updated = get_capacities(files, methodology)
		plt.plot(original, label = f'{methodology} Original')
		plt.plot(updated, label = f'{methodology} Updated')

	plt.xlabel('Time (in Minutes)')
	plt.ylabel('Average Vehicle Capacity')
	plt.title(f'Average Vehicle Capacity Throughout Day: {directory.split("/")[-2]} = {directory.split("/")[-1]}')
	plt.legend()
	plt.show()


method = 'both'
results = 'ADP'
directory = 'Real/Delay/60'

plot_progress(results)
# plot_day_results_compared()
# plot_non_empty_vehicle_groups_compared()
# plot_empty_vehicles_compared()
# plot_rebalanced_agents_compared()
# plot_feasible_acts_compared()
# plot_capacity_compared()










