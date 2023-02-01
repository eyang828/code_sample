import os
import math
import random
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

# We implement the graph coarsening algorithm from the paper at
# https://proceedings.mlr.press/v119/fahrbach20a.html
#
# An exercise in using the networkx library

# Initialize all edge weights to 1.0
def initialize_weights(G):
	for e in G.edges():
		G[e[0]][e[1]]['weight'] = 1.0
	return G

# Keep track of which vertices have been combined. 
# Initialize initial "merged vertices" as just themselves
def initialize_vertex_lists(G):
	merged_nodes = {}
	for v in G.nodes():
		merged_nodes[v] = [v]
	return merged_nodes

# Picks a random edge incident to v proportionally to weight
def rand_edge(G, v):
	neighbors = [u for u in G.neighbors(v)]
	weights = [G[u][v]['weight'] for u in neighbors]
	return random.choices(neighbors, weights)[0]

# Contract an edge using the star idea from Fahrbach et al. 
def contract_edge(G, v_keep, v_elim, merged_nodes):
	neighbors = G.neighbors(v_elim)
	for u in neighbors:
		if u is v_keep:
			continue

		# Initialize a new edge between u and v_keep if it doesn't already exist
		if not G.has_edge(u, v_keep):
			G.add_edge(u, v_keep)
			G[u][v_keep]['weight'] = 0.0

		# Update the edge's weight using the formula from the paper
		G[u][v_keep]['weight'] += (G[u][v_elim]['weight'] * G[v_keep][v_elim]['weight']) \
			/ (G[u][v_elim]['weight'] + G[v_keep][v_elim]['weight'])

	# Add v_elim to the list of v_keep
	# temp_list = merged_nodes[v_keep]
	# tempList.extend(merged_nodes[v_elim])
	merged_nodes[v_keep].extend(merged_nodes[v_elim])

	G.remove_node(v_elim)
	del merged_nodes[v_elim]

	return G, merged_nodes

# Keep eliminating until there are only target_num_nodes left
def contract_vertices(G, target_num_nodes, merged_nodes):
	num_vertices = len(G.nodes())
	if target_num_nodes > num_vertices:
		return G 
	for i in range(num_vertices - target_num_nodes):
		v_elim = sorted(G.degree, key=lambda x: x[1], reverse = False)[0][0]
		v_keep = rand_edge(G, v_elim)
		G, merged_nodes = contract_edge(G, v_keep, v_elim, merged_nodes)
	
	return G, merged_nodes

# Return a matrix whose (i, j) entry is the effective resistance
# between vertices i and j
def get_effective_resistances(G, vertices):
	num_vertices = len(vertices)
	resistances = np.zeros((num_vertices, num_vertices))

	for i in range(num_vertices - 1):
		for j in range(i + 1, num_vertices):
			resistances[i][j] = nx.resistance_distance(G, vertices[i], vertices[j], weight = 'weight', invert_weight = True)
			resistances[j][i] = resistances[i][j]
	return resistances

if __name__ == "__main__":
	num_vertices_list = [10, 20, 30, 40]
	target_fraction_list = [0.2, 0.25, 0.3, 0.35, 0.4]
	num_trials = 5
	frobenius_dict = {}
	frobenius_plot = plt.figure()

	for num_vertices in num_vertices_list:
		print("Number of Vertices: %i" % num_vertices)
		frobenius_dict[num_vertices] = {}

		for target_fraction in target_fraction_list:
			frobenius = []
			target_num_nodes = int(target_fraction * num_vertices)

			for trial in range(num_trials):
				# Generate a random graph and initialize the edge weights
				G = nx.erdos_renyi_graph(num_vertices, 0.5)
				G = initialize_weights(G)
				merged_nodes = initialize_vertex_lists(G)
				copy = G.copy()

				# Contract down to target_num_nodes vertices
				G, merged_nodes = contract_vertices(G, target_num_nodes, merged_nodes)
				vertices = list(G.nodes())

				# Compare sketched and actual resistances
				sketchedResistances = get_effective_resistances(G, vertices)
				actualResistances = get_effective_resistances(copy, vertices)
				diff = (sketchedResistances - actualResistances) / (num_vertices * (num_vertices - 1) / 2)
				frobenius.append(np.linalg.norm(diff))

			frobenius_dict[num_vertices][target_fraction] = sum(frobenius) / (num_trials * 1.0 * target_num_nodes)

		plt.plot(list(frobenius_dict[num_vertices].keys()), list(frobenius_dict[num_vertices].values()), label = "%d Vertices" % num_vertices)
	plt.legend()
	plt.show()