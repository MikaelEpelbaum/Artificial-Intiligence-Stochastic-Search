import networkx as nx
import copy
import numpy as np
from itertools import combinations, permutations, product
import networkx as nx

ids = ["111111111", "222222222"]


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.initial_improved = copy.deepcopy(initial)
        self.map = np.array(initial["map"])
        self.initial_improved['number_taxis'] = len(initial['taxis'].keys())
        self.initial_improved['number_passengers'] = len(initial['passengers'].keys())
        taxis = self.initial_improved["taxis"]
        # original taxi data, passenger in taxi, current fuel
        self.initial_improved['taxis'] = {taxi: (taxis.get(taxi), [], taxis.get(taxi)['fuel']) for taxi in taxis.keys()}
        passengers = initial["passengers"]
        # boolean False to indicate passenger wasn't picked yet
        passengers = {passenger: (passengers.get(passenger), False) for passenger in passengers.keys()}
        initial_state = {'taxis': self.initial_improved['taxis'], 'passengers': passengers, 'remains': len(passengers)}

        all_states = Moves.actions(self, initial_state)
        self.optimal_actions = self.value_iteration_algorithm(all_states)


    def act(self, state):
        raise NotImplemented

    def value_iteration_algorithm(self, all_states):
        # V0 set values (distribute values at beginning)
        for state in all_states:
            taxis_locs = [taxi['location'] for taxi in state['taxis']]
            passengers_loc_at_dest = [passenger['location'] for passenger in state['passengers'] if passenger['location'] == passenger['destination']]
        return 0




class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class Moves:
    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        states = []
        n, m = np.shape(self.map)

        taxis_possibilities = taxisComputation(self)
        passengers_possibilities = passengersComputation(state['passengers'])
        return list(product(taxis_possibilities, passengers_possibilities))


def flatten(l):
    return [item for sublist in l for item in sublist]


def max_fuel_on_min_drivable_manhattan_distance_from_gas_and_departure(pre, map, taxis):
    # n_nodes = sum(x != 'I' for x in [item for sublist in map for item in sublist])
    n, m = np.shape(map)
    n_nodes = n*m
    flat_map = flatten(map)
    imp_indx = np.argwhere(np.array(flat_map) == 'F') +1
    gas_indx = flatten(np.argwhere(np.array(flat_map) == 'G') + 1)
    original_loc = taxis[pre[0][0]]['location']
    original_loc = m * original_loc[0] + original_loc[1]+1
    gas_indx.append(original_loc)

    G = nx.Graph()
    G.add_nodes_from([i+1 for i in range(n_nodes)])
    for i in range(1, n+1):
        for j in range(1, m+1):
            #not the first row
            if i != 1:
                if (i-1)*n + j not in imp_indx and (i-2)*n + j not in imp_indx:
                    G.add_edge((i-1)*n + j, (i-2)*n + j)
            #not first column
            if j != 1:
                if (i-1)*m + j not in imp_indx and (i-1)*m + (j-1) not in imp_indx:
                    G.add_edge((i-1)*m + j, (i-1)*m + (j-1))

    cur_loc = m * pre[1][0][0] + pre[1][0][1]+1
    min_dist_cur_loc_gas = min(nx.shortest_path_length(G, source=cur_loc, target=g) for g in gas_indx)
    return taxis[pre[0][0]]['fuel'] - min_dist_cur_loc_gas



# all the options there are on the grid for the taxi to be and the passengers
def taxisComputation(input):
    map = input.map
    n, m = np.shape(map)
    # all possible locations
    locations = []
    for i in range(n):
        for j in range(m):
            if map[i][j] != 'I':
                locations.append((i, j))

    # all taxis combinations
    taxis = list(state['taxis'].keys())
    taxis_combinations = combination(taxis)

    # all taxis locations cross combinations
    taxis_cross_locations_combinations = []
    locations_combinations = [list(combinations(locations, i)) for i in range(1, 1+len(taxis))]
    for taxis_comb in taxis_combinations:
        for permutation in permutations(taxis_comb):
            taxis_cross_locations_combinations.append([permutation, locations_combinations[len(permutation)-1]])

    # taxis without fuel and capacities

    pre_state = []
    for elem in taxis_cross_locations_combinations:
        # elem[0] are the taxis, elem[1] are the locations to assign to those taxis
        for i in range(len(elem[1])):
            pre_state.append([elem[0], elem[1][i]])


    # every taxis fuel possibilities
    taxis_states = []
    for pre in pre_state:
        fuels = []
        capacities = []
        # je doit metre des fuel possible uniquementet pas toutes les options en fonction de la distance minimal de la station d'escence
        # ou de la case de depart

        highest_fuel_possible_at_location = max_fuel_on_min_drivable_manhattan_distance_from_gas_and_departure(pre, map, input.initial['taxis'])
        fuels = [i for i in range(highest_fuel_possible_at_location + 1)]
        # for i in range(len(pre[0])):
        #     fuels.append([i for i in range(state['taxis'][pre[0][i]]['fuel']+1)])

        # for i in range(len(pre[0])):
        #     fuels.append([i for i in range(highest_fuel_possible_at_location + 1)])
        # for i in range(len(pre[0])):
        #     capacities.append([i for i in range(state['taxis'][pre[0][i]]['capacity']+1)])
        #
        # fuel_capacity_combinations = []
        # for i in range(len(pre[0])):
        #     fuel_capacity_combinations.append(list(product(fuels[i], capacities[i])))

        # one taxi
        # if len(taxis) == 1:
        #     for comb in fuel_capacity_combinations[0]:
        # #       states avec les passager
        #         passengers = list(input.initial['passengers'].keys())
        #         # passengers combinations
        #         for i in range(comb[1]):
        #             c = list(combinations(passengers, i))[0]
        #             taxis_states.append({'taxis': [{pre[0][0]: {'location': pre[1][0], 'fuel': comb[0], 'capacity': comb[1]}}, [c]]})

        # one taxi
        if len(taxis) == 1:
            for fuel in fuels:
                cap = input.initial['taxis'][pre[0][0]]['capacity']
                #states avec les passager
                taxis_states.append({'taxis': [{pre[0][0]: {'location': pre[1][0], 'fuel': fuel, 'capacity': cap}}, []]})
                passengers = list(input.initial['passengers'].keys())
                for i in range(min(cap, len(passengers))):
                    c = list(list(combinations(passengers, i+1))[0])
                    taxis_states.append({'taxis': [{pre[0][0]: {'location': pre[1][0], 'fuel': fuel, 'capacity': cap}}, c]})


        # todo: TEMPORARY, CHANGE WHEN FINISHED WITH ONE TAXI
        # # two taxis
        # if len(taxis) == 2 and len(fuel_capacity_combinations) == 2:
        #     comb = list(product(fuel_capacity_combinations[0], fuel_capacity_combinations[1]))
        #     for c in comb:
        #         taxis_states.append({'taxis': {pre[0][0]: {'location': pre[1][0], 'fuel': c[0][0], 'capacity': c[0][1]},
        #                                        pre[0][1]: {'location': pre[1][1], 'fuel': c[1][0], 'capacity': c[1][1]}}})
    return taxis_states


def passengersComputation(passengers):
    result = []
    for passenger in passengers:
        vector =[]
        location = [passengers[passenger][0]['location']]
        destination = [passengers[passenger][0]['destination']]
        possible_goals = [g for g in passengers[passenger][0]['possible_goals']]
        prob_change_goal = passengers[passenger][0]['prob_change_goal']
        locs = location + destination + possible_goals
        locs = set(locs)
        combinations = list(product(locs, locs))
        for comb in combinations:
            vector.append({'passengers': {passenger: {'location': comb[0], 'destination': comb[1], 'possible_goals': tuple(possible_goals), 'prob_change_goal': prob_change_goal}}})
        result.append(vector)
    prod = []
    if len(result) == 1:
        prod = result[0]
    else:
        prod = list(product(result[0], result[1], result[2], result[3]))
    return prod




def combination(sample_list):
    list_combinations = list()
    for n in range(1, len(sample_list) + 1):
        list_combinations += list(combinations(sample_list, n))
    return list_combinations


class Distances:
    def __init__(self, initial):
        self.state = initial
        self.graph = self.build_graph(initial)
        self.shortest_path_distances = self.create_shortest_path_distances(self.graph)
        self.diameter = nx.diameter(self.graph)
    def build_graph(self, initial):
        """
        build the graph of the problem
        """
        n, m = len(initial['map']), len(initial['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if initial['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def create_shortest_path_distances(self, graph):
        d = {}
        for n1 in graph.nodes:
            for n2 in graph.nodes:
                if n1 == n2:
                    continue
                d[(n1, n2)] = len(nx.shortest_path(graph, n1, n2)) - 1
        return d

    def check_distances(self, graph, node1, node2):
        return graph[(node1,node2)]


if __name__ == "__main__":
    state = {
        "optimal": True,
        "map": [["P", "P", "P"], ["P", "G", "P"], ["P", "P", "P"]],
        "taxis": {"taxi 1": {"location": (0, 0), "fuel": 10, "capacity": 2}},
        "passengers": {
            "Dana": {
                "location": (2, 2),
                "destination": (0, 0),
                "possible_goals": ((0, 0), (2, 2)),
                "prob_change_goal": 0.1,
            }
        },
        "turns to go": 100,
    }
    # state = {
    #     'optimal': False,
    #     "turns to go": 100,
    #     'map': [['P', 'P', 'P', 'P', 'P'],
    #             ['P', 'I', 'P', 'G', 'P'],
    #             ['P', 'P', 'I', 'P', 'P'],
    #             ['P', 'P', 'P', 'I', 'P']],
    #     'taxis': {'taxi 1': {'location': (2, 0), 'fuel': 5, 'capacity': 2},
    #               'taxi 2': {'location': (0, 1), 'fuel': 6, 'capacity': 2}},
    #     'passengers': {'Iris': {'location': (0, 0), 'destination': (1, 4),
    #                             'possible_goals': ((1, 4),), 'prob_change_goal': 0.2},
    #                    'Daniel': {'location': (3, 1), 'destination': (2, 1),
    #                               'possible_goals': ((2, 1), (0, 1), (3, 1)), 'prob_change_goal': 0.2},
    #                    'Freyja': {'location': (2, 3), 'destination': (2, 4),
    #                               'possible_goals': ((2, 4), (3, 0), (3, 2)), 'prob_change_goal': 0.2},
    #                    'Tamar': {'location': (3, 0), 'destination': (3, 2),
    #                              'possible_goals': ((3, 2),), 'prob_change_goal': 0.2}},
    # }
    agent = OptimalTaxiAgent(state)