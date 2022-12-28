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
        self.initial_improved['remains'] = len(initial['passengers'].keys())
        taxis = self.initial_improved["taxis"]
        # original taxi data, passenger in taxi, current fuel
        # self.initial_improved['taxis'] = {taxi: (taxis.get(taxi), [], taxis.get(taxi)['fuel']) for taxi in taxis.keys()}
        self.initial_improved['taxis'] = {taxi: (taxis.get(taxi), []) for taxi in taxis.keys()}
        passengers = initial["passengers"]
        # boolean False to indicate passenger wasn't picked yet
        passengers = {passenger: (passengers.get(passenger), False) for passenger in passengers.keys()}
        initial_state = {'taxis': self.initial_improved['taxis'], 'passengers': passengers, 'remains': len(passengers)}

        all_states = Moves.states(self, initial_state)
        self.optimal_actions = self.value_iteration_algorithm(all_states)


    def act(self, state):
        raise NotImplemented

    def value_iteration_algorithm(self, v0):
        return value_iteration(self.initial_improved, v0)




class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class Moves:
    def states(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        return statesComputation(self)


    def actions(state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        if state["remains"] == 0:
            return
        n, m = np.shape(state['map'])
        steps = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        taxis_dict = state["taxis"]
        taxis_actions = {taxi: [] for taxi in taxis_dict.keys()}
        passengers = state["passengers"]
        passengers_to_collect = [passenger for passenger in list(passengers.keys()) if passengers[passenger]['location'] != passengers[passenger]['destination']]
        refuel = []
        global_actions = {}
        impossibles = tuple(zip(*np.where(state['map'] == 'I')))
        for taxi, taxi_details in taxis_dict.items():
            curr_pos = np.asarray(taxi_details[0]["location"])
            # curr_fuel = taxi_details[2]
            curr_occupancy = len(taxi_details[1])
            all_moves = [curr_pos + np.asarray(step) for step in steps]
            # all possible direction movement listing
            for move in all_moves:
                if 0 <= move[0] < n and 0 <= move[1] < m and tuple(move) not in impossibles:
                    taxis_actions[taxi].append(("move", taxi, tuple(move)))
            # passegers that can be piked listing
            # todo:verifier que les type son comparable
            passengers_to_pick = [passenger for passenger in passengers_to_collect if
                                  tuple(passengers[passenger]["location"]) == tuple(curr_pos)]
            for pas in passengers_to_pick:
                taxis_actions[taxi].append(("pick up", taxi, pas))
            # does the taxi has passengers?
            if len(taxis_dict[taxi][1]) > 0:
                to_drop_off = [passenger for passenger, loc_pos in passengers.items() if
                               loc_pos[0]["destination"] == loc_pos[0]["location"] and passenger in taxis_dict[taxi][1]]
                for pas in to_drop_off:
                    taxis_actions[taxi].append(("drop off", taxi, pas))
            if state['map'][[taxis_dict[taxi][0]["location"]][0][0]][[taxis_dict[taxi][0]["location"]][0][1]] == 'G':
                refuel = [("refuel", taxi)]

            global_actions[taxi] = [taxis_actions[taxi] + [("wait", taxi)] + refuel]

        all_moves = [global_actions[act][0] for act in global_actions.keys()]
        cartesian = [element for element in product(*all_moves)]

        # todo: remove impossible moves if two taxis go to the same tile
        cart_copy = cartesian.copy()
        if len(state['taxis']) == 1:
            for cart in cartesian:
                if len(cart) > 1:
                    moves = []
                    for ac in cart:
                        if ac[0] == 'move':
                            moves.append(ac[2])
                        if ac[0] == 'wait' or ac[0] == 'pick up' or ac[0] == 'refuel' or ac[0] == 'drop off':
                            moves.append(tuple(state['taxis'][ac[1]][0]['location']))
                    moves_length = len(moves)
                    moves_reduced = len(set(moves))
                    if moves_length > moves_reduced:
                        cart_copy.remove(cart)
            cartesian = cart_copy
        return cartesian


def value_iteration(state, v0):
    if state['turns to go'] == 0:
        return
    state['turns to go']-=1
    possible_actions = Moves.actions(state)
    chain = []
    for action_set in possible_actions:
        # chose state representing action
        for action in action_set:
            l = len(action)
            new_state = copy.deepcopy(state)
            new_state['turns to go'] -= 1
            if l > 2:
                if action[0] == 'move':
                    new_state = copy.deepcopy(state)
                    new_state['taxis'][action[1]][0]['fuel'] -= 1
                    new_state['taxis'][action[1]][0]['location'] = action[2]
                    for pas in new_state['taxis'][action[1]][1]:
                        new_state['passengers'][pas]['location'] = action[2]
                    for a in v0:
                        if a['taxis'][action[1]][0]['location'] == new_state['taxis'][action[1]][0]['location']:
                            if a['taxis'][action[1]][0]['fuel'] == new_state['taxis'][action[1]][0]['fuel']:
                                for pas in list(new_state['passengers'].keys()):
                                    if new_state['passengers'][pas]['location'] == a['passengers'][pas]['location']:
                                        temp = copy.deepcopy(new_state)
                                        temp['taxis'][action[1]] = a['taxis'][action[1]]
                                        temp['passengers'] = copy.deepcopy(a['passengers'])
                                        temp['reward'] = a['reward']
                                        chain.append(temp)

                if action[0] == 'pick up':

                    for taxi in a['taxis']:
                        for tax in list(taxi.keys()):
                            if tax == action[1] and new_state['taxis'][tax][0]['capacity'] > len(new_state['taxis'][tax][1]):
                                new_state['taxis'][tax][0][1] = new_state['taxis'][tax][0][1] + [action[2]]
                                chain.append(new_state)
                if action[0] == 'drop off':
                    new_state = copy.deepcopy(state)
            #         reward = 100
            if l > 1:
                if action[0] == 'refuel':
                    print('')
                if action[0] == 'wait':
                    chain.append(copy.deepcopy(state))
#                     RAJOUTER LE REWARD
#     calculate shits now and use recursion principle
    chain = listDictsRemoveDuplicates(chain)
    reward = []
    for c in chain:

        print('')

#     v(T)


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

def get_reward(taxis, passengers):
    #todo: think if I should implement the rest of the grading here or even if this grading is right here
    count = []
    for passenger in passengers:
        for taxi in list(taxis.keys()):
            if passengers[passenger]['location'] == passengers[passenger]['destination'] and passenger not in taxis[taxi][1]:
                count.append(100)
    return sum(count)


# all the options there are on the grid for the taxi to be and the passengers
def statesComputation(input):
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
    # taxis_states = []
    states = []
    for pre in pre_state:
        fuels = []
        capacities = []
        # je doit metre des fuel possible uniquementet pas toutes les options en fonction de la distance minimal de la station d'escence
        # ou de la case de depart

        highest_fuel_possible_at_location = max_fuel_on_min_drivable_manhattan_distance_from_gas_and_departure(pre, map, input.initial['taxis'])
        fuels = [i for i in range(highest_fuel_possible_at_location + 1)]
        # one taxi
        if len(taxis) == 1:
            for fuel in fuels:
                cap = input.initial['taxis'][pre[0][0]]['capacity']
                #states avec les passager
                # taxis_states.append({'taxis': [{pre[0][0]: {'location': pre[1][0], 'fuel': fuel, 'capacity': cap}}, []]})
                passengers = list(input.initial['passengers'].keys())
                for i in range(min(cap, len(passengers))+1):
                    c = list(combinations(passengers, i))
                    for clients in c:
                        #passengers data
                        vector = []
                        for client in clients:
                            # client is in taxi
                            destination = input.initial['passengers'][client]['destination']
                            possible_goals = [g for g in input.initial['passengers'][client]['possible_goals']]
                            prob_change_goal = input.initial['passengers'][client]['prob_change_goal']
                            locs = set(list([destination]) + possible_goals)
                            for possible_dest in locs:
                                for possible_loc in locations:
                                    vector.append({
                                        client: {'location': possible_loc, 'destination': possible_dest, 'possible_goals': tuple(possible_goals),
                                                    'prob_change_goal': prob_change_goal}})

                        for pas in vector:
                            # if t['taxi 1'][0]['location'] == (0, 1):
                            #     if t['taxi 1'][0]['fuel'] == 9:
                            #         if t['taxi 1'][0]['capacity'] == 2:
                            if pas['Dana']['location'] == (2, 2):
                                if pas['Dana']['destination'] == (2, 2):
                                    if pas['Dana']['possible_goals'] == ((0, 0), (2, 2)):
                                        if pas['Dana']['prob_change_goal'] == 0.1:
                                            print()

                            t = {pre[0][0]: ({'location': pre[1][0], 'fuel': fuel, 'capacity': cap}, list(clients))}
                            reward = get_reward(t, pas)
                            states.append({'taxis': t, 'passengers': pas, 'reward': reward})

                            # when passenger not in taxi
                            t = {pre[0][0]: ({'location': pre[1][0], 'fuel': fuel, 'capacity': cap}, [])}
                            reward = get_reward(t, pas)
                            states.append({'taxis': t, 'passengers': pas, 'reward': reward})





        # todo: TEMPORARY, CHANGE WHEN FINISHED WITH ONE TAXI
        # # two taxis
        # if len(taxis) == 2 and len(fuel_capacity_combinations) == 2:
        #     comb = list(product(fuel_capacity_combinations[0], fuel_capacity_combinations[1]))
        #     for c in comb:
        #         taxis_states.append({'taxis': {pre[0][0]: {'location': pre[1][0], 'fuel': c[0][0], 'capacity': c[0][1]},
        #                                        pre[0][1]: {'location': pre[1][1], 'fuel': c[1][0], 'capacity': c[1][1]}}})
    return states

# todo: faire que cette fonction soit incorporer dans la precedente des taxis en faisant que l'emplacement des passager qui
# sont dans le taxi soit correct. et rajouter toutes les autres option d'emplacement des passeger qui sont pas dans le taxi
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


def listDictsRemoveDuplicates(l):
    seen = []
    indx = []
    cnt = 0
    for d in l:
        t = str(tuple(d.items()))
        if t not in seen:
            seen.append(t)
            indx.append(cnt)
        cnt+=1
    return [l[index] for index in indx]

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