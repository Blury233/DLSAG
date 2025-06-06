import networkx as nx
import random

def create_nx_graph(edge_list, terminals=None):
    """
    Creates a NetworkX graph from an edge list and terminals.
    """
    nx_g = nx.Graph()
    if edge_list:
        for u, v, w in edge_list:
            if all(isinstance(i, int) and i >= 0 for i in [u, v]) and isinstance(w, (int, float)) and w >= 0:
                nx_g.add_edge(u, v, weight=w)

    valid_terminals = set()
    if terminals:
        for t in terminals:
            if isinstance(t, int) and t >= 0:
                valid_terminals.add(t)
                if not nx_g.has_node(t):
                    nx_g.add_node(t)
    
    return nx_g, sorted(list(valid_terminals))

def nx_dijkstra(nx_graph, src_node, max_node_id, weight_label='weight'):
    """
    Performs Dijkstra's algorithm and returns results in fixed-size lists.
    """
    num_nodes = max_node_id + 1
    dist_list = [float('inf')] * num_nodes
    parent_list = [-1] * num_nodes

    if not nx_graph.has_node(src_node):
        return dist_list, parent_list

    try:
        pred_dict, dist_dict = nx.dijkstra_predecessor_and_distance(nx_graph, src_node, weight=weight_label)
        
        for node_id, d in dist_dict.items():
            if 0 <= node_id < num_nodes:
                dist_list[node_id] = d
        for node_id, predecessors in pred_dict.items():
            if predecessors and 0 <= node_id < num_nodes:
                parent_list[node_id] = predecessors[0]

    except nx.NodeNotFound:
        if 0 <= src_node < num_nodes:
            dist_list[src_node] = 0.0
    
    return dist_list, parent_list

def generate_random_nx_graph(num_nodes, edge_prob, weight_range=(1, 10)):
    """
    Generates a random weighted Erdos-Renyi graph.
    """
    if num_nodes <= 0: return nx.Graph()
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = round(random.uniform(*weight_range), 2)
    return G

def select_terminals_from_graph(nx_g, num_terminals):
    """
    Selects terminals randomly from the nodes of a graph.
    """
    nodes = list(nx_g.nodes())
    if not nodes or num_terminals <= 0: return []
    actual_num_terminals = min(num_terminals, len(nodes))
    return random.sample(nodes, actual_num_terminals)