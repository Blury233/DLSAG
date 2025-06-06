import torch
import networkx as nx
import heapq
import itertools
import math
import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.gnn_models import CenterScorerGNN
from utils.dataset import extract_center_score_features
from utils.graph import nx_dijkstra, create_nx_graph

def load_center_scorer_model(model_path, num_node_features, device, args):
    """Loads the trained CenterScorerGNN model."""
    try:
        model = CenterScorerGNN(
            num_node_features=num_node_features,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_gin_layers,
            gnn_type=args.gnn_type,
            out_channels=1
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"CenterScorerGNN model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading CenterScorerGNN model: {e}")
        return None

def predict_center_scores(model, nx_graph, terminals_set, term_list, dist_from_term, device, args):
    """Uses the GNN to predict center potential scores for all nodes."""
    model.eval()
    
    features_result = extract_center_score_features(
        nx_graph, terminals_set, term_list, dist_from_term, args.num_nearest_terminals
    )
    if features_result[0] is None:
        print("Error: Failed to extract features for GNN prediction.")
        return None
    
    x, edge_index, edge_attr, _, max_node_id = features_result
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    with torch.no_grad():
        data = pyg_data.to(device)
        logits = model(data)
        return logits.cpu().numpy().flatten()

def _prune_tree(tree, terminals):
    """Iteratively prunes non-terminal leaf nodes from a tree."""
    T_set = set(terminals)
    pruned = True
    while pruned:
        pruned = False
        leaves_to_remove = [n for n in tree.nodes() if tree.degree(n) == 1 and n not in T_set]
        if leaves_to_remove:
            tree.remove_nodes_from(leaves_to_remove)
            pruned = True
    # Remove isolated non-terminals that might result from pruning
    isolated_to_remove = [n for n in tree.nodes() if tree.degree(n) == 0 and n not in T_set]
    if isolated_to_remove:
        tree.remove_nodes_from(isolated_to_remove)
    return tree

def solve(edge_list, terminals, model, device, args):
    """
    The main Robins-Zelikovsky solver accelerated by a GNN.
    """
    # 1. Initialization
    if not terminals or len(terminals) < 2: return [], 0.0
    
    nx_graph, T_list = create_nx_graph(edge_list, terminals)
    T_set = set(T_list)
    if len(T_list) < 2: return [], 0.0
    
    all_nodes = list(nx_graph.nodes()) + T_list
    if not all_nodes: return [], 0.0
    max_node_id = max(all_nodes)
    
    # 2. Pre-computation: Dijkstra from all terminals
    term_to_idx = {t: i for i, t in enumerate(T_list)}
    dist_from_each_term = []
    parents_from_each_term = []
    for t_source in T_list:
        dist_t, parent_t = nx_dijkstra(nx_graph, t_source, max_node_id)
        dist_from_each_term.append(dist_t)
        parents_from_each_term.append(parent_t)

    # 3. Metric Closure MST
    metric_closure_graph = nx.Graph()
    for i, u in enumerate(T_list):
        for j in range(i + 1, len(T_list)):
            v = T_list[j]
            dist_uv = dist_from_each_term[term_to_idx[u]][v]
            if not math.isinf(dist_uv):
                metric_closure_graph.add_edge(u, v, weight=dist_uv)
    
    mst_edges = nx.minimum_spanning_edges(metric_closure_graph, weight='weight', data=True)
    
    # 4. Initial Steiner Tree Construction
    steiner_tree = nx.Graph()
    for u, v, _ in mst_edges:
        parent_list = parents_from_each_term[term_to_idx[u]]
        curr = v
        while curr != u and curr != -1:
            prev = parent_list[curr]
            if prev == -1: break
            w = nx_graph.edges[prev, curr]['weight']
            steiner_tree.add_edge(prev, curr, weight=w)
            curr = prev

    # 5. GNN-Accelerated Greedy Improvement
    iteration = 0
    max_iterations = nx_graph.number_of_nodes()
    
    while iteration < max_iterations:
        iteration += 1
        
        # A. Predict center scores with GNN
        node_scores = predict_center_scores(model, nx_graph, T_set, T_list, dist_from_each_term, device, args)
        if node_scores is None: break

        # B. Select Top-k' candidate centers
        candidate_centers_scored = [(score, node) for node, score in enumerate(node_scores)
                                    if node not in T_set and nx_graph.has_node(node)]
        top_k_centers = [node for _, node in heapq.nlargest(args.k_prime, candidate_centers_scored)]
        
        best_gain = -1.0
        best_swap = None

        # C. Local search for each candidate
        for s in top_k_centers:
            dists_s_to_t = sorted([(dist_from_each_term[term_to_idx[t]][s], t) for t in T_list])
            m_nearest_terminals = [t for _, t in dists_s_to_t[:args.m_nearest]]
            
            if len(m_nearest_terminals) < 3: continue

            for a, b, c in itertools.combinations(m_nearest_terminals, 3):
                try:
                    comp_cost = dist_from_each_term[term_to_idx[a]][s] + \
                                dist_from_each_term[term_to_idx[b]][s] + \
                                dist_from_each_term[term_to_idx[c]][s]
                    if math.isinf(comp_cost): continue

                    path_ab = nx.shortest_path(steiner_tree, source=a, target=b, weight='weight')
                    path_ac = nx.shortest_path(steiner_tree, source=a, target=c, weight='weight')
                    path_bc = nx.shortest_path(steiner_tree, source=b, target=c, weight='weight')
                    
                    cycle_edges = {}
                    for path in [path_ab, path_ac, path_bc]:
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            edge_key = tuple(sorted((u,v)))
                            cycle_edges[edge_key] = steiner_tree.edges[u,v]['weight']

                    if len(cycle_edges) < 2: continue
                    
                    heavy_edges = heapq.nlargest(2, cycle_edges.items(), key=lambda item: item[1])
                    saved_cost = sum(w for _, w in heavy_edges)
                    gain = saved_cost - comp_cost

                    if gain > best_gain:
                        best_gain = gain
                        best_swap = {
                            'center': s, 'terms': (a, b, c),
                            'remove_edges': [e for e, _ in heavy_edges]
                        }
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        # D. Apply best swap if gain is positive
        if best_gain > args.epsilon:
            s = best_swap['center']
            steiner_tree.remove_edges_from(best_swap['remove_edges'])
            for term in best_swap['terms']:
                parent_list = parents_from_each_term[term_to_idx[term]]
                curr = s
                while curr != term and curr != -1:
                    prev = parent_list[curr]
                    if prev == -1: break
                    w = nx_graph.edges[prev, curr]['weight']
                    steiner_tree.add_edge(prev, curr, weight=w)
                    curr = prev
        else:
            break # No more improvements found
    
    # 6. Finalization
    final_tree = _prune_tree(steiner_tree, T_list)
    final_edges = [(u, v, data['weight']) for u, v, data in final_tree.edges(data=True)]
    final_cost = sum(w for _, _, w in final_edges)
    
    return final_edges, final_cost