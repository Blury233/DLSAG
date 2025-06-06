import torch
import numpy as np
import math
import itertools
from torch_geometric.data import Data, InMemoryDataset

def extract_center_score_features(nx_graph, terminals_set, term_list, dist_from_term, num_nearest_terminals):
    """
    Extracts node features for the CenterScorerGNN.
    Features: [norm_degree, is_terminal, norm_dist_to_k_nearest_terminals...]
    """
    all_nodes = list(nx_graph.nodes())
    if not all_nodes and not terminals_set:
        return None, None, None, None, -1

    max_node_id = max(all_nodes + list(terminals_set)) if all_nodes or terminals_set else -1
    num_nodes = max_node_id + 1
    
    num_features = 1 + 1 + num_nearest_terminals
    x = np.zeros((num_nodes, num_features), dtype=np.float32)
    
    # Feature 1: Normalized Degree
    # degrees = np.array([nx_graph.degree(n) for n in range(num_nodes)], dtype=np.float32)
    degrees = np.array([nx_graph.degree(n) if nx_graph.has_node(n) else 0 for n in range(num_nodes)],dtype=np.float32
    )
    max_degree = degrees.max()
    x[:, 0] = degrees / (max_degree if max_degree > 0 else 1.0)
    
    # Feature 2: Is Terminal
    for t in terminals_set:
        if t < num_nodes: x[t, 1] = 1.0

    # Feature 3..N: Normalized distance to k-nearest terminals
    dist_to_k_nearest = np.full((num_nodes, num_nearest_terminals), float('inf'), dtype=np.float32)
    finite_dists = []
    
    if term_list:
        for i in range(num_nodes):
            dists_to_i = [dist_from_term[t_idx][i] for t_idx in range(len(term_list)) if i < len(dist_from_term[t_idx])]
            dists_to_i = sorted([d for d in dists_to_i if not math.isinf(d)])
            for k in range(min(num_nearest_terminals, len(dists_to_i))):
                dist_to_k_nearest[i, k] = dists_to_i[k]
                finite_dists.append(dists_to_i[k])

    max_dist = max(finite_dists) if finite_dists else 1.0
    norm_dist_k = np.where(np.isinf(dist_to_k_nearest), 2.0, dist_to_k_nearest / max_dist)
    x[:, 2:] = np.minimum(norm_dist_k, 2.0)
    
    # Edge Index and Attributes
    edge_indices, edge_weights = [], []
    max_weight = 0.0
    for u, v, data in nx_graph.edges(data=True):
        w = data.get('weight', 1.0)
        edge_indices.extend([[u, v], [v, u]])
        edge_weights.extend([w, w])
        max_weight = max(max_weight, w)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1) / (max_weight if max_weight > 0 else 1.0)

    # Mask for non-terminal nodes
    mask = torch.ones(num_nodes, dtype=torch.bool)
    if terminals_set:
        term_indices = torch.tensor(list(terminals_set), dtype=torch.long)
        mask[term_indices] = False
    
    return torch.from_numpy(x), edge_index, edge_attr, mask, max_node_id

def generate_center_score_data_item(nx_g, terminals, dist_from_term, term_list, term_index, max_node_id, args):
    """
    Generates a single PyG Data object (features, labels, etc.) for a graph.
    """
    T_set = set(terminals)
    if len(term_list) < 3 or max_node_id < 0:
        return None

    features_result = extract_center_score_features(nx_g, T_set, term_list, dist_from_term, args.num_nearest_terminals)
    if features_result[0] is None:
        return None
    x, edge_index, edge_attr, non_terminal_mask, _ = features_result
    
    num_nodes = max_node_id + 1
    y_np = np.zeros(num_nodes, dtype=np.float32)
    
    # Calculate target y (potential label) for each non-terminal node
    costs = []
    node_costs = {}
    for s_node in range(num_nodes):
        if s_node in T_set or not nx_g.has_node(s_node): continue

        min_cost_s = float('inf')
        for a, b, c in itertools.combinations(term_list, 3):
            try:
                cost = dist_from_term[term_index[a]][s_node] + \
                       dist_from_term[term_index[b]][s_node] + \
                       dist_from_term[term_index[c]][s_node]
                if not math.isinf(cost):
                    min_cost_s = min(min_cost_s, cost)
            except (KeyError, IndexError):
                continue
        
        if not math.isinf(min_cost_s):
            costs.append(min_cost_s)
            node_costs[s_node] = min_cost_s
            
    if not costs: return None # No valid components found

    threshold = np.median(costs)
    for s_node, cost in node_costs.items():
        if cost <= threshold + args.epsilon:
            y_np[s_node] = 1.0
            
    y = torch.from_numpy(y_np).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, non_terminal_mask=non_terminal_mask)

class CenterScoreDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for CenterScorerGNN.
    This class is suitable for datasets that can be fully loaded into memory.
    """
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list_provided = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['center_scorer_processed_data.pt']
    
    def download(self):
        pass

    def process(self):
        """
        Processes the `data_list_provided` and saves it.
        This method is called automatically by PyG if the processed file doesn't exist.
        """
        if self.data_list_provided is None:
            raise RuntimeError(
                f"Dataset not found at {self.processed_dir}. "
                "Please provide a `data_list` to process or run data generation."
            )
        
        data, slices = self.collate(self.data_list_provided)
        torch.save((data, slices), self.processed_paths[0])