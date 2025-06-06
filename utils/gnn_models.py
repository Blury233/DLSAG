import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
from torch_geometric.nn import GINConv, GCNConv, GATConv, GatedGraphConv

class CenterScorerGNN(torch.nn.Module):
    """
    A flexible Graph Neural Network designed to predict node scores for identifying
    potential Steiner points. It supports multiple GNN architectures.
    """
    def __init__(self, num_node_features, hidden_channels, num_layers, gnn_type='gin', out_channels=1, heads=4):
        """
        Args:
            num_node_features (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of hidden embeddings.
            num_layers (int): Number of GNN layers.
            gnn_type (str): The type of GNN layer to use ('gin', 'gcn', 'gat', 'gatedgcn').
            out_channels (int): Dimensionality of the output (1 for a single score).
            heads (int): Number of attention heads for GAT.
        """
        super().__init__()
        torch.manual_seed(42)
        self.gnn_type = gnn_type.lower()
        
        # --- Input Layer ---
        # Encodes raw node features into the hidden dimension space.
        self.node_encoder = Linear(num_node_features, hidden_channels)
        
        # --- Hidden GNN Layers ---
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for i in range(num_layers):
            in_channels = hidden_channels
            conv = self._create_conv_layer(in_channels, hidden_channels, heads)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # --- Output Layer ---
        self.output_layer = Linear(hidden_channels, out_channels)

    def _create_conv_layer(self, in_channels, out_channels, heads):
        """Instantiate a message‑passing operator as specified by :pyattr:`gnn_type`."""
        if self.gnn_type == "gin":
            mlp = Sequential(
                Linear(in_channels, 2 * in_channels),
                BatchNorm1d(2 * in_channels),
                ReLU(),
                Linear(2 * in_channels, out_channels)
            )
            return GINConv(nn=mlp, train_eps=True)

        if self.gnn_type == "gcn":
            # Classic GCN layer (Kipf & Welling, 2017).
            return GCNConv(in_channels, out_channels)

        if self.gnn_type == "gat":
            # Multi‑head GAT layer (Veličković et al., 2018).
            return GATConv(
                in_channels=in_channels, out_channels=out_channels, heads=heads, concat=False, edge_dim=1,
            )

        if self.gnn_type == "gatedgcn":
            # Gated Graph Conv (Li et al., 2016) configured for a single
            # propagation step; stacking occurs at the network level.
            return GatedGraphConv(
                out_channels=out_channels, num_layers=1, aggr="mean"
            )

        raise ValueError(
            f"Unsupported gnn_type: '{self.gnn_type}'. "
            "Choose from {'gin', 'gcn', 'gat', 'gatedgcn'}."
        )


    def forward(self, data):
        """
        The forward pass of the GNN.
        
        Args:
            data (torch_geometric.data.Data or Batch): A PyG data object.
        
        Returns:
            torch.Tensor: Node-level logits.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if x is None:
            raise ValueError("Input node features 'x' cannot be None.")

        # 1. Encode input features
        x = self.node_encoder(x)
        
        # 2. Propagate through GNN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x_res = x 
            if self.gnn_type in ['gat', 'gatedgcn']:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + x_res

        # 3. Decode final embeddings to scores
        node_logits = self.output_layer(x)
        
        return node_logits