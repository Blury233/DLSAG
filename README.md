# DLSAG: Dynamic Load-aware Steiner Aggregation

This is the official implementation for our paper "DLSAG: Dynamic Load-aware Steiner Aggregation for Large-Scale Network Path Optimization".

## Dependencies
- Python 3.8
- PyTorch 2.1
- PyTorch Geometric 2.6
- NetworkX 3.0

## Running the Code

The project workflow is divided into three main steps, which can be executed using the main script `main.py`.

### Step 1: Generate Training Data

First, generate a synthetic dataset of graphs with pre-computed labels for training the `CenterScorerGNN`. The GNN learns to identify nodes that are good candidates for being Steiner points.

To generate the dataset with default parameters, run:

```bash
python main.py generate_data
```

This will create and process a dataset, saving it to the `data/center_score_dataset/` directory. If a dataset already exists, this command will do nothing. To force regeneration, use the `--force_regenerate_data` flag:

```bash
python main.py generate_data --force_regenerate_data
```

Customize generation parameters, such as:

```bash
python main.py generate_data --num_graphs_data 1000
```

### Step 2: Train the GNN Model

Next, train the `CenterScorerGNN` on the generated dataset. The script will train the model, monitor validation loss, and save the best-performing model weights to `trained_models/center_scorer_model.pt`.

To train a model with default settings (using GIN architecture):
```bash
python main.py train
```

Switch to another GNN architecture using the `--gnn_type` argument and customize training parameters:
```bash
python main.py train --gnn_type gat --epochs 150
```

### Step 3: Run the DLSAG Solver

Finally, use the trained model to solve the Steiner Tree problem on a test graph.

To run the solver with default settings:
```bash
python main.py run
```
