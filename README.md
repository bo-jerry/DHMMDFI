# DHMMDFI

DHMMDFI is a PyTorch-based heterogeneous graph learning project for Drug-Food and Drug-Food Component interaction prediction. The model combines a semantic meta-path view with a local relation view to predict both association types.

## Tasks

The release supports two datasets and two prediction tasks:

- `drugbank` or `pubmed`
- `df`: Drug-Food interaction prediction
- `dfc`: Drug-Food Component interaction prediction

Only the dataset and task are selected from the command line. The remaining training and model settings are fixed inside the release configuration.

## Project Structure

```text
DHMMDFI/
|-- code/
|   |-- main.py
|   |-- config.py
|   |-- data.py
|   |-- model.py
|   `-- train.py
|-- data/
|   |-- DrugBank-DF/   (Drug-Food)
|   |-- DrugBank-DFC/  (Drug-Food Component)
|   |-- PubMed-DF/     (Drug-Food)
|   `-- PubMed-DFC/    (Drug-Food Component)
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Core Files

- `code/main.py`: command-line entry point.
- `code/config.py`: fixed model and training configuration.
- `code/data.py`: data loading, graph construction, feature loading, negative sampling, and five-fold splitting.
- `code/model.py`: complete DHMMDFI model, including the encoder, Mamba modules, local relation encoder, contrastive objective, and link predictor.
- `code/train.py`: five-fold training, evaluation, metric calculation, and result saving.

## Model Overview

The model contains two complementary representation views:

1. A semantic meta-path view models long-range heterogeneous graph structure with PathSim-weighted paths and Mamba-based sequence encoding.
2. A local relation view models short-range typed neighborhoods.
3. The two views are normalized and fused for link prediction.
4. A cross-view contrastive objective regularizes the learned representations.

## Data

The repository contains preprocessed graph data and features for all four dataset-task combinations. The data folders include graph edges, node counts, multi-hop neighborhoods, sparse matrices, and precomputed node features.

The main files include:

- `df_edges.npy`: Drug-Food interaction edges.
- `dfc_edges.npy`: Drug-Food Component interaction edges.
- `ffc_edges.npy`: food-food-component edges.
- `drug_features.npy`, `food_features.npy`, `fc_features.npy`: node features.
- `drug_multi_hop_*.npy`, `fc_multi_hop_*.npy`: multi-hop neighborhoods.
- `num_nodes.npy`: numbers of nodes of each type.

Raw names, SMILES tables, identifier mappings, and intermediate cleaning files are not required by the release and are not included.

## Environment

Install the required packages with:

```bash
pip install -r requirements.txt
```

The experiments were tested with Python, PyTorch, NumPy, SciPy, scikit-learn, pandas, and RDKit.

## How to Run

Run commands from the `code` directory:

```bash
cd code
python main.py --dataset drugbank --task df
python main.py --dataset drugbank --task dfc
python main.py --dataset pubmed --task df
python main.py --dataset pubmed --task dfc
```

The five-fold split and training protocol are generated automatically from the fixed release configuration. No additional preprocessing or ablation command is required.

## Outputs

During training, the program reports AUC, AUPR, F1, recall, and precision. Detailed text and CSV summaries are written to the `code/results/` directory. This directory is generated at runtime and is excluded from version control.

## Citation

If you use this repository, please cite the corresponding DHMMDFI paper and describe the dataset and task configuration used in your experiments.
