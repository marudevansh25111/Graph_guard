import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw():
    """Load and clean the three Elliptic CSVs."""
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

    feat_cols = ["txId", "timestep"] + [f"feat_{i}" for i in range(165)]
    features.columns = feat_cols
    classes["class"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1})
    return features, classes, edges


def build_graph(features, classes, edges):
    """Directed graph: nodes = transactions, edges = BTC flow."""
    label_map = dict(zip(classes["txId"], classes["class"]))
    G = nx.DiGraph()
    for _, row in features.iterrows():
        txid = int(row["txId"])
        G.add_node(txid, timestep=int(row["timestep"]), label=label_map.get(txid, -1))
    edge_tuples = list(zip(edges.iloc[:, 0].astype(int), edges.iloc[:, 1].astype(int)))
    G.add_edges_from(edge_tuples)
    return G


def compute_node_features(G):
    """Structural features per node."""
    print("Computing degree...")
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    print("Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    print("Computing clustering coefficient...")
    clustering = nx.clustering(G.to_undirected())

    rows = []
    for n in G.nodes():
        indeg, outdeg = in_deg.get(n, 0), out_deg.get(n, 0)
        rows.append({
            "txId": n,
            "in_degree": indeg,
            "out_degree": outdeg,
            "total_degree": indeg + outdeg,
            "in_out_ratio": indeg / (outdeg + 1),
            "pagerank": pagerank.get(n, 0),
            "clustering_coeff": clustering.get(n, 0),
            "label": G.nodes[n]["label"],
            "timestep": G.nodes[n]["timestep"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    features, classes, edges = load_raw()
    print(f"Nodes: {len(features)}, Edges: {len(edges)}")
    print(f"Label counts:\n{classes['class'].value_counts()}")

    G = build_graph(features, classes, edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_features = compute_node_features(G)
    node_features.to_csv(DATA_DIR / "node_features.csv", index=False)
    print("Saved node_features.csv")
    print(node_features.head())