import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from pathlib import Path
import community as community_louvain
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from math import factorial

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_features_and_graph():
    """Load raw 165 features + rebuild edges, aligned by a common node index."""
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    feat_cols = ["txId", "timestep"] + [f"feat_{i}" for i in range(165)]
    features.columns = feat_cols

    edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

    # Map each txId to a row index 0..N-1 so we can use matrix positions
    txids = features["txId"].values
    id_to_idx = {tx: i for i, tx in enumerate(txids)}

    # Feature matrix X: one row per node, 165 columns
    X = features[[f"feat_{i}" for i in range(165)]].values.astype(np.float32)

    return features, edges, id_to_idx, txids, X

def build_normalized_adjacency(edges, id_to_idx, n):
    """Build sparse symmetric-normalized adjacency: D^-1/2 (A+I) D^-1/2."""
    src = edges.iloc[:, 0].map(id_to_idx).values
    dst = edges.iloc[:, 1].map(id_to_idx).values

    # Build sparse A (treat as undirected for diffusion: add both directions)
    data = np.ones(len(src), dtype=np.float32)
    A = sp.coo_matrix((data, (src, dst)), shape=(n, n))
    A = A + A.T                      # make symmetric
    A = A + sp.eye(n, dtype=np.float32)   # add self-loops (the +I)

    # Degree normalization: D^-1/2 A D^-1/2
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    np.power(deg, -0.5, out=d_inv_sqrt, where=deg > 0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.tocsr()

def ppr_diffusion(A_norm, X, alpha=0.2, k=5):
    """Apply PPR kernel to features: sum_{i=0}^k alpha*(1-alpha)^i * A^i X."""
    result = np.zeros_like(X)
    current = X.copy()          # this holds A^i X, starts at A^0 X = X

    for i in range(k + 1):
        weight = alpha * (1 - alpha) ** i
        result += weight * current
        if i < k:
            current = A_norm @ current   # one more hop: A^(i+1) X

    return result

def heat_diffusion(A_norm, X, t=3.0, k=5):
    """Heat kernel (paper Eq. 5): e^-t * sum_{i=0}^k (t^i / i!) * A^i X."""
    result = np.zeros_like(X)
    current = X.copy()
    decay = np.exp(-t)

    for i in range(k + 1):
        weight = decay * (t ** i) / factorial(i)
        result += weight * current
        if i < k:
            current = A_norm @ current

    return result


def cluster_embeddings(diffused_X, n_clusters=20, seed=42):
    """K-means on diffused features (your paper's clustering step)."""
    # Standardize so no single feature dominates the distance metric
    X_scaled = StandardScaler().fit_transform(diffused_X)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels




def louvain_baseline(edges, id_to_idx, n):
    """Baseline: Louvain community detection on the raw graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_list = list(zip(
        edges.iloc[:, 0].map(id_to_idx).values,
        edges.iloc[:, 1].map(id_to_idx).values,
    ))
    G.add_edges_from(edge_list)

    partition = community_louvain.best_partition(G)
    labels = np.array([partition[i] for i in range(n)])
    modularity = community_louvain.modularity(partition, G)
    return labels, modularity, G


def compute_modularity(labels, G):
    """Modularity of a given labeling, to compare against Louvain."""
    partition = {i: int(labels[i]) for i in range(len(labels))}
    return community_louvain.modularity(partition, G)

if __name__ == "__main__":
    print("Loading...")
    features, edges, id_to_idx, txids, X = load_features_and_graph()
    n = len(txids)
    print(f"Nodes: {n}, Features: {X.shape[1]}")

    print("Building normalized adjacency...")
    A_norm = build_normalized_adjacency(edges, id_to_idx, n)

    N_CLUSTERS = 300

    print("Running PPR diffusion (alpha=0.2, k=5)...")
    ppr_X = ppr_diffusion(A_norm, X, alpha=0.2, k=5)
    print(f"K-means on PPR embeddings (k={N_CLUSTERS})... this takes a few minutes")
    ppr_labels = cluster_embeddings(ppr_X, n_clusters=N_CLUSTERS)

    print("Running Heat diffusion (t=3, k=5)...")
    heat_X = heat_diffusion(A_norm, X, t=3.0, k=5)
    print(f"K-means on Heat embeddings (k={N_CLUSTERS})... this takes a few minutes")
    heat_labels = cluster_embeddings(heat_X, n_clusters=N_CLUSTERS)

    print("Louvain baseline...")
    louvain_labels, louvain_mod, G = louvain_baseline(edges, id_to_idx, n)

    print("Computing modularity...")
    ppr_mod = compute_modularity(ppr_labels, G)
    heat_mod = compute_modularity(heat_labels, G)

    print("\n===== MODULARITY RESULTS =====")
    print(f"MSDE PPR  (ours):  {ppr_mod:.4f}   clusters: {len(set(ppr_labels))}")
    print(f"MSDE Heat (ours):  {heat_mod:.4f}   clusters: {len(set(heat_labels))}")
    print(f"Louvain (baseline): {louvain_mod:.4f}   clusters: {len(set(louvain_labels))}")
    print("\nNote: Louvain optimizes modularity directly, so it wins on this metric")
    print("by construction. The decisive test is fraud detection (Day 3).")

    out = pd.DataFrame({
        "txId": txids,
        "ppr_cluster": ppr_labels,
        "heat_cluster": heat_labels,
        "louvain_cluster": louvain_labels,
    })
    out.to_csv(DATA_DIR / "cluster_assignments.csv", index=False)
    print("\nSaved cluster_assignments.csv (PPR + Heat + Louvain)")