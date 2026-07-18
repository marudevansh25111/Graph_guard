import joblib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

print("Loading...")
model = joblib.load(DATA_DIR / "model.pkl")
feature_cols = joblib.load(DATA_DIR / "feature_cols.pkl")
df = pd.read_csv(DATA_DIR / "scored_nodes.csv")

print("Scoring all nodes (batched)...")
X = df[feature_cols].values.astype(np.float32)
df["risk_score"] = model.predict_proba(X)[:, 1]

keep = [
    "txId", "timestep", "label", "risk_score",
    "heat_cluster", "heat_cluster_illicit_frac",
    "heat_cluster_size", "heat_cluster_labeled_count",
    "in_degree", "out_degree", "total_degree",
    "in_out_ratio", "pagerank", "clustering_coeff",
]
slim = df[keep].copy()

slim.to_parquet(DATA_DIR / "nodes_slim.parquet", index=False, compression="snappy")
print(f"Saved nodes_slim.parquet — {len(slim)} rows, {len(keep)} cols")

print("Slimming edges...")
edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")
edges.columns = ["source", "target"]
edges = edges.astype("int64")
edges.to_parquet(DATA_DIR / "edges_slim.parquet", index=False, compression="snappy")
print(f"Saved edges_slim.parquet — {len(edges)} edges")