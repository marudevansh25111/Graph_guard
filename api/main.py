import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from fastapi import FastAPI, HTTPException

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(title="GraphGuard", description="Graph-based fraud ring detection")

print("Loading data...")
scored = pd.read_parquet(DATA_DIR / "nodes_slim.parquet")
scored = scored.set_index("txId", drop=False)

edges_df = pd.read_parquet(DATA_DIR / "edges_slim.parquet")
G = nx.from_pandas_edgelist(
    edges_df, source="source", target="target", create_using=nx.DiGraph()
)
G = nx.from_pandas_edgelist(
    edges_df,
    source=edges_df.columns[0],
    target=edges_df.columns[1],
    create_using=nx.DiGraph(),
)
print(f"Ready: {len(scored)} nodes, {G.number_of_edges()} edges")


@app.get("/")
def root():
    return {
        "service": "GraphGuard",
        "status": "healthy",
        "nodes": len(scored),
        "edges": G.number_of_edges(),
        "method": "MSDE Heat kernel diffusion + XGBoost",
        "endpoints": ["/score/{tx_id}", "/clusters", "/graph/neighborhood/{tx_id}", "/docs"],
    }


@app.get("/score/{tx_id}")
def score_transaction(tx_id: int):
    """Fraud-risk score for a transaction, with explanation."""
    if tx_id not in scored.index:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")

    row = scored.loc[tx_id]
    risk = float(row["risk_score"])
    level = "HIGH" if risk >= 0.7 else "MEDIUM" if risk >= 0.4 else "LOW"

    return {
        "txId": tx_id,
        "risk_score": round(risk, 4),
        "risk_level": level,
        "cluster_id": int(row["heat_cluster"]),
        "explanation": {
            "cluster_illicit_fraction": round(float(row["heat_cluster_illicit_frac"]), 4),
            "cluster_size": int(row["heat_cluster_size"]),
            "cluster_labeled_count": int(row["heat_cluster_labeled_count"]),
            "in_degree": int(row["in_degree"]),
            "out_degree": int(row["out_degree"]),
            "pagerank": round(float(row["pagerank"]), 8),
            "clustering_coefficient": round(float(row["clustering_coeff"]), 4),
        },
        "ground_truth": int(row["label"]),
    }


@app.get("/clusters")
def list_clusters(limit: int = 20, min_labeled: int = 20):
    """Top suspicious clusters, ranked by illicit concentration."""
    agg = scored.groupby("heat_cluster").agg(
        size=("txId", "count"),
        illicit_frac=("heat_cluster_illicit_frac", "first"),
        labeled_count=("heat_cluster_labeled_count", "first"),
        mean_risk=("risk_score", "mean"),
    ).reset_index()

    agg = agg[agg["labeled_count"] >= min_labeled]
    agg = agg.sort_values("illicit_frac", ascending=False).head(limit)

    return {
        "min_labeled_threshold": min_labeled,
        "clusters": [
            {
                "cluster_id": int(r["heat_cluster"]),
                "size": int(r["size"]),
                "illicit_fraction": round(float(r["illicit_frac"]), 4),
                "labeled_nodes": int(r["labeled_count"]),
                "mean_risk_score": round(float(r["mean_risk"]), 4),
            }
            for _, r in agg.iterrows()
        ],
    }


@app.get("/cluster/{cluster_id}")
def cluster_detail(cluster_id: int, limit: int = 50):
    """Riskiest accounts within a given cluster."""
    members = scored[scored["heat_cluster"] == cluster_id]
    if len(members) == 0:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    top = members.sort_values("risk_score", ascending=False).head(limit)

    return {
        "cluster_id": cluster_id,
        "size": len(members),
        "illicit_fraction": round(float(members["heat_cluster_illicit_frac"].iloc[0]), 4),
        "accounts": [
            {
                "txId": int(r["txId"]),
                "risk_score": round(float(r["risk_score"]), 4),
                "label": int(r["label"]),
                "timestep": int(r["timestep"]),
            }
            for _, r in top.iterrows()
        ],
    }


@app.get("/graph/neighborhood/{tx_id}")
def get_neighborhood(tx_id: int, hops: int = 1, max_nodes: int = 100):
    """Local subgraph around a transaction, for visualization."""
    if tx_id not in G:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not in graph")

    undirected = G.to_undirected(as_view=True)
    reachable = nx.single_source_shortest_path_length(undirected, tx_id, cutoff=hops)
    node_ids = list(reachable.keys())[:max_nodes]
    sub = G.subgraph(node_ids)

    node_list = []
    for n in sub.nodes():
        if n in scored.index:
            r = scored.loc[n]
            node_list.append({
                "id": int(n),
                "cluster": int(r["heat_cluster"]),
                "label": int(r["label"]),
                "risk_score": round(float(r["risk_score"]), 4),
                "illicit_frac": round(float(r["heat_cluster_illicit_frac"]), 4),
            })
        else:
            node_list.append({
                "id": int(n), "cluster": -1, "label": -1,
                "risk_score": 0.0, "illicit_frac": 0.0,
            })

    return {
        "center": tx_id,
        "hops": hops,
        "truncated": len(reachable) > max_nodes,
        "nodes": node_list,
        "edges": [{"source": int(u), "target": int(v)} for u, v in sub.edges()],
    }