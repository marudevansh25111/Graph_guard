import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_all():
    """Merge raw features + structural features + cluster assignments."""
    raw = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    raw.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(165)]

    struct = pd.read_csv(DATA_DIR / "node_features.csv")
    clusters = pd.read_csv(DATA_DIR / "cluster_assignments.csv")

    df = raw.merge(
        struct[["txId", "in_degree", "out_degree", "total_degree",
                "in_out_ratio", "pagerank", "clustering_coeff", "label"]],
        on="txId", how="left"
    ).merge(clusters, on="txId", how="left")

    return df

def add_cluster_features(df, cluster_col, train_mask):
    """Cluster stats computed from TRAIN labels only (no leakage)."""
    train_df = df[train_mask & (df["label"] != -1)]

    # Fraction of each cluster's labeled train nodes that are illicit
    illicit_frac = train_df.groupby(cluster_col)["label"].mean()
    # How many labeled train nodes we saw per cluster (confidence in the above)
    labeled_count = train_df.groupby(cluster_col)["label"].count()
    # Total cluster size (structure only, no labels -> safe to use all nodes)
    cluster_size = df.groupby(cluster_col).size()

    out = pd.DataFrame(index=df.index)
    out[f"{cluster_col}_illicit_frac"] = df[cluster_col].map(illicit_frac).fillna(0.0)
    out[f"{cluster_col}_labeled_count"] = df[cluster_col].map(labeled_count).fillna(0)
    out[f"{cluster_col}_size"] = df[cluster_col].map(cluster_size).fillna(0)
    return out




def run_experiment(df, cluster_col, split_time=34):
    """Train XGBoost using one clustering's features; report test metrics."""
    train_mask = df["timestep"] <= split_time
    test_mask = df["timestep"] > split_time

    cluster_feats = add_cluster_features(df, cluster_col, train_mask)
    work = pd.concat([df, cluster_feats], axis=1)

    feature_cols = (
        [f"feat_{i}" for i in range(165)]
        + ["in_degree", "out_degree", "total_degree", "in_out_ratio",
           "pagerank", "clustering_coeff"]
        + list(cluster_feats.columns)
    )

    # Only labeled nodes can be trained/evaluated on
    tr = work[train_mask & (work["label"] != -1)]
    te = work[test_mask & (work["label"] != -1)]

    X_tr, y_tr = tr[feature_cols].values, tr["label"].values
    X_te, y_te = te[feature_cols].values, te["label"].values

    # Class imbalance: illicit is ~10% of labeled nodes
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "clustering": cluster_col,
        "auc": roc_auc_score(y_te, proba),
        "f1": f1_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "train_n": len(tr),
        "test_n": len(te),
    }, model, feature_cols

if __name__ == "__main__":
    print("Loading and merging...")
    df = load_all()
    print(f"Rows: {len(df)}")

    results = []
    for col in ["ppr_cluster", "heat_cluster", "louvain_cluster"]:
        print(f"\nTraining with {col}...")
        res, model, feat_cols = run_experiment(df, col)
        results.append(res)
        print(f"  AUC={res['auc']:.4f}  F1={res['f1']:.4f} "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}")

    print("\n===== FRAUD DETECTION RESULTS (temporal split) =====")
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    table.to_csv(DATA_DIR / "results_table.csv", index=False)
    print("\nSaved results_table.csv")


    import joblib

    # Retrain the winner (Heat) and persist everything the API needs
    print("\nSaving best model (Heat kernel)...")
    best_res, best_model, feat_cols = run_experiment(df, "heat_cluster")

    train_mask = df["timestep"] <= 34
    cluster_feats = add_cluster_features(df, "heat_cluster", train_mask)
    scored = pd.concat([df, cluster_feats], axis=1)

    joblib.dump(best_model, DATA_DIR / "model.pkl")
    joblib.dump(feat_cols, DATA_DIR / "feature_cols.pkl")
    scored.to_csv(DATA_DIR / "scored_nodes.csv", index=False)
    print("Saved model.pkl, feature_cols.pkl, scored_nodes.csv")