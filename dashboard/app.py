import os
import tempfile

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GraphGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ GraphGuard — Fraud Ring Detection")
st.caption(
    "Graph clustering (MSDE Heat kernel diffusion) applied to the Elliptic Bitcoin "
    "transaction graph — 203,769 nodes, 234,355 edges"
)


@st.cache_data(ttl=300)
def api_get(path):
    r = requests.get(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def render_graph(nb, height=520):
    """Render a PyVis network from a /graph/neighborhood response."""
    net = Network(height=f"{height - 20}px", width="100%", directed=True, bgcolor="#ffffff")

    for n in nb["nodes"]:
        is_center = int(n["id"]) == int(nb["center"])
        if is_center:
            color = "#3498db"
        elif n["label"] == 1:
            color = "#e74c3c"
        elif n["label"] == 0:
            color = "#2ecc71"
        else:
            color = "#95a5a6"

        net.add_node(
            n["id"],
            label=str(n["id"]),
            color=color,
            size=30 if is_center else 12,
            title=(
                f"cluster {n['cluster']} | "
                f"risk {n['risk_score']} | "
                f"illicit_frac {n['illicit_frac']}"
            ),
        )

    for e in nb["edges"]:
        net.add_edge(e["source"], e["target"])

    net.repulsion(node_distance=120, spring_length=100)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.save_graph(f.name)
        path = f.name
    try:
        with open(path, "r") as f:
            components.html(f.read(), height=height)
    finally:
        os.unlink(path)


# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.markdown("""
**GraphGuard** detects fraud *rings*, not fraud transactions.

Traditional models score each transaction independently and miss coordinated
laundering: accounts that look unremarkable alone but form suspicious structures
together.

**Method:** Multi-Scale Diffusion Enhancement (MSDE) — Heat and Personalized
PageRank kernels applied to the transaction graph, then K-means clustering, then
XGBoost scoring on structural + cluster features.

Based on: *MSDE: Multi-Scale Diffusion Enhancement for Graph Clustering with Heat
and Personalized PageRank Kernels* (IC2SDT 2025).

---

**Try:** `70384401` — a known-illicit account with in-degree 0 and clustering
coefficient 0. Individually invisible. Flagged at 0.99 because 38% of its
community is illicit.
""")


# ---------- connection check ----------
try:
    health = api_get("/")
except Exception:
    st.error(
        f"Cannot reach the API at {API_URL}. "
        "Start it with: `uvicorn api.main:app --reload --port 8000`"
    )
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Nodes", f"{health['nodes']:,}")
c2.metric("Edges", f"{health['edges']:,}")
c3.metric("Status", health["status"].upper())

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Suspicious Clusters",
    "🕸️ Account Lookup",
    "📊 Cluster Drill-Down",
    "🧪 Method Comparison",
])


# ---------- TAB 1 ----------
with tab1:
    st.subheader("Top fraud rings by illicit concentration")

    col_a, col_b = st.columns(2)
    limit = col_a.slider("How many clusters", 5, 50, 15)
    min_labeled = col_b.slider("Min labeled nodes (evidence floor)", 1, 100, 20)

    data = api_get(f"/clusters?limit={limit}&min_labeled={min_labeled}")
    df = pd.DataFrame(data["clusters"])

    if df.empty:
        st.warning("No clusters meet that evidence threshold. Lower the slider.")
    else:
        st.dataframe(
            df.style.background_gradient(subset=["illicit_fraction"], cmap="Reds"),
            use_container_width=True,
        )
        st.caption(
            "**illicit_fraction** = share of this cluster's labeled nodes that are known illicit. "
            "This is the fraud-ring signal: an account is suspicious because of its community, "
            "not just its own behaviour."
        )
        st.info(
            f"Evidence floor: clusters with fewer than {min_labeled} labeled nodes are excluded. "
            "Without this, a cluster with 1 illicit node scores a meaningless 100%."
        )


# ---------- TAB 2 ----------
with tab2:
    st.subheader("Score an account")
    tx_id = st.text_input("Transaction ID", value="70384401")

    if st.button("Analyze"):
        try:
            res = api_get(f"/score/{tx_id}")
            st.session_state["last_tx"] = tx_id
            st.session_state["last_res"] = res
        except requests.HTTPError:
            st.error(f"Transaction {tx_id} not found")
            st.session_state.pop("last_res", None)

    if "last_res" in st.session_state:
        res = st.session_state["last_res"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{res['risk_score']:.3f}")
        m2.metric("Risk Level", res["risk_level"])
        m3.metric("Cluster", res["cluster_id"])

        icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        st.markdown(f"### {icons[res['risk_level']]} {res['risk_level']} RISK")

        st.write("**Why this score:**")
        e = res["explanation"]

        e1, e2 = st.columns(2)
        with e1:
            st.write("*Community signal (fraud ring)*")
            st.write(f"- Cluster illicit fraction: **{e['cluster_illicit_fraction']}**")
            st.write(f"- Cluster size: {e['cluster_size']:,}")
            st.write(f"- Labeled nodes in cluster: {e['cluster_labeled_count']}")
        with e2:
            st.write("*Individual signal (structural)*")
            st.write(f"- In-degree: {e['in_degree']}")
            st.write(f"- Out-degree: {e['out_degree']}")
            st.write(f"- PageRank: {e['pagerank']}")
            st.write(f"- Clustering coefficient: {e['clustering_coefficient']}")

        gt = {1: "Known illicit", 0: "Known licit", -1: "Unlabeled"}[res["ground_truth"]]
        st.info(f"Ground truth: **{gt}**")

        st.divider()
        st.subheader("Transaction neighborhood")
        hops = st.radio("Hops", [1, 2], horizontal=True, key="hops_lookup")

        nb = api_get(f"/graph/neighborhood/{st.session_state['last_tx']}?hops={hops}")
        st.write(f"{len(nb['nodes'])} nodes, {len(nb['edges'])} edges")
        if nb.get("truncated"):
            st.warning("Neighborhood truncated to 100 nodes for readability.")

        st.caption("🔵 selected  🔴 known illicit  🟢 known licit  ⚪ unlabeled")
        render_graph(nb)


# ---------- TAB 3 ----------
with tab3:
    st.subheader("Investigate a cluster")
    cluster_id = st.number_input("Cluster ID", min_value=0, value=136, step=1)

    if st.button("Load cluster"):
        try:
            detail = api_get(f"/cluster/{int(cluster_id)}?limit=50")
            st.session_state["cluster_detail"] = detail
        except requests.HTTPError:
            st.error(f"Cluster {cluster_id} not found")

    if "cluster_detail" in st.session_state:
        d = st.session_state["cluster_detail"]

        k1, k2, k3 = st.columns(3)
        k1.metric("Cluster ID", d["cluster_id"])
        k2.metric("Size", f"{d['size']:,}")
        k3.metric("Illicit fraction", f"{d['illicit_fraction']:.3f}")

        acc = pd.DataFrame(d["accounts"])
        acc["label"] = acc["label"].map({1: "illicit", 0: "licit", -1: "unlabeled"})

        st.write("**Riskiest accounts in this cluster**")
        st.dataframe(
            acc.style.background_gradient(subset=["risk_score"], cmap="Reds"),
            use_container_width=True,
        )
        st.caption(
            "Ranked by model risk score. Note how many are *unlabeled* — those are the "
            "accounts the system surfaces that ground truth never flagged."
        )


# ---------- TAB 4 ----------
with tab4:
    st.subheader("MSDE diffusion kernels vs. Louvain baseline")
    st.write(
        "Fraud detection performance using each clustering method's features, "
        "XGBoost classifier, temporal split (train ≤ t34, test > t34)."
    )

    res = pd.DataFrame({
        "Clustering": ["MSDE Heat (ours)", "MSDE PPR (ours)", "Louvain (baseline)"],
        "AUC": [0.9294, 0.9261, 0.9226],
        "F1": [0.7929, 0.7843, 0.7902],
        "Precision": [0.8725, 0.8635, 0.9944],
        "Recall": [0.7267, 0.7184, 0.6556],
    })
    st.dataframe(
        res.style.background_gradient(subset=["AUC", "F1"], cmap="Greens"),
        use_container_width=True,
    )

    st.markdown("""
**Heat kernel wins on AUC and F1.** This matches the MSDE paper's finding that the Heat
kernel excels on dense local communities — and fraud rings *are* dense local communities.

**On the Louvain precision anomaly (0.9944):** Louvain's communities follow connected
components, so a flagged cluster is almost purely illicit — near-zero false positives.
But it only catches fraud inside already-known-bad components, missing **34%** of fraud.
Heat trades ~12 points of precision for ~7 points of recall. For a fraud team, missed
fraud is direct loss; a false positive is a review ticket. That trade is usually worth it.

**On modularity:** Louvain scores 0.98 vs Heat's 0.42 — but Louvain *optimizes modularity
directly*, so that's grading it on its own exam. The Elliptic graph is highly fragmented,
so labeling each component a community earns high modularity for free. Modularity doesn't
measure fraud detection; the table above does.
""")