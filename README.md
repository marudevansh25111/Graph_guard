# GraphGuard — Graph-Based Transaction Fraud Detection

A fraud detection system that models transactions as a graph instead of treating each
one in isolation. The idea is simple: fraud rings and money-mule networks don't show
up as one bad transaction, they show up as a *pattern* — a tight little cluster of
accounts moving money in a loop, or a single account bridging two clusters that have
no business being connected. A per-transaction classifier can't see that. A graph can.

This repo takes that idea and turns it into something you can actually poke at: a
trained clustering + scoring pipeline, a REST API, and a dashboard, running on a real
labeled fraud dataset (not a toy graph).

Live demo: `<add your deployed URL here once it's up>`

---

## Why graphs instead of just a classifier

Rule-based fraud detection and per-transaction models plateau because fraud rings
adapt — once a rule catches one pattern, the ring just restructures itself to avoid it.
What doesn't change as easily is the *structure* of how mule accounts connect to each
other. That's the more durable signal, and it's the one this project is built around:
detecting "is this account part of a suspicious structure" instead of just "was this
one transaction suspicious."

Every fintech running fraud detection at scale (payment processors, card networks,
crypto exchanges) eventually ends up doing some version of this. This project is a
smaller, self-contained version of that same approach.

---

## How it works

```
Raw transaction data (Elliptic Bitcoin dataset)
        │
        ▼
Graph construction — accounts = nodes, transactions = edges
        │
        ▼
Node features — degree, PageRank, clustering coefficient,
                 transaction velocity, in/out ratio
        │
        ▼
Clustering layer — community detection on the transaction graph
        │
        ▼
Anomaly scoring — cluster-level anomaly signal + a supervised
                   classifier trained on labeled illicit/licit nodes
        │
        ▼
FastAPI backend — /score, /clusters, /graph endpoints
        │
        ▼
Dashboard — graph view, flagged clusters, ranked suspicious accounts
        │
        ▼
Deployed container (Render / Railway / HF Spaces)
```

### Where the clustering method comes from

The clustering step isn't a generic Louvain/k-means run — it's built on a multi-scale
diffusion approach for graph clustering that combines two complementary diffusion
kernels:

- **Heat kernel** — spreads node information across multi-hop neighborhoods with
  exponential decay, which turns out to be good at finding *dense, local* communities
  (tight fraud rings sitting in one neighborhood of the graph).
- **Personalized PageRank (PPR) kernel** — a restart-probability random walk that's
  better at picking up long-range dependencies and hierarchical structure (accounts
  that bridge or launder across otherwise-separate clusters).

Both kernels get folded into the node embedding step before k-means does the final
clustering. In benchmark testing on citation/co-authorship graphs (Cora, CiteSeer,
PubMed, Coauthor-CS, Coauthor-Physics), this combined approach beat a standard
k-hop-diffusion baseline on NMI, accuracy, and F1 across the board — with the PPR
kernel winning on most datasets and the Heat kernel winning specifically on the
densest, most locally-clustered graph (Cora). That's the same reason it's a good fit
here: fraud rings *are* dense local structures, and mule networks laundering across
clusters *are* the long-range case PPR is built for.

Applying this to the Elliptic dataset is the actual point of the project — going from
"this clustering method does well on academic citation graphs" to "this clustering
method flags real fraud rings in transaction data."

---

## Dataset

**[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)**
— ~203K nodes, ~234K edges, ~2% labeled illicit, published by Elliptic/MIT/IBM. Chosen
because it's already graph-native (nodes = Bitcoin transactions, edges = flow of
funds), so there's no need to reconstruct a graph from raw tabular data the way you
would with something like IEEE-CIS.

---

## Tech stack

| Layer | Tools |
|---|---|
| Graph & ML | Python, NetworkX, PyTorch, PyTorch Geometric, scikit-learn, XGBoost |
| API | FastAPI |
| Dashboard | Streamlit + PyVis (or React/Cytoscape.js) |
| Explainability | SHAP |
| Deployment | Docker → Render / Railway / HuggingFace Spaces |

---

## Running it locally

```bash
git clone <this-repo>
cd graphguard
pip install -r requirements.txt

# build the graph + train the clustering/scoring pipeline
python src/build_graph.py
python src/train.py

# start the API
uvicorn src.api:app --reload --port 8000

# start the dashboard (separate terminal)
streamlit run src/dashboard.py
```

### API endpoints

| Endpoint | What it does |
|---|---|
| `GET /score/{account_id}` | Returns a fraud-risk score + which cluster the account belongs to + why it was flagged |
| `GET /clusters` | Lists all detected clusters with their anomaly scores |
| `GET /graph/neighborhood/{account_id}` | Returns the local subgraph around an account, for drill-down |

---

## Results

*(fill in after training — this is the part that actually matters for the writeup)*

| Method | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Louvain (baseline) | | | | |
| K-means on raw features | | | | |
| This project (Heat/PPR clustering + classifier) | | | | |

Target: AUC > 0.85 — the Elliptic dataset supports this given the labeled illicit
class and existing published baselines on it.

---

## What this is and isn't

This is **not** a system running in production at a bank — that would need data
agreements, compliance review, and integration work well beyond the scope of a
personal project, and I'm not going to pretend otherwise. What this *is*: an
end-to-end system, architected the way a production fraud detection service would
be, running against a real labeled dataset, with a live API and dashboard anyone can
hit and get an actual score back from. Deployed, not just a notebook with an AUC
number in it.

---

## Limitations / honest gaps

- Tested on one primary dataset (Elliptic). Generalization to card-transaction or
  UPI-style data isn't validated here — IEEE-CIS would be the natural next dataset
  to check that against.
- The clustering method uses the Heat and PPR kernels separately, not combined —
  a hybrid kernel might do better but adds compute overhead that wasn't worth it
  for this scope.
- No real streaming infrastructure — the "real-time" mode simulates transactions
  arriving one at a time rather than using actual streaming infra like Kafka.
- Anomaly thresholds are currently tuned empirically, not derived from a formal
  cost/risk model — in a real deployment you'd want that tied to actual
  false-positive costs.

---

## Roadmap / stretch goals

- [ ] Real-time simulation mode (stream transactions in, watch scores update live)
- [ ] SHAP-based explainability panel per flagged account
- [ ] Webhook/email alerting when a cluster crosses a risk threshold
- [ ] Combined Heat+PPR hybrid kernel experiment

---

## Credits

Clustering approach adapted from multi-scale diffusion research (Heat kernel +
Personalized PageRank kernels for graph clustering), applied here to a real
fraud-detection dataset.
