# ==========================================
# KG Embedding Redundancy Analysis (TransE)
# ==========================================

import pickle
import torch
import pandas as pd
import networkx as nx
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

import sys

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


# ==========================================
# CONFIG
# ==========================================

GRAPH_PATH = "directed_test.gpickle"
QUERY_FILE = "directed_test_query.txt"

MODEL_NAME = "TransE"
EPOCHS = 200
EMBEDDING_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Load Graph
# ==========================================

def load_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("Graph must be MultiDiGraph")

    return G


# ==========================================
# Convert Graph to Triples
# ==========================================

def graph_to_triples(G):

    triples = []

    for u, v, key in G.edges(keys=True):
        triples.append((u, key, v))

    return pd.DataFrame(triples, columns=["head", "relation", "tail"])


# ==========================================
# Parse Queries
# ==========================================

def parse_queries(path):

    queries = []

    with open(path, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        for line in lines:
            if line.startswith("DELETE_EDGE:"):
                edge = line.replace("DELETE_EDGE:", "").strip()
                h, r, t = [x.strip() for x in edge.split("|")]
                queries.append((h, r, t))

    return queries


# ==========================================
# Train Embedding Model Once
# ==========================================

def train_model(triples_df):

    tf = TriplesFactory.from_labeled_triples(
        triples_df.values
    )

    result = pipeline(
        model=MODEL_NAME,
        training=tf,
        testing=tf,  # <-- ADD THIS
        training_kwargs=dict(num_epochs=EPOCHS),
        model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
        device=DEVICE,
        random_seed=42,
    )

    return result.model, tf



# ==========================================
# Rank Triple
# ==========================================

def evaluate_triple(model, tf, triple):

    h, r, t = triple

    h_id = tf.entity_to_id[h]
    r_id = tf.relation_to_id[r]
    t_id = tf.entity_to_id[t]

    # Prepare batch
    hr_batch = torch.tensor([[h_id, r_id]], device=DEVICE)

    # Score all candidate tails
    scores = model.score_t(hr_batch)  # shape: (1, num_entities)
    scores = scores.squeeze(0)

    # Sort descending (higher score = more likely)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    # Find rank of true tail
    rank = (sorted_indices == t_id).nonzero(as_tuple=True)[0].item() + 1

    reciprocal_rank = 1.0 / rank

    hits1 = int(rank <= 1)
    hits3 = int(rank <= 3)
    hits10 = int(rank <= 10)

    redundancy = reciprocal_rank

    return {
        "rank": rank,
        "MRR": reciprocal_rank,
        "Hits@1": hits1,
        "Hits@3": hits3,
        "Hits@10": hits10,
        "redundancy": redundancy
    }



# ==========================================
# Main
# ==========================================

def main():

    print("\nLoading graph...")
    G = load_graph(GRAPH_PATH)

    triples_df = graph_to_triples(G)
    queries = parse_queries(QUERY_FILE)

    print("Training TransE model...")
    model, tf = train_model(triples_df)

    print("\nEvaluating deleted triples (inference only removal)...\n")

    for i, triple in enumerate(queries, 1):

        print("=" * 70)
        print(f"Triple {i}: {triple}")
        print("=" * 70)

        metrics = evaluate_triple(model, tf, triple)

        print("Rank:", metrics["rank"])
        print("MRR:", round(metrics["MRR"], 4))
        print("Hits@1:", metrics["Hits@1"])
        print("Hits@3:", metrics["Hits@3"])
        print("Hits@10:", metrics["Hits@10"])
        print("Redundancy (1/Rank):", round(metrics["redundancy"], 4))
        print()

    print("Done.")


if __name__ == "__main__":

    # Redirect terminal output to file
    sys.stdout = Tee("kf_redundancy_output.txt")

    main()

    # Restore normal stdout
    sys.stdout.file.close()
    sys.stdout = sys.stdout.terminal

