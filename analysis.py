# ============================================================
# Experiment Analysis Script (Corrected + Stable)
# ============================================================

import re
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Optional logistic regression
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_FILE = "experiment_output.txt"
KF_FILE = "kf_redundancy_output.txt"

OUTPUT_LLM_CSV = "llm_results.csv"
OUTPUT_KF_CSV = "kf_results.csv"
OUTPUT_MERGED_CSV = "merged_results.csv"

OUTPUT_ROOT = "analysis_outputs"
PLOTS_DIR = os.path.join(OUTPUT_ROOT, "plots")
TABLE_DIR = os.path.join(OUTPUT_ROOT, "summary_tables")

sns.set(style="whitegrid")


# ============================================================
# Parse KG Embedding Output
# ============================================================

def parse_kf_output(path):

    triples = []
    ranks = []
    mrrs = []
    redundancy = []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    triple_blocks = re.split(r"=+\nTriple", content)

    for block in triple_blocks:
        triple_match = re.search(r"\('(.+?)', '(.+?)', '(.+?)'\)", block)
        rank_match = re.search(r"Rank:\s*(\d+)", block)
        mrr_match = re.search(r"MRR:\s*([\d\.]+)", block)
        red_match = re.search(r"Redundancy.*?:\s*([\d\.]+)", block)

        if triple_match and rank_match:
            triple = triple_match.groups()
            triple_string = " | ".join(triple)

            triples.append(triple_string)
            ranks.append(int(rank_match.group(1)))
            mrrs.append(float(mrr_match.group(1)) if mrr_match else None)
            redundancy.append(float(red_match.group(1)) if red_match else None)

    df = pd.DataFrame({
        "triple": triples,
        "rank": ranks,
        "MRR": mrrs,
        "redundancy": redundancy
    })

    # Extract tail entity for merge
    df["answer_entity"] = df["triple"].apply(lambda x: x.split(" | ")[-1].strip())

    return df


# ============================================================
# Parse LLM Experiment Output
# ============================================================

def parse_llm_output(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("----------------------------------------")

    for block in blocks:

        query_match = re.search(r"QUERY:\s*(.+)", block)
        model_match = re.search(r"MODEL:\s*(.+)", block)
        result_match = re.search(r"RESULT:\s*(YES|NO)", block)
        expected_match = re.search(r"EXPECTED:\s*(.+)", block)
        visited_nodes_match = re.search(r"VISITED NODES:\s*\[(.*?)\]", block)

        if query_match and model_match and result_match and expected_match:

            visited_count = 0
            if visited_nodes_match:
                visited_list = visited_nodes_match.group(1)
                if visited_list.strip():
                    visited_count = len(visited_list.split(","))

            rows.append({
                "query": query_match.group(1),
                "model": model_match.group(1),
                "correct": 1 if result_match.group(1) == "YES" else 0,
                "answer_entity": expected_match.group(1).strip(),
                "visited_nodes": visited_count
            })

    return pd.DataFrame(rows)


# ============================================================
# Advanced Analysis + Plotting
# ============================================================

def advanced_analysis(merged_df):

    print("\n===== ADVANCED ANALYSIS =====\n")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    subfolders = ["accuracy", "redundancy", "hop_depth", "correlations"]
    for sub in subfolders:
        os.makedirs(os.path.join(PLOTS_DIR, sub), exist_ok=True)

    # ------------------------------
    # Hop depth summary
    # ------------------------------

    hop_summary = merged_df.groupby("model")["visited_nodes"].mean()
    print("Average hop depth per model:")
    print(hop_summary)
    print()

    hop_summary.to_csv(os.path.join(TABLE_DIR, "avg_hop_depth_per_model.csv"))

    if len(merged_df) > 1:
        corr_hd, p_hd = pearsonr(
            merged_df["visited_nodes"],
            merged_df["correct"]
        )
        print(f"Correlation (Hop Depth vs Accuracy): {corr_hd:.4f}")
        print(f"P-value: {p_hd:.6f}")
        print()

    # ------------------------------
    # PLOTS
    # ------------------------------

    # Accuracy per model
    plt.figure(figsize=(8,5))
    sns.barplot(data=merged_df, x="model", y="correct", estimator=np.mean)
    plt.xticks(rotation=45, ha="right")
    plt.title("Accuracy per Model")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy", "accuracy_per_model.png"))
    plt.close()

    # Redundancy vs Accuracy
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=merged_df, x="redundancy", y="correct", hue="model")
    plt.title("Redundancy vs Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlations", "redundancy_vs_accuracy.png"))
    plt.close()

    # Rank bucket
    merged_df["rank_bucket"] = pd.cut(
        merged_df["rank"],
        bins=[0,1,3,10,50,1000],
        labels=["1","2-3","4-10","11-50","50+"]
    )

    plt.figure(figsize=(6,5))
    sns.barplot(data=merged_df, x="rank_bucket", y="correct", estimator=np.mean)
    plt.title("Accuracy by Rank Bucket")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "redundancy", "accuracy_by_rank_bucket.png"))
    plt.close()

    # Hop distribution
    plt.figure(figsize=(6,5))
    sns.histplot(merged_df["visited_nodes"], bins=10, kde=True)
    plt.title("Hop Depth Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hop_depth", "hop_depth_distribution.png"))
    plt.close()

    # Hop vs Accuracy
    plt.figure(figsize=(6,5))
    sns.boxplot(data=merged_df, x="correct", y="visited_nodes")
    plt.title("Hop Depth vs Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hop_depth", "hop_depth_vs_accuracy.png"))
    plt.close()

    # Redundancy vs Hop depth
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=merged_df, x="redundancy", y="visited_nodes", hue="model")
    plt.title("Redundancy vs Hop Depth")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlations", "redundancy_vs_hop_depth.png"))
    plt.close()

    print("Plots saved in:", OUTPUT_ROOT)


# ============================================================
# Main
# ============================================================

def main():

    print("Parsing KG redundancy output...")
    kf_df = parse_kf_output(KF_FILE)
    kf_df.to_csv(OUTPUT_KF_CSV, index=False)

    print("Parsing LLM experiment output...")
    llm_df = parse_llm_output(EXPERIMENT_FILE)
    llm_df.to_csv(OUTPUT_LLM_CSV, index=False)

    print("Merging datasets...")
    merged_df = pd.merge(
        llm_df,
        kf_df,
        on="answer_entity",
        how="inner"
    )

    merged_df.to_csv(OUTPUT_MERGED_CSV, index=False)

    if merged_df.empty:
        print("ERROR: Merged dataframe is empty.")
        print("Check answer_entity alignment.")
        return

    print("\n===== SUMMARY =====\n")

    print("Accuracy per model:")
    print(merged_df.groupby("model")["correct"].mean())
    print()

    if len(merged_df) > 1:
        corr, pval = pearsonr(
            merged_df["redundancy"],
            merged_df["correct"]
        )
        print(f"Correlation (Redundancy vs Accuracy): {corr:.4f}")
        print(f"P-value: {pval:.6f}")
        print()

    if SKLEARN_AVAILABLE:
        print("Running logistic regression...")
        X = merged_df[["redundancy"]]
        y = merged_df["correct"]
        model = LogisticRegression()
        model.fit(X, y)
        print(f"Logistic coefficient (redundancy): {model.coef_[0][0]:.4f}")
        print()

    advanced_analysis(merged_df)

    print("Analysis complete.")
    print("CSV files saved:")
    print(" -", OUTPUT_LLM_CSV)
    print(" -", OUTPUT_KF_CSV)
    print(" -", OUTPUT_MERGED_CSV)


if __name__ == "__main__":
    main()
