###
## cluster_maker: agglomerative clustering demo
## MA52109 – Programming for Data Science
## This script applies hierarchical (agglomerative) clustering to
## difficult_dataset.csv and demonstrates that it can produce
## sensible clustering structure on a challenging dataset.
###

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "agglomerative_output"


def main(args: List[str]) -> None:
    # Introductory messages
    print("Welcome to the agglomerative clustering demo.")
    print("This script analyses a difficult dataset using hierarchical")
    print("agglomerative clustering to find a sensible separation into clusters.")
    print("Model used in this analysis: AgglomerativeClustering (via cluster_maker).")

    # --- Handle command-line arguments ---
    if len(args) != 1:
        print("Usage: python demo/demo_agglomerative.py <input_csv>")
        print("Example: python demo/demo_agglomerative.py data/difficult_dataset.csv")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load dataset and show a sample ---
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("\nHere is a sample of the dataset (first 5 rows):")
    print(df.head())

    # Automatically detect numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The dataset must contain at least two numeric columns.")
        sys.exit(1)

    print(f"\nDetected numeric columns: {numeric_cols}")

    feature_cols = numeric_cols
    base = os.path.splitext(os.path.basename(input_path))[0]

    # --- Explore a range of k values with agglomerative clustering ---
    candidate_ks = [2, 3, 4, 5]
    metrics_summary: List[Dict[str, Any]] = []

    print("\nExploring different values of k using agglomerative clustering...")
    for k in candidate_ks:
        print(f"\n=== Running agglomerative clustering with k = {k} ===")
        print("Computing clusters, metrics, and a 2D cluster plot...")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="agglomerative",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_agglomerative_k{k}.csv"),
            random_state=None,      # AgglomerativeClustering is deterministic
            compute_elbow=False,
        )

        fig_cluster = result["fig_cluster"]
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_clusters_k{k}.png")
        fig_cluster.savefig(plot_path, dpi=150)
        plt.close(fig_cluster)
        print(f"Cluster plot saved to: {plot_path}")

        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics for this value of k:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # --- Summarise metrics and choose a plausible k ---
    print("\nSummarising metrics across all k values...")
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics summary saved to: {metrics_csv}")

    k_best: Optional[int] = None
    if "silhouette" in metrics_df.columns:
        valid = metrics_df["silhouette"].notna()
        if valid.any():
            best_idx = metrics_df.loc[valid, "silhouette"].idxmax()
            k_best = int(metrics_df.loc[best_idx, "k"])
            best_sil = float(metrics_df.loc[best_idx, "silhouette"])
            print(f"\nBased on silhouette score, a plausible number of clusters is k = {k_best}.")
            print(f"Best silhouette score: {best_sil:.3f}")
        else:
            print("\nSilhouette score was not defined for any k.")
    else:
        print("\nSilhouette score was not available in the metrics.")

    if k_best is None:
        k_best = 3
        print(f"\nDefaulting to k = {k_best} as a reasonable choice for this dataset.")

    # --- Final detailed run with chosen k ---
    print(f"\nRunning final agglomerative clustering with k = {k_best}...")
    final_result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="agglomerative",
        k=k_best,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, f"{base}_agglomerative_k_best.csv"),
        random_state=None,
        compute_elbow=False,
    )

    final_df: pd.DataFrame = final_result["data"]
    final_labels = final_result["labels"]
    final_fig = final_result["fig_cluster"]

    final_plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_clusters_k_best.png")
    final_fig.savefig(final_plot_path, dpi=150)
    plt.close(final_fig)
    print(f"Final agglomerative cluster plot saved to: {final_plot_path}")

    # --- Additional visual: silhouette vs k for agglomerative ---
    if "silhouette" in metrics_df.columns:
        print("Generating silhouette score vs k plot for agglomerative clustering...")
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"])
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score")
        plt.title("Agglomerative clustering: silhouette score vs k")
        plt.tight_layout()
        sil_plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_silhouette_vs_k.png")
        plt.savefig(sil_plot_path, dpi=150)
        plt.close()
        print(f"Silhouette summary plot saved to: {sil_plot_path}")

    print("\nDemo completed.")
    print(f"A plausible separation of the difficult dataset is into {k_best} clusters,")
    print("found by agglomerative (hierarchical) clustering. This method is robust")
    print("on this dataset because it does not assume spherical clusters and can adapt")
    print("to more irregular shapes. All outputs have been saved in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
