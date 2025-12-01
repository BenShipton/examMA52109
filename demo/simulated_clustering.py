###
## cluster_maker: simulated clustering demo
## MA52109 – Programming for Data Science
## This script analyses simulated_data.csv using k-means clustering
## and determines a plausible separation of the data into clusters.
###

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "simulated_output"


def main(args: List[str]) -> None:
    # Introductory messages
    print("Welcome to the simulated clustering demo.")
    print("This script analyses a simulated dataset and searches for a plausible")
    print("number of clusters using k-means clustering.")
    print("Model used in this analysis: k-means clustering (via cluster_maker).")

    # --- Handle command-line arguments ---
    if len(args) != 1:
        print("Usage: python demo/simulated_clustering.py <input_csv>")
        print("Example: python demo/simulated_clustering.py data/simulated_data.csv")
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

    # Use all numeric columns as features for clustering, but plots will use first two
    feature_cols = numeric_cols
    base = os.path.splitext(os.path.basename(input_path))[0]

    # --- Explore different k values to find a plausible number of clusters ---
    candidate_ks = [2, 3, 4, 5, 6]
    metrics_summary: List[Dict[str, Any]] = []

    print("\nExploring different values of k to find a good clustering...")
    for k in candidate_ks:
        print(f"\n=== Running k-means with k = {k} ===")
        print("Computing clusters, metrics, and a 2D cluster plot...")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=40,
            compute_elbow=False,
        )

        # Save cluster plot produced by cluster_maker
        fig_cluster = result["fig_cluster"]
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_clusters_k{k}.png")
        fig_cluster.savefig(plot_path, dpi=150)
        plt.close(fig_cluster)
        print(f"Cluster plot saved to: {plot_path}")

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics for this value of k:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # --- Summarise metrics across candidate k values ---
    print("\nSummarising metrics across all k values...")
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_simulated_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics summary saved to: {metrics_csv}")

    # Choose a plausible k based on silhouette score (higher is better)
    k_best: Optional[int] = None
    if "silhouette" in metrics_df.columns:
        valid = metrics_df["silhouette"].notna()
        if valid.any():
            best_idx = metrics_df.loc[valid, "silhouette"].idxmax()
            k_best = int(metrics_df.loc[best_idx, "k"])
            best_score = float(metrics_df.loc[best_idx, "silhouette"])
            print(f"\nBased on silhouette score, a plausible number of clusters is k = {k_best}.")
            print(f"Best silhouette score: {best_score:.3f}")
        else:
            print("\nSilhouette score was not defined for any k (e.g. only one cluster).")
    else:
        print("\nSilhouette score was not available in the metrics.")

    # Fallback if no silhouette information is available
    if k_best is None:
        k_best = 3
        print(f"\nDefaulting to k = {k_best} as a reasonable choice.")

    # --- Optional: Compute an elbow curve to support the choice of k ---
    print("\nGenerating elbow curve for additional validation...")

    elbow_k_values = candidate_ks  # reuse same ks as silhouette evaluation

    elbow_result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=k_best,                     # used only for labelling
        standardise=True,
        random_state=42,
        compute_elbow=True,
        elbow_k_values=elbow_k_values,
    )

    fig_elbow = elbow_result["fig_elbow"]
    elbow_plot_path = os.path.join(OUTPUT_DIR, f"{base}_elbow_curve.png")
    fig_elbow.savefig(elbow_plot_path, dpi=150)
    plt.close(fig_elbow)

    print(f"Elbow curve saved to: {elbow_plot_path}")

    # --- Run a final clustering with the chosen k and create extra visualisations ---
    print(f"\nRunning final clustering with k = {k_best} for detailed visualisation...")
    final_result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=k_best,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k_best.csv"),
        random_state=42,
        compute_elbow=False,
    )

    final_df: pd.DataFrame = final_result["data"]
    final_labels = final_result["labels"]

    # Save the final cluster plot again with a clear name
    final_fig = final_result["fig_cluster"]
    final_plot_path = os.path.join(OUTPUT_DIR, f"{base}_clusters_k_best.png")
    final_fig.savefig(final_plot_path, dpi=150)
    plt.close(final_fig)
    print(f"Final cluster plot saved to: {final_plot_path}")

        # --- Additional visualisation: Feature 1 vs Feature 3 ---
    if len(feature_cols) >= 3:
        print("Generating Feature 1 vs Feature 3 cluster plot...")

        # Extract the two features manually (standardised values already in X)
        X_final = final_df[feature_cols].to_numpy()
        x1 = X_final[:, 0]   # Feature 1
        x3 = X_final[:, 2]   # Feature 3

        plt.figure()
        scatter = plt.scatter(x1, x3, c=final_labels, cmap="tab10", alpha=0.9)

        # Plot centroids (already in standardised space)
        centroids = final_result["centroids"]
        plt.scatter(
            centroids[:, 0],
            centroids[:, 2],
            marker="h",
            s=200,
            linewidths=2,
            edgecolor="black",
            c=range(k_best),
            cmap="tab10",
            label="Centroids",
        )
        plt.legend()

        plt.xlabel(f"{feature_cols[0]} (Feature 1)")
        plt.ylabel(f"{feature_cols[2]} (Feature 3)")
        plt.title(f"Cluster plot: Feature 1 vs Feature 3 (k = {k_best})")
        plt.tight_layout()

        f13_plot_path = os.path.join(OUTPUT_DIR, f"{base}_feature1_vs_feature3_k_best.png")
        plt.savefig(f13_plot_path, dpi=150)
        plt.close()

        print(f"Feature 1 vs Feature 3 cluster plot saved to: {f13_plot_path}")
    else:
        print("Dataset has fewer than 3 numeric features — skipping Feature 1 vs Feature 3 plot.")


    # --- Supporting visualisation 1: silhouette vs k (if available) ---
    if "silhouette" in metrics_df.columns:
        print("Generating silhouette score vs k plot...")
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"])
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different values of k")
        plt.tight_layout()
        sil_plot_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette_vs_k.png")
        plt.savefig(sil_plot_path, dpi=150)
        plt.close()
        print(f"Silhouette summary plot saved to: {sil_plot_path}")

    # --- Supporting visualisation 2: cluster size distribution ---
    print("Generating cluster size distribution plot for the chosen k...")
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    plt.figure()
    plt.bar(unique_labels, counts, tick_label=unique_labels)
    plt.xlabel("Cluster label")
    plt.ylabel("Number of samples")
    plt.title(f"Cluster size distribution (k = {k_best})")
    plt.tight_layout()
    size_plot_path = os.path.join(OUTPUT_DIR, f"{base}_cluster_sizes_k_best.png")
    plt.savefig(size_plot_path, dpi=150)
    plt.close()
    print(f"Cluster size distribution plot saved to: {size_plot_path}")

    # --- Supporting visualisation 3: pairwise scatter plots of all numeric features ---
    print("Generating pairwise scatter plots for all numeric features...")
    X_all = final_df[feature_cols].to_numpy()
    n_features = len(feature_cols)

    fig, axes = plt.subplots(
        n_features, n_features,
        figsize=(3 * n_features, 3 * n_features)
    )

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram of each feature
                ax.hist(X_all[:, j], bins=20)
                ax.set_ylabel(feature_cols[i])
            else:
                # Off-diagonal: scatter plot coloured by cluster label
                ax.scatter(
                    X_all[:, j],
                    X_all[:, i],
                    c=final_labels,
                    cmap="tab10",
                    s=10,
                    alpha=0.8,
                )

            if i == n_features - 1:
                ax.set_xlabel(feature_cols[j])
            if j == 0:
                ax.set_ylabel(feature_cols[i])

    fig.suptitle(f"Pairwise feature scatter plots coloured by cluster (k = {k_best})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pair_plot_path = os.path.join(OUTPUT_DIR, f"{base}_pairwise_features_k_best.png")
    fig.savefig(pair_plot_path, dpi=150)
    plt.close(fig)
    print(f"Pairwise feature scatter plot grid saved to: {pair_plot_path}")

    print("\nAnalysis completed.")
    print(f"A plausible separation of the data is into {k_best} clusters,")
    print("based on clustering metrics, the elbow curve, and inspection of the")
    print("resulting cluster plots and pairwise feature projections.")
    print("All outputs have been saved in:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
