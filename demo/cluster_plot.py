###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    # Introductory message
    print("Welcome to the cluster_maker clustering demo.")
    print("This script performs k-means clustering for k = 2, 3, 4, 5 on a 2D dataset.")
    print("Models used in this demo: k-means clustering (via cluster_maker).")

    if len(args) != 1:
        print("Usage: python demo/cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)

    # Show sample of dataset, as requested
    print("\nHere is a sample of the dataset (first 5 rows):")
    print(df.head())

    # The CSV is assumed to have two or more data columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)

    feature_cols = numeric_cols[:2]  # Use the first two numeric columns
    print(f"\nUsing the following numeric columns for clustering: {feature_cols}")

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Main job: run clustering for k = 2, 3, 4, 5
    metrics_summary = []

    print("\nStarting k-means clustering for k = 2, 3, 4, 5...")
    for k in (2, 3, 4, 5):
        print(f"\n=== Running k-means with k = {k} ===")
        print("Computing clusters and generating clustered plot...")

        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,  # FIX: use the actual k, not min(k, 3)
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False,  # no elbow diagram
        )

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])
        print(f"Cluster plot saved to: {plot_path}")

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics for this value of k:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics across k
    print("\nSummarising metrics across all k values...")
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics summary saved to: {metrics_csv}")

    # Plot some statistics (no elbow: avoid inertia vs k)
    if "silhouette_score" in metrics_df.columns:
        print("Generating silhouette score summary plot...")
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette_score"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()
        print(f"Silhouette summary plot saved to: {stats_path}")

    print("\nDemo completed.")
    print(f"All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
