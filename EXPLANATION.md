# Explanation for Fixing `cluster_plot.py`

## 1. What was wrong with the original script and how I fixed it

The intention of the demo script `cluster_plot.py` is to run k-means clustering
for **k = 2, 3, 4, and 5**, generate clustered 2D plots for each value of *k*,
and save all results inside `demo_output/`.

However, inside the loop over the values `(2, 3, 4, 5)`, the script contained
the following line:

```python
result = run_clustering(
    input_path=input_path,
    feature_cols=feature_cols,
    algorithm="kmeans",
    k=min(k, 3),
    ...
)

The expression min(k, 3) meant that:

for k = 2 → clustering used 2 clusters (correct)

for k = 3 → clustering used 3 clusters (correct)

for k = 4 → clustering still used 3 clusters (incorrect)

for k = 5 → clustering still used 3 clusters (incorrect)

Therefore, although the script printed “Running k-means with k = 4” and
“Running k-means with k = 5”, the underlying analysis continued to use
only 3 clusters, producing misleading results and inconsistent output.

Fix

I replaced:
k = min(k, 3)

with:
k = k

so that the script truly performs clustering with k = 2, 3, 4, and 5 as intended.

Additional Improvements and Required User-Facing Messages

To satisfy the user interaction requirements from the marking criteria and the
extra exam instructions, I added:

A clear introductory message explaining what the script does.

A reminder to the user that the model used is k-means clustering.

A preview of the dataset with df.head() after reading the CSV file.

Messages describing what is being computed at each step (loading data,
preprocessing, clustering, plotting, saving).

A corrected usage message:
print("Usage: python demo/cluster_plot.py <input_csv>")
replacing the outdated name in the original script.

These changes improve clarity, transparency, and usability as required.

## 2. Summary of What the Corrected Script Now Does

The corrected script performs the following steps:

Prints an introductory message describing the purpose of the demo.

Reminds the user that k-means clustering is the algorithm being used.

Reads the input CSV and displays the first five rows using df.head().

Automatically selects the first two numeric columns for 2D clustering.

For each k in {2, 3, 4, 5}:

Runs the clustering algorithm via run_clustering.

Saves a clustered 2D plot for that value of k.

Writes labelled data to a CSV if an output path is specified.

Prints evaluation metrics (inertia and silhouette score).

Summarises all metrics across the four clustering runs into a CSV file.

Generates and saves a bar plot of silhouette score vs k (if silhouette score is defined).

Prints a final message indicating that the demo is complete and where outputs are saved.

The script now behaves fully as described in its header comments and produces
consistent, correct, and interpretable outputs.

## 3. Overview of the cluster_maker Package and the Purpose of Its Main Components

The cluster_maker package provides a complete, modular clustering pipeline.
Its components each serve a specific purpose in the workflow:

1. preprocessing.py

select_features verifies that requested feature columns exist and are numeric.

standardise_features scales numeric data to zero mean and unit variance
using StandardScaler.

This ensures that the clustering operates on valid, comparable input data.

2. algorithms.py

Implements the full manual K-means algorithm:

init_centroids selects initial centroids.

assign_clusters assigns each point to its nearest centroid.

update_centroids recomputes centroid positions.

kmeans runs the full iterative algorithm.

Provides sklearn_kmeans, a wrapper around scikit-learn’s KMeans.

This module performs the actual clustering computations.

3. evaluation.py

compute_inertia computes within-cluster sum of squared distances.

silhouette_score_sklearn computes silhouette score for cluster evaluation.

elbow_curve computes inertia values for different k values for the elbow method.

These tools evaluate how good a clustering result is.

4. plotting_clustered.py

plot_clusters_2d visualises clustered data in a 2D scatter plot.

plot_elbow visualises inertia vs k for elbow analysis.

This module handles visualisation of clustering results.

5. interface.py

Defines run_clustering, a high-level function that performs the entire workflow:

Load data

Select features

Standardise features

Run chosen clustering algorithm

Compute metrics

Produce plots

Optionally export data

This module acts as the main user-facing API for performing clustering.

6. data_exporter.py

Exports DataFrames (including cluster labels) to CSV files.

7. data_analyser.py

Contains additional helper tools for inspecting and analysing datasets and clusters.

Overall Structure

Together, these modules implement a coherent and well-organised clustering
pipeline. Each stage of the analysis — from feature selection to evaluation
and plotting — is encapsulated cleanly within the appropriate module,
allowing the demo scripts (such as cluster_plot.py) to remain clean,
readable, and focused on workflow rather than implementation details.