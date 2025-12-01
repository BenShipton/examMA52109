###
## cluster_maker
## Agglomerative (hierarchical) clustering module
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run agglomerative (hierarchical) clustering on the given data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature data to cluster. Assumed to be numeric and, typically,
        already standardised.
    n_clusters : int, default 3
        The number of clusters to form.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Which linkage criterion to use.
    metric : str, default "euclidean"
        Distance metric to use. For linkage="ward", the metric is forced
        to "euclidean" by scikit-learn.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    centroids : ndarray of shape (n_clusters, n_features)
        Centroid of each cluster, computed as the mean of the points in
        that cluster. These pseudo-centroids are used for plotting and
        inertia computation, to integrate with the rest of the package.
    """
    X = np.asarray(X, dtype=float)

    # For "ward" linkage, scikit-learn expects Euclidean distance and
    # ignores the metric argument.
    if linkage == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )

    labels = model.fit_predict(X)

    # Compute pseudo-centroids as the mean of each cluster so that
    # evaluation and plotting functions can be reused.
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=float)
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            # Very unlikely, but handle just in case
            centroids[cluster_id] = X.mean(axis=0)

    return labels, centroids
