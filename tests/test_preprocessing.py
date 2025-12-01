import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # 1) This test checks that select_features correctly raises an error
    #    when a requested column is missing. In real workflows, a typo
    #    or schema change can silently corrupt downstream clustering if
    #    missing columns are not detected early.
    def test_select_features_missing_column(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with self.assertRaises(KeyError):
            select_features(df, ["A", "C"])   # 'C' does not exist


    # 2) This test checks that select_features rejects non-numeric columns.
    #    If non-numeric data passes through, standardisation and KMeans
    #    would both fail, so this protects the pipeline from invalid input.
    def test_select_features_non_numeric_column(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": ["hello", "world", "test"]   # non-numeric
        })
        with self.assertRaises(TypeError):
            select_features(df, ["x", "y"])   # 'y' is not numeric


    # 3) This test ensures that standardise_features produces zero-mean,
    #    unit-variance features. If this step is incorrect, clustering
    #    performance becomes distorted (features with larger scales dominate).
    def test_standardise_features_zero_mean_unit_variance(self):
        X = np.array([[10.0, 100.0],
                      [12.0, 110.0],
                      [14.0, 90.0]])
        
        X_scaled = standardise_features(X)

        means = X_scaled.mean(axis=0)
        stds = X_scaled.std(axis=0)

        # Means approximately zero
        self.assertTrue(np.allclose(means, [0.0, 0.0], atol=1e-7))

        # Standard deviations approximately one
        self.assertTrue(np.allclose(stds, [1.0, 1.0], atol=1e-7))


if __name__ == "__main__":
    unittest.main()
