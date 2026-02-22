import numpy as np
import pandas as pd

from voice_screening.clustering import ClusterAggregator


def test_cluster_aggregation_mean_shape() -> None:
    x = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [2.0, 3.0, 4.0],
            "f3": [10.0, 9.0, 8.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "cluster": [1, 1, 2],
        }
    )

    agg = ClusterAggregator(assignments, method="mean")
    out, names = agg.fit_transform(x)

    assert out.shape == (3, 2)
    assert names == ["cluster_1", "cluster_2"]
    assert np.allclose(out[:, 0], np.array([1.5, 2.5, 3.5]))
