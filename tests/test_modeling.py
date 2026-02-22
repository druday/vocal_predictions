import pytest

from voice_screening.modeling import build_model


@pytest.mark.parametrize(
    ("model_name", "expected_class"),
    [
        ("logistic_regression", "LogisticRegression"),
        ("elastic_net_logistic", "SGDClassifier"),
        ("random_forest", "RandomForestClassifier"),
        ("extra_trees", "ExtraTreesClassifier"),
        ("hist_gradient_boosting", "HistGradientBoostingClassifier"),
        ("svc_rbf", "SVC"),
        ("mlp", "TorchRegularizedHybridNetClassifier"),
        ("residual_mlp", "TorchResidualHybridNetClassifier"),
        ("wide_deep_mlp", "TorchWideDeepNetClassifier"),
    ],
)
def test_build_model_supported(model_name: str, expected_class: str) -> None:
    model = build_model(model_name, hyperparams={}, random_seed=42)
    assert type(model).__name__ == expected_class


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_model("unknown_model", hyperparams={}, random_seed=42)
