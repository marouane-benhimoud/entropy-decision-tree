import math
import pytest
from entropy_decision_tree import EntropyDecisionTree, entropy


def test_entropy():
    assert entropy(["yes", "yes", "yes"]) == 0
    assert entropy(["yes", "no"]) == pytest.approx(1.0)
    assert entropy(["yes", "yes", "no", "no"]) == pytest.approx(1.0)


def test_categorical_fit_predict():
    X = [["sunny"], ["sunny"], ["rain"], ["rain"]]
    y = ["no", "no", "yes", "yes"]
    model = EntropyDecisionTree().fit(X, y)
    assert model.predict(X) == y
    assert model.score(X, y) == 1.0


def test_numeric_split():
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = ["low", "low", "high", "high"]
    model = EntropyDecisionTree().fit(X, y)
    assert model.predict([[1.5], [3.5]]) == ["low", "high"]


def test_max_depth_zero_is_majority_leaf():
    X = [["a"], ["b"], ["c"]]
    y = ["yes", "yes", "no"]
    model = EntropyDecisionTree(max_depth=0).fit(X, y)
    assert model.predict(X) == ["yes", "yes", "yes"]
    assert "predict" in model.export_text()


def test_unknown_category_falls_back_to_majority():
    model = EntropyDecisionTree().fit([["a"], ["a"], ["b"]], ["yes", "yes", "no"])
    assert model.predict([["unseen"]]) == ["yes"]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        EntropyDecisionTree().predict([[1]])
