"""A small decision tree classifier implemented from scratch using entropy."""
from collections import Counter
import math
import numbers


def entropy(y):
    """Compute Shannon entropy (base 2) for a sequence of class labels."""
    y = list(y)
    if not y:
        return 0.0
    counts = Counter(y)
    n = len(y)
    return -sum((count / n) * math.log2(count / n)
                for count in counts.values() if count)


class _Node:
    def __init__(self, prediction, samples, class_counts, feature=None,
                 threshold=None, category=None, children=None):
        self.prediction = prediction
        self.samples = samples
        self.class_counts = dict(class_counts)
        self.feature = feature
        self.threshold = threshold
        self.category = category
        self.children = children or {}

    @property
    def is_leaf(self):
        return self.feature is None


class EntropyDecisionTree:
    """
    Decision tree classifier using information gain.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum number of split levels. None grows until leaves are pure or
        no positive-gain split remains. A value of 0 creates a root-only tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_gain : float, default=1e-12
        Minimum information gain required to accept a split.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=1e-12):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root_ = None
        self.n_features_in_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = self._rows(X)
        y = list(y)
        if not X or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have equal lengths")
        width = len(X[0])
        if any(len(row) != width for row in X):
            raise ValueError("All rows in X must have the same number of features")
        if any(value is None for row in X for value in row):
            raise ValueError("Missing values (None) are not supported")
        self.n_features_in_ = width
        self.classes_ = list(dict.fromkeys(y))
        self.root_ = self._build(X, y, depth=0)
        return self

    def predict(self, X):
        self._check_fitted()
        rows = self._rows(X)
        return [self._predict_one(row, self.root_) for row in rows]

    def score(self, X, y):
        y = list(y)
        predictions = self.predict(X)
        if len(y) != len(predictions) or not y:
            raise ValueError("X and y must have equal, non-zero lengths")
        return sum(a == b for a, b in zip(predictions, y)) / len(y)

    def _build(self, X, y, depth):
        counts = Counter(y)
        prediction = counts.most_common(1)[0][0]
        node = _Node(prediction, len(y), counts)
        if len(counts) == 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        split = self._best_split(X, y)
        if split is None or split[0] <= self.min_gain:
            return node
        gain, feature, kind, split_value, partitions = split
        node.feature = feature
        if kind == "numeric":
            node.threshold = split_value
            for branch, indices in partitions.items():
                node.children[branch] = self._build(
                    [X[i] for i in indices], [y[i] for i in indices], depth + 1)
        else:
            for category, indices in partitions.items():
                node.children[category] = self._build(
                    [X[i] for i in indices], [y[i] for i in indices], depth + 1)
        return node

    def _best_split(self, X, y):
        parent_entropy = entropy(y)
        n = len(y)
        best = None
        n_features = len(X[0])
        for feature in range(n_features):
            values = [row[feature] for row in X]
            # Numeric feature: test midpoints between sorted unique values.
            if all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values):
                unique = sorted(set(values))
                candidates = [(a + b) / 2 for a, b in zip(unique, unique[1:])]
                for threshold in candidates:
                    left = [i for i, v in enumerate(values) if v <= threshold]
                    right = [i for i, v in enumerate(values) if v > threshold]
                    if not left or not right:
                        continue
                    gain = parent_entropy - (
                        len(left) / n * entropy(y[i] for i in left)
                        + len(right) / n * entropy(y[i] for i in right))
                    candidate = (gain, feature, "numeric", threshold,
                                 {"left": left, "right": right})
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            else:
                groups = {}
                for i, value in enumerate(values):
                    groups.setdefault(value, []).append(i)
                if len(groups) < 2:
                    continue
                weighted_child_entropy = sum(
                    len(indices) / n * entropy(y[i] for i in indices)
                    for indices in groups.values())
                gain = parent_entropy - weighted_child_entropy
                candidate = (gain, feature, "categorical", None, groups)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best

    def _predict_one(self, row, node):
        while not node.is_leaf:
            value = row[node.feature]
            if node.threshold is not None:
                branch = "left" if value <= node.threshold else "right"
                node = node.children[branch]
            else:
                # Unknown categories fall back to this node's majority class.
                if value not in node.children:
                    return node.prediction
                node = node.children[value]
        return node.prediction

    def export_text(self, feature_names=None):
        """Return a readable, indented representation of the learned tree."""
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names length must match number of features")
        lines = []

        def walk(node, prefix="", edge="ROOT"):
            if node.is_leaf:
                lines.append(f"{prefix}{edge} -> predict {node.prediction!r} "
                             f"(samples={node.samples}, counts={node.class_counts})")
                return
            if node.threshold is not None:
                lines.append(f"{prefix}{edge}: {feature_names[node.feature]} "
                             f"(threshold={node.threshold:g})")
                walk(node.children["left"], prefix + "  ", f"<= {node.threshold:g}")
                walk(node.children["right"], prefix + "  ", f"> {node.threshold:g}")
            else:
                lines.append(f"{prefix}{edge}: {feature_names[node.feature]}")
                for category, child in node.children.items():
                    walk(child, prefix + "  ", f"== {category!r}")

        walk(self.root_)
        return "\n".join(lines)

    def _check_fitted(self):
        if self.root_ is None:
            raise RuntimeError("Call fit(X, y) before prediction or display")

    @staticmethod
    def _rows(X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return [list(row) for row in X]
