# Entropy Decision Tree From Scratch

A small, dependency-free classification tree implemented from scratch with Shannon entropy and information gain. It does not use scikit-learn to train the tree.

## Features

- Shannon entropy using log base 2.
- Information gain to choose the best split greedily.
- Categorical features use multi-way splits.
- Numeric features use binary threshold splits.
- `max_depth`, `min_samples_split`, and `min_gain` controls.
- Majority-class predictions at non-pure leaves.
- Readable text-tree display with `export_text`.
- Tests and a tennis dataset example.

## Install locally

```bash
python -m pip install -e .
```

## Quick start

```python
from entropy_decision_tree import EntropyDecisionTree

X = [["Sunny"], ["Rain"], ["Sunny"], ["Rain"]]
y = ["No", "Yes", "No", "Yes"]

tree = EntropyDecisionTree(max_depth=3)
tree.fit(X, y)
print(tree.predict([["Sunny"], ["Rain"]]))
print(tree.export_text(["Outlook"]))
```

Input `X` should be a list/array of rows. A feature column should consistently be numeric or categorical. Missing values (`None`) are not supported. For unseen categorical values, prediction falls back to the majority class at the node.

## Run the example and tests

```bash
python examples/tennis.py
python -m pip install -e ".[test]"
pytest -q
```

## How it works

For class-label distribution \(Y\), entropy is

`H(Y) = -sum(p(c) * log2(p(c)))`

For a candidate split, weighted child entropy is

`H_after = sum((n_child / n_parent) * H(child))`

Information gain is

`IG = H(parent) - H_after`

At each node, the implementation evaluates candidate splits and chooses the one with the largest information gain. This is a greedy algorithm. Training stops when the node is pure, no split has gain above `min_gain`, fewer than `min_samples_split` examples remain, or `max_depth` is reached.

## PyPI packaging bonus

The project includes `pyproject.toml`, so it is configured for building a Python distribution. Publishing to PyPI is not automatic: create a PyPI account, choose a unique package name if this one is taken, build the distribution (`python -m pip install build && python -m build`), and upload with `python -m pip install twine && python -m twine upload dist/*`. Do not put API tokens in source control.
