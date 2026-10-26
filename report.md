# Decision Tree From Scratch — Report

## GitHub repository
**GitHub link:** [ADD YOUR GITHUB REPOSITORY URL HERE]

> Replace the placeholder above with the URL of the repository after pushing this project to GitHub. Do not submit the placeholder as the final link.

## Project objective
The objective is to implement a classification decision tree from scratch using **entropy** and **information gain**, without using a machine-learning library to build the tree. This follows the course material on decision trees, where entropy measures uncertainty and information gain measures the reduction in uncertainty after a split.

## Implementation
The implementation is provided in `entropy_decision_tree/tree.py`.

- **Entropy:** computes Shannon entropy with logarithm base 2. A pure set has entropy 0.
- **Information gain:** computes parent entropy minus the weighted sum of child entropies.
- **Split selection:** evaluates candidate splits and greedily selects the split with the largest information gain.
- **Categorical features:** split into one child for each observed category.
- **Numerical features:** evaluate thresholds between sorted unique values and create two branches (`<= threshold` and `> threshold`).
- **Prediction:** follows the learned branches. A leaf predicts its most frequent training class. If prediction encounters an unseen category, it uses the majority class at the current node.
- **Stopping rules:** stop when a node is pure, the depth limit is reached, there are too few samples to split, or no candidate split has information gain above `min_gain`.

## Hyperparameters
| Parameter | Purpose |
|---|---|
| `max_depth` | Maximum number of split levels. `None` allows the tree to grow until another stopping rule applies; `0` creates a root-only tree. |
| `min_samples_split` | Minimum number of samples required to consider splitting a node. |
| `min_gain` | Minimum information gain required to accept a split. |

The main requested size control is `max_depth`. Limiting depth can make the tree easier to interpret and can reduce overfitting.

## Bonus features
- `export_text(feature_names)` displays the learned tree as an indented text representation.
- `pyproject.toml` configures the project for Python package building and installation.
- A small tennis classification example and automated tests are included.

## Validation
The project includes tests for entropy, categorical classification, numeric thresholds, depth limiting, unseen categories, and calling prediction before fitting. Run:

```bash
python -m pip install -e ".[test]"
pytest -q
```

Run the demonstration with:

```bash
python examples/tennis.py
```

Record the actual test output here after running the tests in your environment.

## Limitations
This is a compact educational classifier, not a replacement for production-grade tree libraries. Missing values are not supported. Categorical splits are multi-way, and the greedy procedure is not guaranteed to find a globally optimal tree. Pruning and probability estimates are not implemented.
