
from entropy_decision_tree import EntropyDecisionTree, entropy

# Test entropy
print("Entropy:", entropy(["Yes", "No"]))

# Training data
X = [[1], [2], [3], [4], [5], [6]]
y = ["Low", "Low", "Low", "High", "High", "High"]

# Train the tree
tree = EntropyDecisionTree(max_depth=2)
tree.fit(X, y)

# Predict new values
print("Predictions:", tree.predict([[1.5], [5.5]]))

# Display the tree
print(tree.export_text(["Score"]))

# Evaluate accuracy on training data
print("Accuracy:", tree.score(X, y))
