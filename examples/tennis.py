"""Small categorical example inspired by the tennis example in the course."""
from entropy_decision_tree import EntropyDecisionTree

X = [
    ["Sunny", "Hot", "High", "Weak"],
    ["Sunny", "Hot", "High", "Strong"],
    ["Overcast", "Hot", "High", "Weak"],
    ["Rain", "Mild", "High", "Weak"],
    ["Rain", "Cool", "Normal", "Weak"],
    ["Rain", "Cool", "Normal", "Strong"],
    ["Overcast", "Cool", "Normal", "Strong"],
    ["Sunny", "Mild", "High", "Weak"],
    ["Sunny", "Cool", "Normal", "Weak"],
    ["Rain", "Mild", "Normal", "Weak"],
    ["Sunny", "Mild", "Normal", "Strong"],
    ["Overcast", "Mild", "High", "Strong"],
    ["Overcast", "Hot", "Normal", "Weak"],
    ["Rain", "Mild", "High", "Strong"],
]
y = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
     "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

model = EntropyDecisionTree(max_depth=3)
model.fit(X, y)
print("Tree:")
print(model.export_text(["Outlook", "Temperature", "Humidity", "Wind"]))
print("\\nPrediction for Sunny, Cool, High, Strong:",
      model.predict([["Sunny", "Cool", "High", "Strong"]])[0])
