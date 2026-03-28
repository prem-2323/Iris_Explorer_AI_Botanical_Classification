# Step 1 — Import libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 2 — Load Iris dataset
iris = load_iris()
X = iris.data   # features
y = iris.target # labels

# Step 3 — Train model
model = RandomForestClassifier()
model.fit(X, y)

# Step 4 — Save model as .pkl file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as iris_model.pkl")