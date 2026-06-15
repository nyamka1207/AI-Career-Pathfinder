import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create synthetic training data
np.random.seed(42)

# Define careers
careers = ["Software Engineer", "Data Scientist", "Graphic Designer", "Business Manager", "General Specialist"]

# Generate synthetic training data with 100 samples per career
X_data = []
y_data = []

# Software Engineer: high coding, high math, medium problem solving
for _ in range(100):
    X_data.append([
        np.random.randint(6, 10),      # math
        np.random.randint(8, 10),      # coding
        np.random.randint(2, 6),       # creativity
        np.random.randint(3, 7),       # communication
        np.random.randint(3, 7),       # leadership
        np.random.randint(7, 10)       # problem solving
    ])
    y_data.append("Software Engineer")

# Data Scientist: high math, high coding, high problem solving
for _ in range(100):
    X_data.append([
        np.random.randint(8, 10),      # math
        np.random.randint(7, 10),      # coding
        np.random.randint(2, 6),       # creativity
        np.random.randint(3, 7),       # communication
        np.random.randint(2, 6),       # leadership
        np.random.randint(8, 10)       # problem solving
    ])
    y_data.append("Data Scientist")

# Graphic Designer: high creativity, medium communication
for _ in range(100):
    X_data.append([
        np.random.randint(2, 6),       # math
        np.random.randint(3, 7),       # coding
        np.random.randint(8, 10),      # creativity
        np.random.randint(7, 10),      # communication
        np.random.randint(3, 7),       # leadership
        np.random.randint(3, 7)        # problem solving
    ])
    y_data.append("Graphic Designer")

# Business Manager: high communication, high leadership
for _ in range(100):
    X_data.append([
        np.random.randint(4, 8),       # math
        np.random.randint(2, 6),       # coding
        np.random.randint(2, 6),       # creativity
        np.random.randint(8, 10),      # communication
        np.random.randint(8, 10),      # leadership
        np.random.randint(5, 8)        # problem solving
    ])
    y_data.append("Business Manager")

# General Specialist: balanced skills
for _ in range(100):
    X_data.append([
        np.random.randint(4, 7),       # math
        np.random.randint(4, 7),       # coding
        np.random.randint(4, 7),       # creativity
        np.random.randint(4, 7),       # communication
        np.random.randint(4, 7),       # leadership
        np.random.randint(4, 7)        # problem solving
    ])
    y_data.append("General Specialist")

# Train k-NN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_data, y_data)

# Save model
with open("career_knn.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as career_knn.pkl")
