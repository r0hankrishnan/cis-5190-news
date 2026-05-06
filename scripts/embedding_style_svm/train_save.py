import numpy as np
import torch
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_data
from pathlib import Path

CLEANED_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "combined_base_data.csv"

# Get data -- should work on our data since it has url and headline ("title")
X, y = prepare_data(csv_path = str(CLEANED_DATA_PATH))

# Fit scaler on data
print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM
print("Train SVM...")
svm = SVC(kernel = "rbf", C = 1.0, probability = False)
svm.fit(X_scaled, y)
print("SVM fit")

# Save
blob = pickle.dumps((scaler, svm))
torch.save({"sklearn_bytes": torch.frombuffer(blob, dtype = torch.uint8)}, "model.pt")
print("Saved model and scaler as svm_model.pt")

