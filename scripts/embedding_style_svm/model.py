import numpy as np
import torch
from sklearn.svm._classes import SVC
from sklearn.preprocessing._data import StandardScaler
from pathlib import Path
import sys
import pickle

sys.modules.setdefault("model", sys.modules[__name__])
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"

class NewsClassifier:
    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.svm: SVC | None = None
        if MODEL_PATH.exists():
            payload = torch.load(MODEL_PATH, map_location = "cpu", weights_only = False)
            self.load_state_dict(payload)
        
    def load_state_dict(self, state:dict):
        blob = state["sklearn_bytes"]
        if isinstance(blob, torch.Tensor):
            blob = blob.cpu().numpy().tobytes()
        self.scaler, self.svm = pickle.loads(blob)
        
    def predict(self, X) -> list[str]:
        X = np.array(X, dtype = float)
        X_scaled = self.scaler.transform(X)
        preds = self.svm.predict(X_scaled)
        
        return preds.tolist()
    
    def __call__(self, batch):
        return self.predict(batch)
    
def get_model() -> NewsClassifier:
    return NewsClassifier()