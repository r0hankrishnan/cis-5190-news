import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig
)

MAX_LENGTH = 64
BATCH_SIZE = 64

class _HeadlineDataset(Dataset):
    def __init__(self, headlines, tokenizer):
        self.encodings = tokenizer(headlines, 
                                   truncation = True,
                                   max_length = MAX_LENGTH,
                                   padding = "max_length",
                                   return_tensors = "pt"
                                   )
        
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
        
    def __getitem__(self, idx):
        return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx]
            }

class NewsClassifier:
    def __init__(self, weights_path='model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.model = DistilBertForSequenceClassification(config)
        if weights_path and weights_path != '__no_weights__.pth':
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict = False)
        self.model.to(self.device)
        self.model.eval()
        
     
    @torch.no_grad()
    def predict_batch(self, X):
        """
        X: list of headline strings (from preprocess.prepare_data)
        Returns: list of label strings ('Fox News' or 'NBC News')
        """
        dataset = _HeadlineDataset(list(X), self.tokenizer)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
 
        all_preds = []
        for batch in loader:
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
 
        return all_preds
    
    def __call__(self, X):
        return self.predict_batch(X)
    
def get_model():
    return NewsClassifier()
