import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

# 1. Load dataset 

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df

# 2. Tokenize text using transformers.AutoTokenizer

def tokenize_texts(texts, tokenizer_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Ensure texts is a list of strings
    if isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()
    elif isinstance(texts, (list, tuple)):
        texts = [str(t) for t in texts]
    else:
        texts = [str(texts)]
    # Lowercase for uncased models
    if 'uncased' in tokenizer_name:
        texts = [t.lower() for t in texts]
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# 3. Encode labels

def encode_labels(labels):
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le

# 4. Create PyTorch DataLoaders for training and testing

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def create_dataloader(encodings, labels=None, batch_size=32, shuffle=True):
    dataset = CustomDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def compute_class_weights(labels, num_classes=None):
    """
    Compute class weights for imbalanced datasets.
    Args:
        labels: Encoded integer labels (numpy array, list, or tensor)
        num_classes: Number of classes (optional, inferred if None)
    Returns:
        torch.Tensor of class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if num_classes is None:
        num_classes = len(set(labels))
    classes = np.array(range(num_classes))
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def _test():
    """Quick test for data_utils functions."""
    df = load_dataset("data/twitter_training.csv")
    print("Loaded dataset shape:", df.shape)
    labels = df.iloc[:, 2] # Third column: label
    texts = df.iloc[:, 3] # Fourth column: text

    encodings = tokenize_texts(texts, "bert-base-cased")
    encoded_labels, le = encode_labels(labels)
    dataloader = create_dataloader(encodings, encoded_labels, batch_size=4)

    for batch in dataloader:
        print({k: v.shape for k, v in batch.items()})
        break
    print("Test completed successfully.")

if __name__ == "__main__":
    _test()
