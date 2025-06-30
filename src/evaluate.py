import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import BertClassifier
from data_utils import create_dataloader, load_dataset, tokenize_texts, encode_labels

# Settings
BATCH_SIZE = 32
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load validation data
df = load_dataset("data/twitter_validation.csv")
df = df[df.iloc[:, 2].isin(["Positive", "Negative", "Neutral"])]
labels = df.iloc[:, 2]
texts = df.iloc[:, 3]
encodings = tokenize_texts(texts, "bert-base-cased")
encoded_labels, le = encode_labels(labels)
dataloader = create_dataloader(encodings, encoded_labels, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = BertClassifier(num_labels=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        outputs = model(input_ids, attention_mask, token_type_ids)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

# Save logs
log_path = os.path.join(OUTPUT_DIR, "eval_metrics.txt")
with open(log_path, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
print(f"Metrics saved to {log_path}")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to {cm_path}")
