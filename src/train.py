import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import os
from model import BertClassifier
from data_utils import create_dataloader, load_dataset, tokenize_texts, encode_labels

# Settings
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data (adjust paths as needed)
df = load_dataset("data/twitter_training.csv")
df = df[df.iloc[:, 2].isin(["Positive", "Negative", "Neutral"])]
labels = df.iloc[:, 2]
texts = df.iloc[:, 3]
encodings = tokenize_texts(texts, "bert-base-cased")
encoded_labels, le = encode_labels(labels)
dataloader = create_dataloader(encodings, encoded_labels, batch_size=BATCH_SIZE)

# Model, optimizer, scheduler
model = BertClassifier(num_labels=4).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
criterion = CrossEntropyLoss()

best_loss = float('inf')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")
