import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AdamW
from model import get_tokenizer, get_model
from config import MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS, MAX_LENGTH

# defining custom dataset for Twitter data
class TwitterDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['cleaned_tweet'])
        label = int(self.data.iloc[idx]['label'])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

# loading tokenizer and model

tokenizer = get_tokenizer(MODEL_NAME)
model = get_model(MODEL_NAME, num_labels=2)

dataset = TwitterDataset('../data/processed/twitter_cleaned.csv', tokenizer, max_length=MAX_LENGTH)

# splitting into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# training and validation loop
best_val_loss = float('inf')
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # validating
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # saving best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()

if best_model_state is not None:
    torch.save(best_model_state, '../data/processed/best_model.pth')
    print("Best model saved to ../data/processed/best_model.pth")
else:
    print("No model was saved.")
