import argparse
import torch
import random
import numpy as np
import os
from src.data_utils import load_dataset, tokenize_texts, encode_labels, create_dataloader, compute_class_weights
from src.model import BertClassifier
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def preprocess_labels(df):
    # Map 'Irrelevant' to 'Neutral' and lowercase all labels for consistency
    df = df.copy()
    df.iloc[:, 2] = df.iloc[:, 2].replace({'Irrelevant': 'Neutral'})
    df.iloc[:, 2] = df.iloc[:, 2].str.lower()
    return df

def main():
    parser = argparse.ArgumentParser(description='Sentiment Pipeline')
    parser.add_argument('--train_path', type=str, default='data/twitter_training.csv')
    parser.add_argument('--val_path', type=str, default='data/twitter_validation.csv')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    parser.add_argument('--epochs', type=int, default=41)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--train_subset', type=int, default=1000)
    parser.add_argument('--val_subset', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=0, help='Warmup steps for scheduler (if supported)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Load and preprocess training data
    train_df = load_dataset(args.train_path)
    train_df = preprocess_labels(train_df)
    train_df = train_df[train_df.iloc[:, 2].isin(["positive", "negative", "neutral"])]
    train_df = train_df.sample(n=min(args.train_subset, len(train_df)), random_state=42)
    train_labels = train_df.iloc[:, 2]
    train_texts = train_df.iloc[:, 3].astype(str)  # No .str.lower() needed for cased model
    train_encodings = tokenize_texts(train_texts, "prajjwal1/bert-tiny")
    train_encoded_labels, le = encode_labels(train_labels)
    # Compute class weights for balancing
    class_weights = compute_class_weights(train_encoded_labels, num_classes=3).to(device)
    train_loader = create_dataloader(train_encodings, train_encoded_labels, batch_size=args.batch_size)

    # Build model
    model = BertClassifier(num_labels=3, model_name='prajjwal1/bert-tiny').to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    # Evaluation
    val_df = load_dataset(args.val_path)
    val_df = preprocess_labels(val_df)
    val_df = val_df[val_df.iloc[:, 2].isin(["positive", "negative", "neutral"])]
    val_df = val_df.sample(n=min(args.val_subset, len(val_df)), random_state=42)
    val_labels = val_df.iloc[:, 2]
    val_texts = val_df.iloc[:, 3].astype(str)  # No .str.lower() needed for cased model
    val_encodings = tokenize_texts(val_texts, "prajjwal1/bert-tiny")
    val_encoded_labels, _ = encode_labels(val_labels)
    val_loader = create_dataloader(val_encodings, val_encoded_labels, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1: {f1:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")

if __name__ == "__main__":
    main()
