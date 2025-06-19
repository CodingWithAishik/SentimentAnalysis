import torch
from model import get_tokenizer, get_model
from config import MODEL_NAME, MAX_LENGTH

# loading tokenizer and model

tokenizer = get_tokenizer(MODEL_NAME)
model = get_model(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load('../data/processed/best_model.pth', map_location='cpu'))
model.eval()

def predict_sentiment(text):
    # tokenizing input text
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()
    return 'positive' if predicted == 1 else 'negative'

if __name__ == "__main__":
    # doing interactive sentiment prediction
    while True:
        user_input = input("Enter a sentence (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}")
