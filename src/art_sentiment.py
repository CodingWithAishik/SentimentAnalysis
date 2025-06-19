import torch
from model import get_tokenizer, get_model
from config import MODEL_NAME, MAX_LENGTH
import random

# loading tokenizer and model

tokenizer = get_tokenizer(MODEL_NAME)
model = get_model(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load('../data/processed/best_model.pth', map_location='cpu'))
model.eval()

# defining ASCII art for moods
ART = {
    'positive': [
        r"""
        \(^_^)/
        """,
        r"""
        (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
        """,
        r"""
        (•‿•)
        """
    ],
    'negative': [
        r"""
        (╯︵╰,)
        """,
        r"""
        (ಥ﹏ಥ)
        """,
        r"""
        (¬_¬)
        """
    ]
}

def predict_sentiment_with_art(text):
    # tokenizing input text
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()
    sentiment = 'positive' if predicted == 1 else 'negative'
    # picking random ASCII art for the sentiment
    art = random.choice(ART[sentiment])
    return sentiment, art

if __name__ == "__main__":
    # doing interactive sentiment prediction with ASCII art
    while True:
        user_input = input("Enter a sentence (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        sentiment, art = predict_sentiment_with_art(user_input)
        print(f"Predicted sentiment: {sentiment}\n{art}")
