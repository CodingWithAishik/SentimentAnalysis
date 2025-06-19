# Sentiment Analysis Project

This project uses BERT to classify the sentiment of tweets as positive or negative.

## Structure
- `notebooks/` — doing exploratory analysis and prototyping
- `src/` — doing model, training, inference, and creative features
- `data/` — doing storage of raw and processed datasets

## Usage

### Training
Run the training script to train and validate the model:
```bash
python src/train.py
```

### Inference
Run the inference script for interactive sentiment prediction:
```bash
python src/inference.py
```

### Unconventional Feature: ASCII Art Sentiment
Run the creative script for sentiment prediction with ASCII art:
```bash
python src/art_sentiment.py
```

## Requirements
- torch
- transformers
- pandas

Install requirements with:
```bash
pip install torch transformers pandas
```
⚠️ **Disclaimer**

This project was created for learning purposes and may contain code generated with the assistance of GitHub Copilot.

No license is granted for reuse or redistribution.  
If you wish to use any part of this code, please contact the author or verify the original sources and licensing yourself.

