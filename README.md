# SentiMent: BERT-based Sentiment Analysis

This project provides a full pipeline for sentiment analysis on Twitter data using a BERT-based model.

## Features
- Data loading and preprocessing
- Class distribution visualization
- Tokenization and quick predictions
- Model training and evaluation
- Reproducible results with fixed seeds

## Project Structure
- `src/` - Source code (data utilities, model, training, evaluation)
- `data/` - Training and validation CSV files
- `models/` - Saved model checkpoints
- `outputs/` - Evaluation logs and confusion matrices
- `notebooks/` - Jupyter notebooks for exploration
- `run.py` - Entry point for the full pipeline

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train and evaluate:
   ```bash
   python run.py
   ```
   Use `--help` for configurable options.


