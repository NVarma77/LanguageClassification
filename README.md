# Language Classification with Character-Level NLP

## Overview
This project implements a **language identification system** that predicts the language of a text snippet using **character-level features**. The pipeline compares **Multinomial Naive Bayes** and a **linear Support Vector Machine (LinearSVC)**, demonstrating why character n-grams are effective for multilingual and short-text classification.

## Motivation
Word-level features often fail on short, noisy, or unseen words. Character-level TF-IDF captures **morphological patterns, scripts, and orthographic structure**, making it well-suited for language detection.

## Dataset
- Input: short text snippets  
- Labels: language classes  

### Preprocessing
- Whitespace normalization  
- Removal of empty or invalid samples  

## Methods

### Feature Engineering
- **TF-IDF Vectorization**
  - Analyzer: character-level  
  - N-grams: 3 to 5  
  - Lowercasing enabled  

### Models
- **Multinomial Naive Bayes**
  - Tuned smoothing parameter (`alpha`)  
- **Linear Support Vector Machine**
  - `LinearSVC` (one-vs-rest)  

### Training & Tuning
- Stratified K-Fold Cross-Validation  
- Grid search for hyperparameter tuning  
- Models compared on held-out test data  

### Evaluation
- Accuracy  
- Macro F1 score  
- **Column-normalized confusion matrix** (normalized by true class)  

## Results
- **Best model:** LinearSVC  
- **Test Accuracy:** ~98.8%  
- **Macro F1:** ~0.99  

## Key Takeaways
- Character n-grams are highly effective for language identification  
- Linear SVMs scale well and handle high-dimensional sparse text features  
- Stratified evaluation ensures fair performance across languages  

## Tech Stack
- Python  
- scikit-learn  
- NumPy  
- pandas  
- Matplotlib  

## Future Work
- Extend to low-resource languages  
- Compare against neural models (CNNs or transformers)  
- Add real-time inference or API deployment  

