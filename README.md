# Movie Review Sentiment Classifier 🎬🔍

A simple NLP-based machine learning project that classifies movie reviews as **positive** or **negative** using **TF-IDF** and **Logistic Regression**. The dataset used is the popular [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), and the project demonstrates end-to-end sentiment analysis including preprocessing, feature extraction, model training, and testing.

---

## 📁 Dataset

- **Source**: IMDB Large Movie Review Dataset (50,000 labeled reviews)
- **Format**: CSV
- **Columns**: `review` (text), `sentiment` (positive/negative)

---

## ⚙️ Features

- Text preprocessing using NLTK and SpaCy:
  - HTML tag removal
  - Lowercasing
  - Punctuation and number removal
  - Tokenization
  - Stopword removal
  - Lemmatization
- TF-IDF vectorization of text features
- Logistic Regression for sentiment classification
- Testable sentiment prediction function
- Train-test split with stratified sampling
- Sample predictions on custom input

---

## 🧪 Model

- **Algorithm**: Logistic Regression (binary classification)
- **Vectorizer**: TfidfVectorizer (`max_features=500`)
- **Accuracy**: Tested on 500-sample subset for fast experimentation

---

## 🐍 Libraries Used

- `pandas`
- `nltk`
- `spacy`
- `scikit-learn`
- `re`

---

## 🚀 How to Run

1. **Install requirements** (use pip or conda):
    ```bash
    pip install pandas scikit-learn nltk spacy
    python -m nltk.downloader stopwords punkt
    python -m spacy download en_core_web_sm
    ```

2. **Run the Python script**:
    ```bash
    python sentiment_classifier.py
    ```

3. **Test sample predictions** at the end of the file.

---

## 📝 Sample Output

```python
Review: 'This movie was absolutely bad! The acting was normal and the plot was not good.'
Predicted Sentiment: Negative

Review: 'I was so bored throughout the entire film. It was a complete waste of time and money.'
Predicted Sentiment: Negative

Review: 'The film was good, great but not terrible either. Some parts were good.'
Predicted Sentiment: Positive
