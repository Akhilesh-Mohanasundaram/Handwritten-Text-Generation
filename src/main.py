import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from text_preprocessing import preprocess_text
from sklearn_nltk_classifier import SklearnNLTKClassifier

def main():
    print("Loading and preprocessing data...")
    data = pd.read_csv('../data/spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['processed_message'] = data['message'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['processed_message'], 
        data['label'], 
        test_size=0.2, 
        random_state=42
    )

    print("Vectorizing text data...")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training classifier...")
    sklearn_classifier = MultinomialNB()
    sklearn_classifier.fit(X_train_tfidf, y_train)

    nltk_classifier = SklearnNLTKClassifier(sklearn_classifier)

    print("Evaluating model...\n")
    y_pred = sklearn_classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')

    print("-" * 60)
    print("Model Performance(Evaluation) Results:\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 60)

    # Save the model and vectorizer to the models folder
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nSaving model and vectorizer...")
    joblib.dump(sklearn_classifier, os.path.join(models_dir, 'spam_classifier.pkl'))
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    print(f"Model saved to {os.path.join(models_dir, 'spam_classifier.pkl')}")
    print(f"Vectorizer saved to {os.path.join(models_dir, 'tfidf_vectorizer.pkl')}")

    print("\nExample classification:")
    sample_message = "Hey, are we still meeting for coffee at 5 PM? Let me know!"
    processed_sample = preprocess_text(sample_message)
    sample_vector = tfidf.transform([processed_sample])

    print(f"Sample message: {sample_message}")
    print(f"Classification: {sklearn_classifier.predict(sample_vector)[0]}")

if __name__ == "__main__":
    main()
