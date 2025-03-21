import os
import joblib
import sys
from src.text_preprocessing import preprocess_text

def test_spam_classifier():
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    # Define paths to saved model files
    classifier_path = os.path.join(models_dir, 'spam_classifier.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    # Check if model files exist
    if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
        print(f"Error: Model files not found. Make sure to train the model first.")
        print(f"Looking for files in: {models_dir}")
        sys.exit(1)
    
    # Load the saved model and vectorizer
    print("Loading saved model and vectorizer...")
    classifier = joblib.load(classifier_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Test examples
    test_examples = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize reward!",
        "Hey, are we still meeting for coffee at 5 PM? Let me know!",
        "Congratulations! You've been selected for a free iPhone. Call now to claim your prize!",
        "Don't forget to pick up milk on your way home.",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
    ]
    
    print("\nTesting example messages:")
    print("-" * 60)
    
    for i, message in enumerate(test_examples, 1):
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Transform using the loaded vectorizer
        features = vectorizer.transform([processed_message])
        
        # Predict using the loaded classifier
        prediction = classifier.predict(features)[0]
        
        # Print results
        print(f"Example {i}:")
        print(f"Message: {message}")
        print(f"Classification: {prediction}")
        print("-" * 60)

if __name__ == "__main__":
    test_spam_classifier()
