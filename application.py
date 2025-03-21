import os
import joblib
import sys
from src.text_preprocessing import preprocess_text

def load_models():
    """Load the saved classifier and vectorizer models"""
    try:
        # Define paths to model files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        
        classifier_path = os.path.join(models_dir, 'spam_classifier.pkl')
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        
        # Check if model files exist
        if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
            print(f"Error: Model files not found in {models_dir}")
            print("Please train the model first by running 'python src/main.py'")
            sys.exit(1)
            
        # Load models
        classifier = joblib.load(classifier_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return classifier, vectorizer
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        sys.exit(1)

def classify_sms(message, classifier, vectorizer):
    """Classify an SMS message as spam or ham"""
    try:
        processed_message = preprocess_text(message)
        features = vectorizer.transform([processed_message])
        prediction = classifier.predict(features)[0]
        return prediction
    except Exception as e:
        return f"Classification error: {str(e)}"

def main():
    """Main function to run the SpamSense application"""
    print("=" * 60)
    print("Welcome to SpamSense - SMS Spam Detection System")
    print("=" * 60)
    
    print("Loading models...")
    classifier, vectorizer = load_models()
    print("Models loaded successfully!")
    print("-" * 60)
    
    print("Instructions:")
    print("- Type an SMS message to classify it as spam or ham")
    print("- Type 'exit' to quit the application")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nEnter SMS message: ")
            
            if user_input.lower() == 'exit':
                print("\nThank you for using SpamSense. Goodbye!")
                break
                
            if not user_input.strip():
                print("Please enter a valid message.")
                continue
                
            # Classify the message
            result = classify_sms(user_input, classifier, vectorizer)
            
            # Display result with appropriate formatting
            if result == 'spam':
                print("\nResult: [SPAM] - This message appears to be spam.")
            else:
                print("\nResult: [HAM] - This message appears to be legitimate.")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
