from flask import Flask, request, render_template
import joblib
from src.text_preprocessing import preprocess_text

app = Flask(__name__)

# Load the saved model and vectorizer
classifier = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        features = vectorizer.transform([processed_message])
        prediction = classifier.predict(features)[0]
        result = 'spam' if prediction == 'spam' else 'ham'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
