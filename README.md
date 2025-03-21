# SpamSense, Spam SMS Detection System: A Machine Learning-Based Approach
SpamSense is an intelligent SMS filtering system that leverages natural language processing and machine learning techniques to accurately classify text messages as spam or legitimate communications, helping users avoid unwanted and potentially harmful messages.

## Task Objectives

- Build a machine learning system for SMS spam detection
- Implement text preprocessing techniques including lowercasing, tokenization, stopword removal, and stemming
- Create a custom NLTK classifier wrapper for scikit-learn models
- Train a Naive Bayes classifier on SMS data
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Save and load models for future use

## Steps to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/username/SpamSMSDetection.git
cd SpamSMSDetection
```

### 2. Set Up a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

- Ensure the `spam.csv` file is placed in the `data` directory in the root directory
- The dataset should contain SMS messages labeled as 'spam' or 'ham'

### 5. Train the Model

```bash
cd src
python main.py
```

This will:
- Load and preprocess the SMS data
- Train a Naive Bayes classifier
- Evaluate the model's performance
- Save the trained model and vectorizer to the `models` directory

### 6. Test the Model

Return to the project root directory and run the test script:

```bash
# From the project root
cd ..
python test.py
```

This will:
- Load the saved model and vectorizer
- Test the model on example SMS messages
- Display the classification results

### 7. Deactivate the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

### Dataset:

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/58131223/bd6f2881-8f9a-40c3-b1f6-b3c4f0fb7d64/spam.csv



---