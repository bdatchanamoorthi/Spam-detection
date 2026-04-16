# spam-detection
 SMS Spam Classification using Machine Learning

An intelligent machine learning project that classifies SMS messages as Spam or Ham (Legitimate) using Natural Language Processing (NLP) techniques and classification algorithms.

# Overview

Spam messages are a common problem in digital communication. This project builds a robust model that automatically detects spam SMS messages using:

TF-IDF (Term Frequency - Inverse Document Frequency)
Machine Learning algorithms like:
Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
# Features

-> Text preprocessing (cleaning, tokenization, stopword removal)
-> Feature extraction using TF-IDF
-> Multiple ML models for comparison
-> Performance evaluation (Accuracy, Precision, Recall, F1-score)
-> Predict custom SMS messages
-> Simple and modular code structure

# Project Structure
sms-spam-classifier/
│
├── data/
│   └── spam.csv
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── evaluate.py
│
├── app.py                # Optional (Flask Web App)
├── requirements.txt
└── README.md
# Dataset
Contains SMS messages labeled as:
Spam
Ham (Legitimate)

Common dataset used: SMS Spam Collection Dataset

# Installation

Clone the repository:

git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier

Install dependencies:

pip install -r requirements.txt
# Usage
1. Train the Model
python src/train_model.py
2. Evaluate the Model
python src/evaluate.py
3. Predict New Message

You can modify the script or use:

message = ["Congratulations! You've won a free ticket."]
prediction = model.predict(message)
print(prediction)
# Model Performance
Model	Accuracy
Naive Bayes	High
Logistic Regression	High
SVM	Very High

(Actual results may vary based on dataset and preprocessing)

# Example

Input:

"Win a free iPhone now!!! Click here"

Output:

Spam
# Technologies Used
Python
Scikit-learn
Pandas
NumPy
NLP (Natural Language Processing)
 Future Enhancements
 Deep Learning (LSTM, BERT)
 Web deployment (Flask / React)
 Mobile app integration
 Advanced visualization dashboard
 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

# License

This project is open-source and available under the MIT License.

 Author
Datchana moorthi B
GitHub: https://github.com/bdatchanamoorthi

⭐ If you like this project, don’t forget to star the repository!
