import joblib

# Load saved files
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict(message):
    message = [message]
    X = vectorizer.transform(message)
    result = model.predict(X)

    return "Spam" if result[0] == 1 else "Good"


# Test
print(predict("Win a free iPhone now!!!"))