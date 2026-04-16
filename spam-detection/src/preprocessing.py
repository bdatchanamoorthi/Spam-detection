import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')

    print("\nRAW DATA PREVIEW:")
    print(df.head())

    return df

def preprocess_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    # keep it SIMPLE
    text = text.replace("\n", " ").replace("\r", " ")

    return text

def apply_preprocessing(df):
    df['cleaned'] = df['message'].apply(preprocess_text)
    return df