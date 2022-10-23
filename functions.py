# Importing Libraries
import pandas as pd
import numpy as np
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer, WordNetLemmatizer
import pickle
import nltk
nltk.download('wordnet')

# Text Preprocessing

# Converting Text to Lower Case


def text_lower(text):
    return text.str.lower()

# Stop Words Removal


def clean_stopwords(text):
    stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                    'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                    'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                    's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                    'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                    'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                    "youve", 'your', 'yours', 'yourself', 'yourselves']
    STOPWORDS = set(stopwordlist)
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Punctuations Removal


def clean_puctuations(text):
    english_puctuations = string.punctuation
    translator = str.maketrans('', '', english_puctuations)
    return text.translate(translator)

# Repeating Characters Removal


def clean_repeating_characters(text):
    return re.sub(r'(.)1+', r'1', text)

# URLs Removal


def clean_URLs(text):
    return re.sub(r"((www.[^s]+)|(http\S+))", "", text)

# Numeric Data Removal


def clean_numeric(text):
    return re.sub('[0-9]+', '', text)

# Text Tokenization


def tokenize_tweet(text):
    tokenizer = RegexpTokenizer('\w+')
    text = text.apply(tokenizer.tokenize)
    return text

# Stemmatization


def text_stemming(text):
    st = PorterStemmer()
    text = [st.stem(word) for word in text]
    return text

# Lemmatization


def text_lemmatization(text):
    lm = WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]
    return text

# Combined Function for Text Preprocessing


def preprocess(text):
    text = text_lower(text)
    text = text.apply(lambda text: clean_stopwords(text))
    text = text.apply(lambda x: clean_puctuations(x))
    text = text.apply(lambda x: clean_repeating_characters(x))
    text = text.apply(lambda x: clean_URLs(x))
    text = text.apply(lambda x: clean_numeric(x))
    text = tokenize_tweet(text)
    text = text.apply(lambda x: text_stemming(x))
    text = text.apply(lambda x: text_lemmatization(x))
    text = text.apply(lambda x: " ".join(x))
    return text

# Final Function for Prediction of Custom Text Input


def custom_input_prediction(text):
    import nltk
    nltk.download('omw-1.4')
    text = pd.Series(text)
    text = preprocess(text)
    text = [text[0], ]
    vectoriser = pickle.load(open("vectoriser.pkl", "rb"))
    text = vectoriser.transform(text)
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(text)
    prediction = prediction[0]

    interpretations = {
        0: "Age",
        1: "Ethnicity",
        2: "Gender",
        3: "Not Cyberbullying",
        4: "Other Cyberbullying",
        5: "Religion"
    }

    for i in interpretations.keys():
        if i == prediction:
            return interpretations[i]
