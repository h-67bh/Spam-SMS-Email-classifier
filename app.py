import streamlit as st
import numpy as np
import pickle
import string
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load models
scaler = pickle.load(open("scaler.pkl", "rb"))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS classifier app')

input_sms = st.text_area('Enter your email/sms')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Predict"):
    # 1. Preprocess
    transformed_text = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_text]).toarray()
    vector_input = scaler.transform(vector_input)

    # 3. Append num_characters (because model was trained with it)
    num_char = len(input_sms)
    vector_input = np.hstack([vector_input, [[num_char]]])

    # 4. Predict
    result = model.predict(vector_input)[0]

    # 5. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

