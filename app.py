import streamlit as st
import pickle
import string
import one
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Specify the path to the local nltk_data folder
one.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = one.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)


# Load your vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS/Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("Spam 🚫")
        else:
            st.header("Not Spam ✅")
