import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure nltk_data is in the current directory
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download required nltk packages if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app interface
st.title("📧 SMS/Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict the result
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
