import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not present
nltk_packages = ['punkt', 'stopwords']
for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

ps = PorterStemmer()

def transform_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    
    # 3. Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]
    
    # 4. Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english')]
    
    # 5. Apply stemming
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

# Load the pre-trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.title("SMS/Email Spam Classifier")

# Input box for the message
input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Make prediction
        result = model.predict(vector_input)[0]
        
        # 4. Display the result
        if result == 1:
            st.header("Spam")
            st.markdown("### 🚫 This message is likely spam.")
        else:
            st.header("Not Spam")
            st.markdown("### ✅ This message seems legitimate.")
