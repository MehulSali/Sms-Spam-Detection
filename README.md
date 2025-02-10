# SMS/Email Spam Classifier

## Project Overview
This project is a web application that classifies SMS and email messages as spam or not spam. It uses a machine learning model trained on a dataset of labeled messages. The application is built using Python, Streamlit for the user interface, and the NLTK library for text preprocessing.

## Features
- Classifies SMS or email messages as **Spam** or **Not Spam**.
- User-friendly web interface using Streamlit.
- Preprocesses and cleans text using NLP techniques such as tokenization, stemming, and stopword removal.
- Utilizes a pre-trained machine learning model and a TF-IDF vectorizer for prediction.

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing for text preprocessing
- **Pickle**: Model and vectorizer serialization
- **Scikit-learn**: Machine learning library for building and training the model

## Installation and Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sms-spam-classifier.git
    cd sms-spam-classifier
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open the app in your web browser at `http://localhost:8501`.

5. Enter the SMS or email message in the input box and click on **Predict** to see the classification result.

## File Descriptions
- `app.py`: Main Python file that runs the Streamlit application.
- `vectorizer.pkl`: Pre-trained TF-IDF vectorizer used for text feature extraction.
- `model.pkl`: Trained machine learning model for spam classification.
- `README.md`: Project documentation.

## Text Preprocessing
The text preprocessing includes the following steps:
1. Converting text to lowercase
2. Tokenizing the text
3. Removing non-alphanumeric characters
4. Removing stopwords and punctuation
5. Applying stemming using Porter Stemmer

## Example
- **Input**: "Congratulations! You won a free lottery ticket."
- **Output**: Spam

- **Input**: "Let's meet for lunch at 1 PM."
- **Output**: Not Spam

## Future Improvements
- Add more advanced NLP techniques like lemmatization.
- Deploy the application to a cloud platform like Heroku.
- Enhance the UI with more features and custom styling.

## Author
**Mehul Sali**
