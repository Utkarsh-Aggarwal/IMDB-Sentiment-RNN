import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
#load the model
model = load_model("simple_rnn_model.keras", compile=False)  # Add compile=False to avoid config issues

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is the index for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(text):
    preprocessed_input = preprocess_text(text)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st
st.title("IMBD movie review Sentiment Analysis")
st.write("Enter your movie review below:")
user_input = st.text_area("Review")

if st.button("Classify"):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {prediction[0][0]}")
else:
    st.write("Enter a movie review to classify its sentiment.")