import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# ✅ Set Streamlit page config (put this before anything else)
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")

# ✅ Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("simple_rnn_model.keras")

model = load_model()

# ✅ Constants
max_features = 10000
max_length = 500

# ✅ Load and prepare word index
@st.cache_resource
def get_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index

word_index = get_word_index()

# ✅ Encode input review to sequence
def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # Start token
    for word in words:
        index = word_index.get(word, 2)
        if index < max_features:
            encoded.append(index)
    return sequence.pad_sequences([encoded], maxlen=max_length)

# ✅ UI Layout
st.title("🎬 IMDB Sentiment Analyzer")
st.write("Enter a movie review to predict if it's **positive** or **negative**.")

review = st.text_area("Your Movie Review")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        encoded = encode_review(review)
        prediction = model.predict(encoded)[0][0]
        sentiment = "🟢 Positive" if prediction >= 0.5 else "🔴 Negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence:.2f}")
