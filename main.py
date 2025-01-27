from keras.models import load_model  
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st
model = load_model("simple_rnn.h5")


word_index = imdb.get_word_index()
#function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.7 else 'Negative'

    return sentiment, prediction[0][0]

st.title("Movie review prediction")

st.write("enter a movie review")

user_input = st.text_area("enter a review")

if st.button("Classify"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"sentiment: {sentiment}")
    st.write(f"confidence: {confidence}")
else:
    st.write("enter something")
