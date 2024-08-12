import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer if needed
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app title
st.title("Sentiment Analysis App")

# Text input from user
user_input = st.text_area("Enter the text you want to analyze:")
if st.button("Analyze"):
    if user_input:
        # Preprocess and vectorize the input text
        input_vectorized = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_vectorized)[0]
        prediction_proba = model.predict_proba(input_vectorized)[0]

        # Display results
        st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
        st.write(f"Confidence: {prediction_proba[prediction]:.2f}")
    else:
        st.write("Please enter some text to analyze.")