# Install required libraries
pip install streamlit pandas scikit-learn

import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load trained model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Placeholder vectorizer (replace with your trained vectorizer)
    cv = CountVectorizer(stop_words="english")
    sample_messages = ["sample message one", "sample message two", "spam message three"]
    cv.fit(sample_messages)

except FileNotFoundError:
    st.error("Model or Vectorizer file not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop()

# App title
st.title("Spam Message Classifier")

# User input
user_input = st.text_area("Enter a message to classify:")

# Classification button
if st.button("Classify"):
    if user_input:
        try:
            data = cv.transform([user_input]).toarray()
            prediction = model.predict(data)

            if prediction[0] == 0:
                st.success("This is NOT Spam")
            else:
                st.error("This IS Spam")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.warning("Please enter a message to classify.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn")
