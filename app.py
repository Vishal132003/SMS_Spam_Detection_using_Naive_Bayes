# Install necessary libraries (if not already installed)
!pip install streamlit pandas scikit-learn

import streamlit as st
import pickle
import pandas as pd

# Load the trained model and the CountVectorizer
# Make sure 'model.pkl' and 'vectorizer.pkl' are in the same directory as your app.py file
# You might need to save the CountVectorizer as well in your notebook.
# For this example, I'll assume the vectorizer is saved as 'vectorizer.pkl'
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # You need to save your CountVectorizer object as well to transform new messages
    # If you haven't saved it, you'll need to add code to your notebook to save the 'cv' object
    # For now, let's assume you have a 'vectorizer.pkl'
    # If you don't have 'vectorizer.pkl', you will need to train/load your vectorizer here.
    # As a workaround for this Colab environment, let's re-create a simple vectorizer.
    # In a real-world app, you should save and load your trained vectorizer.

    # **Important:** In a real application, save and load the *trained* cv object.
    # Since we don't have the original data here to refit, this is a placeholder.
    # You'll need to replace this with loading your saved vectorizer.
    from sklearn.feature_extraction.text import CountVectorizer
    # This is a placeholder. Replace with loading your actual fitted vectorizer.
    cv = CountVectorizer(stop_words="english")
    # This fit is just to make the example runnable in isolation; it's not ideal.
    # You should load the fitted vectorizer from your training process.
    sample_messages = ["sample message one", "sample message two", "spam message three"]
    cv.fit(sample_messages)


except FileNotFoundError:
    st.error("Model or Vectorizer file not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are not found

# Set the title of the Streamlit application
st.title("Spam Message Classifier")

# Add a text input box for the user to enter a message
user_input = st.text_area("Enter a message to classify:")

# Add a button to trigger the classification
if st.button("Classify"):
    if user_input:
        # Transform the input text using the loaded CountVectorizer
        # In a real app, ensure the loaded vectorizer is the one fitted on your training data.
        try:
            data = cv.transform([user_input]).toarray()
            # Make a prediction using the loaded model
            prediction = model.predict(data)

            # Display the result
            if prediction[0] == 0:
                st.success("This is NOT Spam")
            else:
                st.error("This IS Spam")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

    else:
        st.warning("Please enter a message to classify.")

# Add a footer or additional information
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn")
