import streamlit as st
import pickle

# Load the trained spam detection model
with open('trained_spam_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer (TF-IDF or similar)
with open('vectorizers.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App Config
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
st.title("📩 Spam Mail Detector")
st.markdown("Enter your email content below to check if it's SPAM or NOT SPAM.")

# Input box
input_mail = st.text_area("✉ Paste the email content here:")

# Button to trigger prediction
if st.button("Check Now"):
    if input_mail.strip() == "":
        st.warning("⚠ Please enter some email content.")
    else:
        # Transform input text using the vectorizer
        transformed_input = vectorizer.transform([input_mail])

        # Predict using the trained model
        prediction = model.predict(transformed_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM.")
