import streamlit as st
import pickle

# Load the trained spam detection model
with open('trained_spam_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit App Config
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
st.title("📩 Spam Mail Detector")
st.markdown("Enter your email content below to check if it's *SPAM* or *NOT SPAM*.")

# Input box
input_mail = st.text_area("✉ Paste the email content here:")

# Button to trigger prediction
if st.button("Check Now"):
    if input_mail.strip() == "":
        st.warning("⚠ Please enter some email content.")
    else:
        prediction = model.predict([input_mail])[0]
        if prediction == 1:
            st.error("🚨 This is *SPAM*!")
        else:
            st.success("✅ This is *NOT SPAM*.")
