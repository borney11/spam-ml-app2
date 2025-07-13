import streamlit as st
import pickle
import re

# ✅ Load the trained model
with open('trained_spam_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# ✅ Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ✅ Function to clean the input text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    return text.strip()

# ✅ Streamlit UI
st.set_page_config(page_title="Spam Mail Detector", layout="centered")
st.title("📩 Spam Mail Detector")
st.markdown("Enter your email content below to check if it's SPAM or NOT SPAM.")

# ✉ Input field
input_mail = st.text_area("✉ Paste the email content here:")

# 🔍 Prediction
if st.button("Check Now"):
    if input_mail.strip() == "":
        st.warning("⚠ Please enter some email content.")
    else:
        try:
            cleaned_input = clean_text(input_mail)
            transformed_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(transformed_input)[0]

            if prediction == 1:
                st.error("🚨 This is SPAM!")
            else:
                st.success("✅ This is NOT SPAM.")
        except Exception as e:
            st.error("❌ Prediction failed. Possibly due to vectorizer mismatch.")
            st.exception(e)
