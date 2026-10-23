import streamlit as st
import joblib

st.set_page_config(page_title="Fraud Email Detection", page_icon="📧")

@st.cache_resource
def load_model():
    return joblib.load("email_model.pkl")

model = load_model()

st.title("Fraud Email Detection System")
st.write("Check whether an email may be fraudulent.")

email_text = st.text_area("Paste your email here")

if st.button("Check Email"):
    if email_text.strip():
        prediction = str(model.predict([email_text])[0]).lower()

        if any(word in prediction for word in ["fraud", "spam", "phishing"]):
            st.error("⚠️ This email may be fraudulent!")
        else:
            st.success("✅ This email appears legitimate.")
    else:
        st.warning("Please enter an email.")