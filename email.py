import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_classifier_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# App title
st.title("📩 Email Spam Classifier")
st.write("Type a message below and I'll tell you if it's **SPAM** or **HAM**.")

# Input text
user_input = st.text_area("Enter your message here:")

# Predict function
def predict_message(msg):
    msg = msg.lower()
    vec = tfidf.transform([msg]).toarray()
    pred = model.predict(vec)[0]
    return '🛑 SPAM' if pred == 1 else '✅ HAM (Not Spam)'

# Predict button
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_message(user_input)
        st.subheader("Result:")
        st.success(result if "HAM" in result else result)

