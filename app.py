import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

# Function to predict
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred]

# Emoji mapping
emoji_map = {
    "Positive": "😊👍",
    "Neutral": "😐",
    "Negative": "😠👎"
}

# Streamlit UI
st.title("📝 Sentiment Analysis App")
st.markdown("Enter a product review below:")

user_input = st.text_area("Your Review", "")

if st.button("Analyze"):
    if user_input.strip():
        prediction = predict_sentiment(user_input)
        emoji = emoji_map[prediction]
        st.success(f"**Predicted Sentiment:** {prediction} {emoji}")
    else:
        st.warning("Please enter some text.")
