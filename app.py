import streamlit as st
from transformers import pipeline

# Load Hugging Face Sentiment Analysis Pipeline
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

# Function to predict sentiment
def predict_sentiment(text):
    result = model(text)[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_map.get(result['label'], "Unknown")
    confidence = result['score']
    return sentiment, confidence

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
        prediction, confidence = predict_sentiment(user_input)
        emoji = emoji_map[prediction]
        st.success(f"**Predicted Sentiment:** {prediction} {emoji}")
        st.info(f"Confidence Score: {confidence:.2f}")
    else:
        st.warning("Please enter some text.")
