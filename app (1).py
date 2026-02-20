import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="StackOverflow Tag Predictor",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Load Everything
# ----------------------------
@st.cache_resource
def load_files():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    mlb = pickle.load(open("mlb.pkl", "rb"))
    return model, vectorizer, mlb

model, vectorizer, mlb = load_files()

# ----------------------------
# UI
# ----------------------------
st.title("🧠 StackOverflow Tag Predictor")
st.write("Enter your programming question to predict relevant tags.")

user_input = st.text_area(
    "✍️ Enter your question:",
    height=150,
    placeholder="Example: How to merge two pandas dataframes?"
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict Tags"):

    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            # Transform text
            X = vectorizer.transform([user_input])

            # Predict binary output
            prediction = model.predict(X)

            # Convert binary → actual tag names
            predicted_tags = mlb.inverse_transform(prediction)

            if predicted_tags and len(predicted_tags[0]) > 0:
                st.success("### ✅ Predicted Tags:")
                for tag in predicted_tags[0]:
                    st.write(f"- {tag}")
            else:
                st.warning("No tags predicted. Try a more detailed question.")

        except Exception as e:
            st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️")
