
import streamlit as st
import joblib

import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{data}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("br.png")

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")
threshold = joblib.load("threshold.joblib")

st.title("AI vs Human Text Detector")

st.write('Enter your text')

text = st.text_area("Enter your text", height=200)


if st.button("Analyse"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        X = vectorizer.transform([text])
        proba = model.predict_proba(X)[0][1]

        # твой threshold!
        pred = 1 if proba > threshold else 0

        if pred == 1:
            st.error(f"AI-generated (confidence: {proba:.3f})")
        else:
            st.success(f"Human-written (confidence: {1 - proba:.3f})")



