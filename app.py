import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin

import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
import spacy
import re
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import numpy as np
import base64

with open("ai.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    h1 {{
        text-align: center;
        margin-top: 120px;
    }}

    div[data-testid="stTextArea"] {{
        max-width: 600px;
        margin: 180px auto 0 auto;
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(6px);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.4);
        transform: translateX(-40px);
    }}

    </style>
    """,
    unsafe_allow_html=True
)


class W2VTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).astype(str)
        vectors = []

        for text in X:
            tokens = text.split()
            vecs = [self.model.wv[w] for w in tokens if w in self.model.wv]

            if len(vecs) == 0:
                vectors.append(np.zeros(self.dim))
            else:
                vectors.append(np.mean(vecs, axis=0))

        return np.vstack(vectors)

    def get_feature_names_out(self, input_features=None):

        return [f"w2v_{i}" for i in range(self.dim)]

nltk.download("wordnet")
nltk.download("omw-1.4")

@st.cache_resource
def load_models():
    xg = joblib.load("xg.joblib")
    ...
    return xg

model = load_models()



def create_input(text):
    text_lemma= lemma(text)
    text_trunc = trunc(text_lemma)

    df=pd.DataFrame({"text_trunc":[text_trunc],
                     'length':[len(text.split())],
                     })
    return df


lemmatizer = WordNetLemmatizer()

def lemma(text):
    t_p=re.sub(r'[^\w\s]','', text).lower()
    words=t_p.split()
    words_lemma = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words_lemma)

def trunc(text, max_words=800):
    words = text.split()
    return " ".join(words[:max_words])


st.title("Human or AI?")



text = st.text_area("Enter your text", height=200)


if st.button("Analyse"):
    if text.strip() == "":
        st.warning("Please enter some text")
    elif len(text.split()) < 30:
        st.warning("Text too short for reliable prediction.")
    else:

        input_df = create_input(text)

        # Make prediction
        predict = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        if predict == 0:
            st.success(f"Human-written text (confidence: {proba[0]:.2f})")
        else:
            st.warning(f"AI-generated text (confidence: {proba[1]:.2f})")


