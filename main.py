import streamlit as st
import pickle
import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()
nlp = spacy.load("en_core_web_sm")

with open("model_fit.pkl", 'rb') as file:
    model = pickle.load(file)

with open("countvectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)


def text_transformed(text):
    # print("Original Text : ", text)

    # LowerCasing
    text = text.lower()
    # print("Lower Cased Text : \n", text)

    # Converting text into spacy.token
    text = nlp(text)

    # Converting spacy token into list using list comprehension.
    # doc = [token.text for token in text]
    # print('doc of spacy token \n',type(doc))

    # Removing special characters from the text
    y = [token.text for token in text if not token.is_punct]
    text = y[:]
    # print('after removing special characters : \n', text)
    y.clear()

    # Removing Stopwords and punctuations.
    y = [i for i in text if i not in stopwords.words('english')]
    text = y[:]
    # print('after removing stopwords : \n', text, type(text))
    y.clear()

    # Stemming
    y = [ps.stem(i) for i in text]
    text = y[:]
    # print('after stemming : \n', text)
    y.clear()

    return " ".join(text)


st.title("Spam Classifier!")

text_input = st.text_input("Input Text")

button = st.button("Check")

if button:

    transformed_txt = text_transformed(text_input)
    st.text(transformed_txt)

    vectorized = vectorizer.transform([transformed_txt])
    result = model.predict(vectorized)[0]

    if result == 1:
        st.text("email is a spam")
    else:
        st.text("email is ham")
