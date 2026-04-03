import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.feature_extraction.text import CountVectorizer # pyright: ignore[reportMissingModuleSource]
from sklearn.naive_bayes import MultinomialNB # pyright: ignore[reportMissingModuleSource]
import streamlit as st # pyright: ignore[reportMissingImports]

# بيانات للتجربة
data = {
    "text": [
        "The company made huge profits",
        "The news is terrible and disappointing",
        "I am very happy with this",
        "This is a bad decision",
        "The stock market is rising",
        "The product failed completely"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# تدريب الموديل
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# واجهة Streamlit
st.title("News Sentiment Analyzer")
st.write("Enter a news text and see if it is Positive or Negative:")

user_input = st.text_area("Enter news text here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)
        if prediction[0] == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😢")