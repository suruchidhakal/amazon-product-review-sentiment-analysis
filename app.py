import streamlit as st
import pickle


with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl','rb') as f:
    tfidf = pickle.load(f)


st.set_page_config(page_title="Amazon Review Sentiment", page_icon="🛒")
st.title("🛒 Amazon product Review Sentiment Analyzer")
st.markdown("paste any Amazon product review below and find out if it's **Positive**  or **Negative** .")

user_input = st.text_area("Type or paste an Amazon product review:", height=150)


if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please write a review first!")
    else:
        input_vector = tfidf.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.success(" This is a Positive Review!")
        else:
            st.error(" This is a Negative Review.")
