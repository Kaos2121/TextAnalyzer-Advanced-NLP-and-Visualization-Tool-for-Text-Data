import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from text_analyzer import TextAnalyzer

st.title("TextAnalyzer: Comprehensive Text Analysis and Visualization Tool")

uploaded_file = st.file_uploader("Upload a CSV file with a text column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    text_column = st.selectbox("Select the text column for analysis", df.columns)
    text_data = df[text_column].dropna().tolist()

    analyzer = TextAnalyzer(text_data)

    st.header("Sentiment Analysis")
    sentiment_scores = analyzer.sentiment_analysis()
    df['Sentiment Score'] = sentiment_scores
    st.dataframe(df[['Sentiment Score']])
    st.write("Average Sentiment Score:", df['Sentiment Score'].mean())

    st.header("Keyword Extraction")
    keywords = analyzer.keyword_extraction()
    st.write("Top Keywords:", keywords)

    st.header("Topic Modeling")
    topics = analyzer.topic_modeling()
    for topic in topics:
        st.write(topic)

    st.header("Named Entity Recognition")
    entities = analyzer.named_entity_recognition()
    st.write(entities)

    st.header("Sentiment Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sentiment Score'], bins=20, kde=True)
    plt.title("Sentiment Score Distribution")
    st.pyplot(plt.gcf())
