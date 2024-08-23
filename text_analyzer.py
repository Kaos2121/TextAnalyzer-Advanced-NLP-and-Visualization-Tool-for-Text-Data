import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import pandas as pd

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class TextAnalyzer:
    def __init__(self, text_data):
        self.text_data = text_data
        self.stop_words = set(stopwords.words('english'))

    def sentiment_analysis(self):
        sentiment_scores = [TextBlob(text).sentiment.polarity for text in self.text_data]
        return sentiment_scores

    def keyword_extraction(self, top_n=10):
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words=self.stop_words)
        X = vectorizer.fit_transform(self.text_data)
        keywords = vectorizer.get_feature_names_out()
        return keywords

    def topic_modeling(self, num_topics=5, num_words=5):
        tokens = [word_tokenize(text.lower()) for text in self.text_data]
        tokens = [[word for word in text if word.isalnum() and word not in self.stop_words] for text in tokens]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        topics = ldamodel.print_topics(num_words=num_words)
        return topics

    def named_entity_recognition(self):
        entities = []
        for text in self.text_data:
            doc = nlp(text)
            entities.append([(X.text, X.label_) for X in doc.ents])
        return entities
