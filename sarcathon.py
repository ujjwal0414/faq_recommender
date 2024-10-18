import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import nltk
import string
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load GloVe embeddings


def load_glove_embeddings():
    """Load GloVe embeddings from file into a dictionary."""
    embeddings = {}
    with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load FAQ data


def load_faq_data():
    """Load FAQ data (questions, answers, and vectors) from Pickle file."""
    with open('faq_data.pkl', 'rb') as f:
        return pickle.load(f)


# Load embeddings and FAQ data
glove_embeddings = load_glove_embeddings()
questions, answers, question_vectors = load_faq_data()

# Initialize stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation


def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        word for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return processed_tokens


embedding_dimension = 50


def get_glove_words_vector(sentence, glove_embeddings, embedding_dim=embedding_dimension):
    words = preprocess_text(sentence)
    word_vectors = np.array([glove_embeddings[word] for word in words if word in glove_embeddings])

    if np.any(np.isnan(word_vectors)):
        return np.zeros((embedding_dim,))

    return word_vectors


question_vectors = [get_glove_words_vector(q, glove_embeddings, embedding_dimension) for q in questions]


def word_similarities_glove(user_query):
    blob = TextBlob(user_query)
    corrected_query = str(blob.correct())

    query_vector = get_glove_words_vector(corrected_query, glove_embeddings)
    query_normalised = query_vector/np.linalg.norm(query_vector, axis=1, keepdims=True)
    max_length = max(len(q) for q in question_vectors)
    question_array = np.array([np.pad(question, ((0, max_length - question.shape[0]), (0, 0)), mode='constant', constant_values=0) for question in question_vectors])
    questions_normalised = question_array/np.linalg.norm(question_array, axis=2, keepdims=True)

    similarity = np.tensordot(questions_normalised, query_normalised, axes=([-1], [-1]))
    similarity = np.nan_to_num(similarity, nan=0)
    similarity = similarity.reshape(22, -1)
    similarity = np.mean(np.sort(similarity, axis=1)[:, -(max(2, len(query_vector))):], axis=1)

    best_match_idx = np.argsort(similarity)[-5:][::-1]
    results = []
    for idx in best_match_idx:
        results.append((questions[idx], answers[idx]))

    return results


st.image("saras logo.jpeg", width=700)
st.title("Saras AI - FAQ Search System")
st.write("Ask a question to find relevant answers:")

user_query = st.text_input("Enter your query:")

if user_query:
    matches = word_similarities_glove(user_query)
    st.header("Top 5 FAQs based on your query are:")
    for question, answer in matches:
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
        st.write("---")