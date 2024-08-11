import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample corpus of sentences
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A penny saved is a penny earned."
]

# Function to complete a given sentence
def complete_sentence(partial_sentence, corpus):
    # Tokenize and compute TF-IDF vectors for the corpus
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Vectorize the partial sentence
    partial_sentence_vector = vectorizer.transform([partial_sentence])

    # Compute cosine similarities between the partial sentence and the corpus
    cosine_similarities = cosine_similarity(partial_sentence_vector, tfidf_matrix).flatten()

    # Find the most similar sentence in the corpus
    most_similar_idx = np.argmax(cosine_similarities)
    return corpus[most_similar_idx]

# Example usage
partial_sentence = "A journey of a thousand miles"
completion = complete_sentence(partial_sentence, corpus)
print(f"Completion for '{partial_sentence}': {completion}")
