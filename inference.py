import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib


def load_saved_data():
    """Load saved TF-IDF model, matrix, and counts."""
    model_tf = joblib.load('model_tf.pkl')
    tfidf_matrice = joblib.load('tfidf_matrice.pkl')
    texts_counts = pd.read_csv('texts_counts.csv')
    return model_tf, tfidf_matrice, texts_counts


def post_process_response(input_text, response):
    """
    Post-process the response to remove the input text from the start of the response,
    if it starts with the input text (case-insensitive).
    """
    # Convert both input and response to lowercase for case-insensitive comparison
    input_text_lower = input_text.lower()
    response_lower = response.lower()

    # Remove the input text from the start of the response if it starts with it
    if response_lower.startswith(input_text_lower):
        # Only remove input text if it's at the start of the response
        response = response[len(input_text):].strip()

    # Further remove any repeated starting words from the response
    input_tokens = input_text_lower.split()
    response_tokens = response.split()

    # Find the position where the first mismatch occurs
    mismatch_index = 0
    while (mismatch_index < len(response_tokens) and
           mismatch_index < len(input_tokens) and
           response_tokens[mismatch_index].lower() == input_tokens[mismatch_index]):
        mismatch_index += 1

    # Join the remaining tokens from the first mismatch onwards
    return ' '.join(response_tokens[mismatch_index:]).strip()


def generate_completions(prefix_string, model_tf, tfidf_matrice, texts_counts):
    """Generate autocomplete suggestions based on a given prefix."""
    start_time = time.time()

    # Transform the prefix into a TF-IDF vector
    tfidf_matrice_spelling = model_tf.transform([prefix_string])

    # Compute cosine similarity between the prefix and the TF-IDF matrix
    cosine_similarity = linear_kernel(tfidf_matrice, tfidf_matrice_spelling)

    # Enumerate and sort similarity scores
    similarity_scores = list(enumerate(cosine_similarity))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Extract the top indices
    top_indices = [i[0] for i in similarity_scores[:10]]

    # Filter indices based on count >= 5
    valid_indices = [i for i in top_indices if texts_counts.loc[i, 'Counts'] >= 5]

    if not valid_indices:
        return []

    # Adjust similarity scores with frequency weights
    weights = texts_counts.loc[valid_indices, 'Counts'].values
    adjusted_scores = []
    for i, idx in enumerate(valid_indices):
        adjusted_score = similarity_scores[top_indices.index(idx)][1][0] * (1 + np.log1p(weights[i]))
        adjusted_scores.append((idx, adjusted_score))

    # Sort scores again based on the adjusted scores
    sorted_scores = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)

    # Calculate inference time
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.6f} seconds")

    # Get the top suggestion and apply post-processing
    top_suggestions = [texts_counts.loc[i, 'cleanedContent'] for i, _ in sorted_scores]

    # Apply post-processing to each suggestion
    processed_suggestions = [post_process_response(prefix_string, suggestion) for suggestion in top_suggestions]

    # Filter suggestions based on the minimum token count
    filtered_suggestions = [s for s in processed_suggestions if len(s.split()) >= 2]

    return filtered_suggestions


def main():
    """Main loop to handle console input and provide autocomplete suggestions."""
    # Load the saved model, matrix, and text counts
    model_tf, tfidf_matrice, texts_counts = load_saved_data()

    while True:
        # Prompt for user input
        prefix_string = input("Enter the start of a sentence (or type 'exit' to quit): ").strip()

        # Exit condition
        if prefix_string.lower() == 'exit':
            break

        # Token count condition
        token_count = len(prefix_string.split())
        if token_count < 3:
            print("Please enter at least 3 tokens for suggestions.")
            continue

        # Generate autocomplete suggestions
        suggestions = generate_completions(prefix_string, model_tf, tfidf_matrice, texts_counts)

        # Print the top suggestion(s)
        if suggestions:
            print("Autocomplete suggestions:")
            for suggestion in suggestions:
                print(f"{prefix_string} - {suggestion}")
                break
        else:
            print("No suggestions found.")


# Run the main function
if __name__ == "__main__":
    main()
