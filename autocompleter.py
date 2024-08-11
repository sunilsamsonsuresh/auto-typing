import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # For saving/loading model and matrices

class AutocompleterSaver:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        self.model_tf = None
        self.tfidf_matrice = None

    def load_data(self):
        """Load and process data."""
        self.df = pd.read_csv(self.input_file)
        self.df = pd.DataFrame(self.df['cleanedContent'].dropna()).reset_index(drop=True)
        print(f"Data loaded with {self.df.shape[0]} rows.")

    def process_data(self):
        """Process the DataFrame to clean and prepare text data."""
        self.df = self.split_sentences(self.df)
        self.df = self.count_sentence_occurrences(self.df)
        return self.df

    def split_sentences(self, df):
        """Split sentences based on punctuation."""
        separators = ['. ', ', ', '? ', '! ', '; ']
        for sep in separators:
            df = self.split_dataframe_list(df, 'cleanedContent', sep)
        return df

    def split_dataframe_list(self, df, target_column, separator):
        """Split entries in a DataFrame's column into multiple rows."""
        new_rows = []
        for _, row in df.iterrows():
            self.split_row(row, new_rows, target_column, separator)
        return pd.DataFrame(new_rows)

    def split_row(self, row, row_accumulator, target_column, separator):
        """Helper function to split a single row."""
        split_texts = row[target_column].split(separator)
        for text in split_texts:
            if text.strip():  # Avoid adding empty texts
                # Check if the split text has at least 5 tokens
                if len(text.split()) >= 5:
                    new_row = row.to_dict()
                    new_row[target_column] = text.strip()
                    row_accumulator.append(new_row)

    def count_sentence_occurrences(self, df):
        """Count occurrences of each sentence."""
        df['Counts'] = df.groupby('cleanedContent')['cleanedContent'].transform('count')
        return df

    def calc_matrice(self):
        """Compute the TF-IDF matrix."""
        if self.df is None:
            raise ValueError("DataFrame is not loaded or processed.")
        self.model_tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5)
        self.tfidf_matrice = self.model_tf.fit_transform(self.df['cleanedContent'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrice.shape}")

    def save(self):
        """Save the TF-IDF model and matrix."""
        if self.model_tf is None or self.tfidf_matrice is None:
            raise ValueError("TF-IDF model and matrix are not calculated.")
        joblib.dump(self.model_tf, 'model_tf.pkl')
        joblib.dump(self.tfidf_matrice, 'tfidf_matrice.pkl')
        self.df.to_csv('texts_counts.csv', index=False)
        print("Model, matrix, and counts saved.")

    def run(self):
        """Load data, process, and save model."""
        self.load_data()
        self.process_data()
        self.calc_matrice()
        self.save()

# Example usage
saver = AutocompleterSaver("DEU-Agent-Messages-Cleaned.csv")
saver.run()
