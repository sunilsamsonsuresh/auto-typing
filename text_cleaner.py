import pandas as pd
import re
from unidecode import unidecode

# Read CSV and drop any rows with missing values
df = pd.read_csv("DEU-Agent-Messages.csv").dropna()

def clean_text(text):
    # text = str(text)
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\u00A0', ' ')
    # Remove double quotes
    text = text.replace('"', '')
    # Transliterate non-ASCII characters to their closest ASCII equivalents
    text = unidecode(text)
    # Remove any character that is not an ASCII letter, number, or specified punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?/;()'\[\]{}<>]", ' ', text)
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    text = text.strip()

    # Remove text after "Best, "
    if '. Best' in text:
        text = text.split('. Best')[0].strip()

    # Remove tokens preceding the first comma
    if ',' in text:
        text = text.split(',', 1)[1].strip()

    return text

# Apply the cleaning function to the 'contentPlainText' column
df['cleanedContent'] = df['contentPlainText'].apply(clean_text)

# Drop the 'contentPlainText' column
df = df.drop(columns=['contentPlainText'])

# Save the cleaned DataFrame to a new CSV file
df.to_csv("DEU-Agent-Messages-Cleaned.csv", index=False)