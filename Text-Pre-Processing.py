"""
Text Preprocessing Script

This Python script performs multiple preprocessing steps on a given text file to clean and standardize the text for further analysis or NLP tasks. The preprocessing steps include:

1. Lowercasing: Converts text to lowercase for uniformity.
2. Removing Punctuation: Eliminates punctuation marks.
3. Removing Special Characters: Strips out symbols like @, #, &.
4. Removing Numbers: Deletes numerical data if irrelevant.
5. Removing Extra Whitespaces: Cleans up multiple spaces, tabs, or newlines.
6. Removing URLs and HTML Tags: Strips web addresses and HTML code.
7. Handling Emojis and Emoticons: Removes or converts emojis to textual equivalents.
8. Handling Contractions: Expands contractions (e.g., "don't" -> "do not").
9. Text Unification: Unifies text variants.
10. Removing Non-ASCII Characters: Deletes or normalizes non-ASCII characters.
11. Word Tokenization: Splits text into individual words.
12. Stop Words Removal: Removes common stop words.
13. Stemming: Reduces words to their root form.
14. Lemmatization: Converts words to their base dictionary form.

"""

import re
import string
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Removing Special Characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Removing Numbers
    text = re.sub(r'\d+', '', text)

    # Removing Extra Whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Removing URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Removing HTML Tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Handling Emojis
    text = emoji.demojize(text)

    # Handling Contractions
    contractions = {"don't": "do not", "can't": "cannot", "i'm": "i am", "you're": "you are"}
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Removing Non-ASCII Characters
    text = text.encode('ascii', 'ignore').decode()

    # Tokenization
    words = word_tokenize(text)

    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    cleaned_text = preprocess_text(text)
    return cleaned_text


# Example usage
if __name__ == "__main__":
    input_file = "input.txt"  # Change this to your file path
    processed_text = process_file(input_file)
    print(processed_text)
