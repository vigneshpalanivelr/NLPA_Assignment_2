"""
Text Preprocessing Module for Sentiment Analysis
This module handles all text preprocessing operations including:
- Tokenization
- Stemming
- Lemmatization
- Text cleaning and normalization
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextPreprocessor:
    """
    A class to handle text preprocessing for sentiment analysis
    """

    def __init__(self):
        """
        Initialize the preprocessor with necessary NLTK components
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Clean text by removing URLs, mentions, special characters

        Args:
            text (str): Raw input text

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and numbers, keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text):
        """
        Tokenize text into words

        Args:
            text (str): Input text

        Returns:
            list: List of tokens
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list

        Args:
            tokens (list): List of tokens

        Returns:
            list: Filtered tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        """
        Apply stemming to tokens using Porter Stemmer

        Args:
            tokens (list): List of tokens

        Returns:
            list: Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize_tokens(self, tokens):
        """
        Apply lemmatization to tokens

        Args:
            tokens (list): List of tokens

        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess(self, text, remove_stops=True, use_stemming=False, use_lemmatization=True):
        """
        Complete preprocessing pipeline

        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization

        Returns:
            dict: Dictionary containing original text, cleaned text, and processed tokens
        """
        # Count words in original text (split by whitespace)
        original_word_count = len(text.split())

        # Clean the text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned_text)

        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)

        # Apply stemming if requested
        if use_stemming:
            tokens = self.stem_tokens(tokens)

        # Apply lemmatization if requested (preferred over stemming)
        if use_lemmatization and not use_stemming:
            tokens = self.lemmatize_tokens(tokens)

        return {
            'original_text': text,
            'original_word_count': original_word_count,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'processed_text': ' '.join(tokens)
        }
