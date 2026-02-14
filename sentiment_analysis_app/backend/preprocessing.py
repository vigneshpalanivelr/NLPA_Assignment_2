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
        Complete preprocessing pipeline with detailed step tracking

        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization

        Returns:
            dict: Dictionary containing original text, intermediate steps, and final processed tokens
        """
        # Step 1: Original text
        original_word_count = len(text.split())

        # Step 2: Text cleaning (lowercase, remove URLs, mentions, special chars)
        cleaned_text = self.clean_text(text)

        # Step 3: Tokenization
        tokens_after_tokenization = self.tokenize(cleaned_text)

        # Step 4: Stopword removal
        tokens_after_stopword_removal = None
        if remove_stops:
            tokens_after_stopword_removal = self.remove_stopwords(tokens_after_tokenization)
            working_tokens = tokens_after_stopword_removal
        else:
            working_tokens = tokens_after_tokenization

        # Step 5: Stemming or Lemmatization
        tokens_after_stemming = None
        tokens_after_lemmatization = None

        if use_stemming:
            tokens_after_stemming = self.stem_tokens(working_tokens)
            final_tokens = tokens_after_stemming
        elif use_lemmatization:
            tokens_after_lemmatization = self.lemmatize_tokens(working_tokens)
            final_tokens = tokens_after_lemmatization
        else:
            final_tokens = working_tokens

        return {
            # Original input
            'original_text': text,
            'original_word_count': original_word_count,

            # Step-by-step preprocessing details
            'steps': {
                '1_original': {
                    'text': text,
                    'word_count': original_word_count,
                    'description': 'Original input text'
                },
                '2_cleaned': {
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()) if cleaned_text else 0,
                    'description': 'After lowercasing, URL removal, mention/hashtag removal, special character removal'
                },
                '3_tokenized': {
                    'tokens': tokens_after_tokenization,
                    'token_count': len(tokens_after_tokenization),
                    'description': 'After tokenization into words'
                },
                '4_stopwords_removed': {
                    'tokens': tokens_after_stopword_removal if remove_stops else tokens_after_tokenization,
                    'token_count': len(tokens_after_stopword_removal) if remove_stops else len(tokens_after_tokenization),
                    'description': 'After removing common stopwords' if remove_stops else 'Stopword removal skipped',
                    'applied': remove_stops
                },
                '5_stemmed_or_lemmatized': {
                    'tokens': final_tokens,
                    'token_count': len(final_tokens),
                    'description': (
                        'After stemming (reducing to root form)' if use_stemming
                        else 'After lemmatization (reducing to dictionary form)' if use_lemmatization
                        else 'No stemming or lemmatization applied'
                    ),
                    'method': 'stemming' if use_stemming else 'lemmatization' if use_lemmatization else 'none'
                }
            },

            # Final results
            'cleaned_text': cleaned_text,
            'tokens': final_tokens,
            'token_count': len(final_tokens),
            'processed_text': ' '.join(final_tokens)
        }
