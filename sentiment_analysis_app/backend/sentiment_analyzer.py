"""
Sentiment Analyzer Module
This module implements sentiment analysis using multiple NLP techniques:
- VADER (Valence Aware Dictionary and sEntiment Reasoner) for social media text
- TextBlob for general text sentiment analysis
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from preprocessing import TextPreprocessor

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on text using multiple approaches
    """

    def __init__(self):
        """
        Initialize sentiment analyzer with VADER and TextPreprocessor
        """
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.preprocessor = TextPreprocessor()

    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER (best for social media and short text)

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Sentiment scores including compound, positive, negative, neutral
        """
        scores = self.vader_analyzer.polarity_scores(text)

        # Determine overall sentiment based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'method': 'VADER',
            'sentiment': sentiment,
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu']
        }

    def analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob (general purpose)

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Sentiment scores including polarity and subjectivity
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'method': 'TextBlob',
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }

    def analyze_combined(self, text):
        """
        Perform comprehensive sentiment analysis using both VADER and TextBlob
        Also applies preprocessing to the text

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Complete analysis results including preprocessing and both methods
        """
        # Preprocess the text
        preprocessed = self.preprocessor.preprocess(text)

        # Analyze using VADER (on original text - VADER works better with original punctuation)
        vader_results = self.analyze_vader(text)

        # Analyze using TextBlob (on cleaned text)
        textblob_results = self.analyze_textblob(preprocessed['cleaned_text'])

        # Determine final sentiment by combining both methods
        # VADER is weighted more heavily as it's more robust for varied text types
        final_sentiment = self._determine_final_sentiment(
            vader_results['sentiment'],
            textblob_results['sentiment'],
            vader_results['compound_score']
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(vader_results, textblob_results)

        return {
            'text': text,
            'preprocessing': {
                'cleaned_text': preprocessed['cleaned_text'],
                'tokens': preprocessed['tokens'],
                'token_count': len(preprocessed['tokens'])
            },
            'vader_analysis': vader_results,
            'textblob_analysis': textblob_results,
            'final_sentiment': final_sentiment,
            'confidence': confidence,
            'sentiment_scores': {
                'positive': vader_results['positive_score'],
                'negative': vader_results['negative_score'],
                'neutral': vader_results['neutral_score']
            }
        }

    def _determine_final_sentiment(self, vader_sentiment, textblob_sentiment, compound_score):
        """
        Determine final sentiment by combining VADER and TextBlob results

        Args:
            vader_sentiment (str): VADER sentiment result
            textblob_sentiment (str): TextBlob sentiment result
            compound_score (float): VADER compound score

        Returns:
            str: Final sentiment classification
        """
        # If both agree, return that sentiment
        if vader_sentiment == textblob_sentiment:
            return vader_sentiment

        # If they disagree, use VADER's result if compound score is strong
        if abs(compound_score) > 0.3:
            return vader_sentiment

        # Otherwise, default to neutral for ambiguous cases
        return 'neutral'

    def _calculate_confidence(self, vader_results, textblob_results):
        """
        Calculate confidence score based on agreement between methods

        Args:
            vader_results (dict): VADER analysis results
            textblob_results (dict): TextBlob analysis results

        Returns:
            float: Confidence score between 0 and 1
        """
        # Check if sentiments agree
        agreement = 1.0 if vader_results['sentiment'] == textblob_results['sentiment'] else 0.5

        # Factor in the strength of VADER compound score
        vader_strength = abs(vader_results['compound_score'])

        # Factor in TextBlob polarity strength
        textblob_strength = abs(textblob_results['polarity'])

        # Calculate weighted confidence
        confidence = (agreement * 0.4) + (vader_strength * 0.4) + (textblob_strength * 0.2)

        return round(min(confidence, 1.0), 2)

    def analyze_batch(self, texts):
        """
        Analyze multiple texts in batch

        Args:
            texts (list): List of text strings to analyze

        Returns:
            list: List of analysis results for each text
        """
        return [self.analyze_combined(text) for text in texts]
