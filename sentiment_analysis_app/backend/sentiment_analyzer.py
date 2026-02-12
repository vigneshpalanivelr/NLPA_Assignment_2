"""
Sentiment Analyzer Module using Deep Learning
This module implements sentiment analysis using:
- DistilBERT (Transformer-based model from Hugging Face)
- TensorFlow backend via transformers library
- Pre-trained on sentiment analysis tasks
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using DistilBERT transformer model
    """

    def __init__(self):
        """
        Initialize sentiment analyzer with DistilBERT model and preprocessor
        """
        # Use DistilBERT fine-tuned for sentiment analysis
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

        print(f"Loading BERT model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        print("BERT model loaded successfully!")

        # Set model to evaluation mode
        self.model.eval()

        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()

        # Label mapping for the model
        self.label_map = {0: 'negative', 1: 'positive'}

    def analyze_with_bert(self, text):
        """
        Analyze sentiment using DistilBERT transformer model

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Sentiment prediction with probabilities for each class
        """
        # Tokenize the input text for BERT
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Extract probabilities
        probs = predictions[0].tolist()
        negative_prob = probs[0]
        positive_prob = probs[1]

        # Calculate neutral score (when positive and negative are both moderate)
        # If both positive and negative are close, it indicates neutrality
        neutral_prob = 1.0 - max(positive_prob, negative_prob)

        # Determine primary sentiment
        if positive_prob > 0.6:
            sentiment = 'positive'
            confidence = positive_prob
        elif negative_prob > 0.6:
            sentiment = 'negative'
            confidence = negative_prob
        else:
            sentiment = 'neutral'
            confidence = neutral_prob

        return {
            'method': 'DistilBERT (Transformer)',
            'model': self.model_name,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'probabilities': {
                'positive': round(positive_prob, 4),
                'negative': round(negative_prob, 4),
                'neutral': round(neutral_prob, 4)
            },
            'scores': {
                'positive_score': round(positive_prob, 4),
                'negative_score': round(negative_prob, 4),
                'neutral_score': round(neutral_prob, 4)
            }
        }

    def analyze_combined(self, text):
        """
        Perform comprehensive sentiment analysis using BERT
        Also applies detailed preprocessing to the text

        Args:
            text (str): Input text to analyze

        Returns:
            dict: Complete analysis results including preprocessing and BERT prediction
        """
        # Preprocess the text with detailed steps
        preprocessed = self.preprocessor.preprocess(text)

        # Analyze using BERT (on original text - BERT handles its own tokenization)
        bert_results = self.analyze_with_bert(text)

        # Also analyze the cleaned text for comparison
        bert_results_cleaned = self.analyze_with_bert(preprocessed['cleaned_text'])

        # Use the original text analysis as primary, but note if cleaned gives different result
        primary_sentiment = bert_results['sentiment']
        confidence = bert_results['confidence']

        return {
            'text': text,
            'preprocessing': {
                'original_word_count': preprocessed['original_word_count'],
                'cleaned_text': preprocessed['cleaned_text'],
                'tokens': preprocessed['tokens'],
                'token_count': preprocessed['token_count'],
                'steps': preprocessed['steps']  # Detailed preprocessing steps
            },
            'bert_analysis': {
                'original_text': bert_results,
                'cleaned_text': bert_results_cleaned
            },
            'final_sentiment': primary_sentiment,
            'confidence': confidence,
            'sentiment_scores': bert_results['scores'],
            'model_info': {
                'name': 'DistilBERT',
                'full_name': self.model_name,
                'type': 'Transformer-based Deep Learning',
                'framework': 'PyTorch/Transformers'
            }
        }

    def analyze_batch(self, texts):
        """
        Analyze multiple texts in batch

        Args:
            texts (list): List of text strings to analyze

        Returns:
            list: List of analysis results for each text
        """
        return [self.analyze_combined(text) for text in texts]
