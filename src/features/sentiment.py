import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np

class FinBERTSentimentAnalyzer:
    """Financial news sentiment analysis using FinBERT."""
    
    def __init__(self, device="/gpu:0"):
        """
        Initialize the FinBERT sentiment analyzer.
        
        Args:
            device: Device to run the model on
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        with tf.device(self.device):
            self.model = TFBertModel.from_pretrained('yiyanghkust/finbert-tone')
            
    def batch_sentiment_analysis(self, text_list, batch_size=32):
        """
        Calculate sentiment scores for a list of texts.
        
        Args:
            text_list: List of text strings to analyze
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of sentiment scores
        """
        sentiment_scores = []
        
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="tf", truncation=True, padding=True, max_length=512)
            
            with tf.device(self.device):
                outputs = self.model(inputs)
                batch_scores = tf.reduce_mean(outputs.last_hidden_state[:, 0, :], axis=1).numpy()
                
            sentiment_scores.extend(batch_scores)
            
        return np.array(sentiment_scores)
