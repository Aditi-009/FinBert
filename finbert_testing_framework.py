import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import mean_absolute_error, mean_squared_error, pearsonr, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Dict, Tuple
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSentimentDataset(Dataset):
    """Custom dataset for SemEval 2017 Task 5 financial sentiment data"""
    
    def __init__(self, texts: List[str], targets: List[str], sentiments: List[float], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.targets = targets
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = str(self.targets[idx])
        sentiment = self.sentiments[idx]
        
        # Combine text and target for context
        combined_text = f"[CLS] {text} [SEP] {target} [SEP]"
        
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(sentiment, dtype=torch.float32)
        }

class FinBERTTargetSentimentAnalyzer:
    """FinBERT implementation for target-based sentiment analysis"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_entities(self, text: str) -> List[str]:
        """Extract potential target entities from text"""
        # Simple entity extraction - can be enhanced with NER
        # Look for company names, stock symbols, financial terms
        entities = []
        
        # Extract potential stock symbols (uppercase letters)
        stock_symbols = re.findall(r'\b[A-Z]{1,5}\b', text)
        entities.extend(stock_symbols)
        
        # Extract capitalized words (potential company names)
        company_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(company_names)
        
        # Remove duplicates and common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'It', 'We', 'They'}
        entities = list(set([e for e in entities if e not in common_words]))
        
        return entities if entities else ['general']
    
    def predict_sentiment(self, text: str, target: str) -> float:
        """Predict sentiment score for a given text-target pair"""
        combined_text = f"{text} [SEP] {target}"
        
        inputs = self.tokenizer(
            combined_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to sentiment score (-1 to 1)
            probabilities = torch.softmax(logits, dim=-1)
            # Assuming 3 classes: negative, neutral, positive
            sentiment_score = (probabilities[0][2] - probabilities[0][0]).item()
            
        return sentiment_score
    
    def batch_predict(self, texts: List[str], targets: List[str]) -> List[float]:
        """Batch prediction for efficiency"""
        predictions = []
        
        for text, target in zip(texts, targets):
            score = self.predict_sentiment(text, target)
            predictions.append(score)
            
        return predictions

class SemEvalDataLoader:
    """Load and preprocess SemEval 2017 Task 5 data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_data(self, split: str = 'train') -> Tuple[List[str], List[str], List[float]]:
        """Load SemEval data format"""
        # Expected format: text, target_entity, sentiment_score
        # This is a placeholder - adjust based on actual data format
        
        if split == 'train':
            # Load training data (1,142 headlines)
            file_path = f"{self.data_path}/Headline_Trainingdata.json"
        else:
            # Load test data (491 headlines)
            file_path = f"{self.data_path}/Headline_Testdata.json"
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            texts = []
            targets = []
            sentiments = []
            
            for item in data:
                texts.append(item['text'])
                targets.append(item['target'])
                sentiments.append(float(item['sentiment']))
                
            return texts, targets, sentiments
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            # Return dummy data for testing
            return self._generate_dummy_data(split)
    
    def _generate_dummy_data(self, split: str) -> Tuple[List[str], List[str], List[float]]:
        """Generate dummy data for testing purposes"""
        if split == 'train':
            size = 100  # Smaller for demo
        else:
            size = 50
            
        texts = [
            f"Company ABC reported strong earnings this quarter {i}" 
            for i in range(size)
        ]
        targets = [f"ABC" for _ in range(size)]
        sentiments = [np.random.uniform(-1, 1) for _ in range(size)]
        
        return texts, targets, sentiments

class SentimentEvaluator:
    """Evaluation metrics for sentiment analysis"""
    
    @staticmethod
    def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Correlation
        pearson_corr, _ = pearsonr(y_true, y_pred)
        
        # Classification metrics (convert to discrete classes)
        y_true_class = np.where(y_true > 0.1, 1, np.where(y_true < -0.1, -1, 0))
        y_pred_class = np.where(y_pred > 0.1, 1, np.where(y_pred < -0.1, -1, 0))
        
        accuracy = accuracy_score(y_true_class, y_pred_class)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'pearson_correlation': pearson_corr,
            'accuracy': accuracy,
            'cosine_similarity': np.dot(y_true, y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Print formatted metrics"""
        print("\n=== EVALUATION METRICS ===")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        print(f"Classification Accuracy: {metrics['accuracy']:.4f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        print("=" * 27)

def main():
    """Main testing pipeline"""
    
    # Initialize components
    print("Initializing FinBERT Target Sentiment Analyzer...")
    analyzer = FinBERTTargetSentimentAnalyzer()
    
    # Load data
    print("Loading SemEval 2017 Task 5 data...")
    data_loader = SemEvalDataLoader('./data')  # Adjust path as needed
    
    # Load training data
    train_texts, train_targets, train_sentiments = data_loader.load_data('train')
    print(f"Training data loaded: {len(train_texts)} samples")
    
    # Load test data
    test_texts, test_targets, test_sentiments = data_loader.load_data('test')
    print(f"Test data loaded: {len(test_texts)} samples")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    predictions = analyzer.batch_predict(test_texts, test_targets)
    
    # Evaluate results
    print("Evaluating results...")
    evaluator = SentimentEvaluator()
    metrics = evaluator.calculate_metrics(test_sentiments, predictions)
    evaluator.print_metrics(metrics)
    
    # Entity detection evaluation
    print("\n=== ENTITY DETECTION EVALUATION ===")
    entity_detection_accuracy = []
    for i, text in enumerate(test_texts[:10]):  # Sample evaluation
        detected_entities = analyzer.extract_entities(text)
        actual_target = test_targets[i]
        
        # Check if actual target is in detected entities
        is_correct = actual_target in detected_entities
        entity_detection_accuracy.append(is_correct)
        
        print(f"Text: {text[:50]}...")
        print(f"Detected: {detected_entities}")
        print(f"Actual: {actual_target}")
        print(f"Correct: {is_correct}")
        print("-" * 40)
    
    entity_accuracy = np.mean(entity_detection_accuracy)
    print(f"Entity Detection Accuracy: {entity_accuracy:.4f}")
    
    # Save results
    results = {
        'model': 'FinBERT',
        'dataset': 'SemEval 2017 Task 5',
        'test_samples': len(test_texts),
        'metrics': metrics,
        'entity_detection_accuracy': entity_accuracy
    }
    
    with open('finbert_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'finbert_results.json'")

if __name__ == "__main__":
    main()
