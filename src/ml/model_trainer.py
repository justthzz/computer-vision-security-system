"""
Machine Learning Model Training Module
Demonstrates ML engineering skills for computer vision applications.
"""

import os
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import time
from datetime import datetime
import logging

class DatasetManager:
    """Manages training datasets for computer vision models."""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.annotations = []
        self.classes = []
        self.setup_directories()
    
    def setup_directories(self):
        """Setup training data directory structure."""
        dirs = [
            self.data_dir,
            os.path.join(self.data_dir, "images"),
            os.path.join(self.data_dir, "annotations"),
            os.path.join(self.data_dir, "models"),
            os.path.join(self.data_dir, "logs")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def add_annotation(self, image_path: str, bboxes: List[Tuple], 
                      class_names: List[str], confidence: float = 1.0):
        """Add annotation to dataset."""
        annotation = {
            'image_path': image_path,
            'bboxes': bboxes,
            'class_names': class_names,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.annotations.append(annotation)
    
    def save_annotations(self, filename: str = None):
        """Save annotations to JSON file."""
        if filename is None:
            filename = f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, "annotations", filename)
        
        data = {
            'annotations': self.annotations,
            'classes': self.classes,
            'total_images': len(self.annotations),
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Annotations saved to {filepath}")
        return filepath
    
    def load_annotations(self, filepath: str):
        """Load annotations from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.classes = data['classes']
        
        logging.info(f"Loaded {len(self.annotations)} annotations")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.annotations:
            return {}
        
        class_counts = {}
        total_objects = 0
        
        for ann in self.annotations:
            for class_name in ann['class_names']:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_objects += 1
        
        return {
            'total_images': len(self.annotations),
            'total_objects': total_objects,
            'class_distribution': class_counts,
            'average_objects_per_image': total_objects / len(self.annotations) if self.annotations else 0
        }

class ModelTrainer:
    """Base class for model training."""
    
    def __init__(self, model_name: str, data_manager: DatasetManager):
        self.model_name = model_name
        self.data_manager = data_manager
        self.model = None
        self.training_history = []
        self.is_trained = False
        
    def prepare_data(self, train_split: float = 0.8) -> Tuple[List, List]:
        """Prepare training and validation data."""
        annotations = self.data_manager.annotations
        
        # Simple train/validation split
        split_idx = int(len(annotations) * train_split)
        train_data = annotations[:split_idx]
        val_data = annotations[split_idx:]
        
        logging.info(f"Data split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def train(self, epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train method")
    
    def evaluate(self, test_data: List) -> Dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def save_model(self, filepath: str):
        """Save trained model."""
        raise NotImplementedError("Subclasses must implement save_model method")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        raise NotImplementedError("Subclasses must implement load_model method")

class CustomDetectorTrainer(ModelTrainer):
    """Trainer for custom detection models."""
    
    def __init__(self, data_manager: DatasetManager):
        super().__init__("CustomDetector", data_manager)
        self.feature_extractor = None
        self.classifier = None
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image using traditional CV methods."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Extract HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(blurred)
        
        return features.flatten()
    
    def train(self, epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train custom detection model."""
        logging.info(f"Starting training for {self.model_name}")
        
        train_data, val_data = self.prepare_data()
        
        if not train_data:
            raise ValueError("No training data available")
        
        # Extract features from training data
        X_train = []
        y_train = []
        
        for ann in train_data:
            try:
                image = cv2.imread(ann['image_path'])
                if image is None:
                    continue
                
                features = self.extract_features(image)
                X_train.append(features)
                
                # Create label vector
                label = [1 if 'person' in ann['class_names'] else 0]
                y_train.append(label)
                
            except Exception as e:
                logging.warning(f"Error processing {ann['image_path']}: {e}")
                continue
        
        if not X_train:
            raise ValueError("No valid training samples found")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Simple logistic regression classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train classifier
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier.fit(X_train_scaled, y_train.ravel())
        
        # Evaluate on validation data
        val_accuracy = self.evaluate(val_data)
        
        self.is_trained = True
        
        training_results = {
            'model_name': self.model_name,
            'training_samples': len(X_train),
            'validation_samples': len(val_data),
            'accuracy': val_accuracy.get('accuracy', 0.0),
            'training_time': time.time(),
            'epochs': epochs
        }
        
        self.training_history.append(training_results)
        logging.info(f"Training completed. Accuracy: {val_accuracy.get('accuracy', 0.0):.3f}")
        
        return training_results
    
    def evaluate(self, test_data: List) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained or not self.classifier:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        X_test = []
        y_test = []
        
        for ann in test_data:
            try:
                image = cv2.imread(ann['image_path'])
                if image is None:
                    continue
                
                features = self.extract_features(image)
                X_test.append(features)
                
                label = [1 if 'person' in ann['class_names'] else 0]
                y_test.append(label)
                
            except Exception as e:
                logging.warning(f"Error processing {ann['image_path']}: {e}")
                continue
        
        if not X_test:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Predict
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def predict(self, image: np.ndarray) -> Tuple[bool, float]:
        """Predict if image contains a person."""
        if not self.is_trained or not self.classifier:
            return False, 0.0
        
        try:
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            
            # Scale features (would need to save scaler in real implementation)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            prediction = self.classifier.predict(features_scaled)[0]
            confidence = self.classifier.predict_proba(features_scaled)[0][1]
            
            return bool(prediction), confidence
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return False, 0.0
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model_name': self.model_name,
            'classifier': self.classifier,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        # In a real implementation, you'd use joblib or pickle
        # For demo purposes, we'll save metadata
        metadata = {
            'model_name': self.model_name,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.training_history = metadata['training_history']
        self.is_trained = metadata['is_trained']
        
        # In real implementation, load the actual classifier
        logging.info(f"Model loaded from {filepath}")

class ModelEvaluator:
    """Evaluates and compares different models."""
    
    def __init__(self):
        self.evaluation_results = []
    
    def evaluate_model(self, model: ModelTrainer, test_data: List) -> Dict[str, Any]:
        """Evaluate a model and return comprehensive results."""
        start_time = time.time()
        
        # Run evaluation
        metrics = model.evaluate(test_data)
        
        # Calculate additional metrics
        evaluation_time = time.time() - start_time
        
        results = {
            'model_name': model.model_name,
            'evaluation_time': evaluation_time,
            'test_samples': len(test_data),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.evaluation_results.append(results)
        return results
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare all evaluated models."""
        if not self.evaluation_results:
            return {}
        
        # Find best model by accuracy
        best_model = max(self.evaluation_results, 
                        key=lambda x: x['metrics'].get('accuracy', 0))
        
        comparison = {
            'total_models': len(self.evaluation_results),
            'best_model': best_model['model_name'],
            'best_accuracy': best_model['metrics'].get('accuracy', 0),
            'all_results': self.evaluation_results
        }
        
        return comparison
    
    def generate_report(self, filepath: str):
        """Generate evaluation report."""
        comparison = self.compare_models()
        
        report = {
            'evaluation_summary': comparison,
            'detailed_results': self.evaluation_results,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Evaluation report saved to {filepath}")

class TrainingPipeline:
    """Complete training pipeline for computer vision models."""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_manager = DatasetManager(data_dir)
        self.trainers = {}
        self.evaluator = ModelEvaluator()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for training pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_training_data(self, image_path: str, bboxes: List[Tuple], 
                         class_names: List[str], confidence: float = 1.0):
        """Add training data to dataset."""
        self.data_manager.add_annotation(image_path, bboxes, class_names, confidence)
    
    def train_custom_detector(self, epochs: int = 10) -> Dict[str, Any]:
        """Train custom detection model."""
        trainer = CustomDetectorTrainer(self.data_manager)
        results = trainer.train(epochs=epochs)
        
        self.trainers['custom_detector'] = trainer
        return results
    
    def evaluate_all_models(self, test_data: List) -> Dict[str, Any]:
        """Evaluate all trained models."""
        results = {}
        
        for name, trainer in self.trainers.items():
            if trainer.is_trained:
                results[name] = self.evaluator.evaluate_model(trainer, test_data)
        
        return results
    
    def generate_training_report(self, filepath: str = None):
        """Generate comprehensive training report."""
        if filepath is None:
            filepath = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Get dataset stats
        dataset_stats = self.data_manager.get_dataset_stats()
        
        # Get evaluation results
        evaluation_comparison = self.evaluator.compare_models()
        
        # Get training history
        training_history = []
        for trainer in self.trainers.values():
            training_history.extend(trainer.training_history)
        
        report = {
            'dataset_statistics': dataset_stats,
            'training_history': training_history,
            'evaluation_results': evaluation_comparison,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training report saved to {filepath}")
        return filepath
