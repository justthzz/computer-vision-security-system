#!/usr/bin/env python3
"""
Machine Learning Training Demo
Demonstrates ML engineering skills for computer vision applications.
"""

import cv2
import numpy as np
import os
import sys
from src.ml.model_trainer import TrainingPipeline, DatasetManager
import logging

def create_synthetic_training_data(data_dir: str, num_samples: int = 100):
    """Create synthetic training data for demonstration."""
    print(f"Creating {num_samples} synthetic training samples...")
    
    # Create directories
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    
    # Create synthetic images with and without people
    for i in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        cv2.rectangle(image, (100, 100), (200, 300), (128, 128, 128), -1)
        cv2.circle(image, (300, 200), 50, (64, 64, 64), -1)
        
        # Randomly add a "person" (simple rectangle)
        has_person = np.random.random() > 0.5
        
        if has_person:
            # Add person-like shape
            person_x = np.random.randint(50, 500)
            person_y = np.random.randint(50, 400)
            person_w = np.random.randint(30, 80)
            person_h = np.random.randint(80, 150)
            
            cv2.rectangle(image, (person_x, person_y), 
                         (person_x + person_w, person_y + person_h), 
                         (0, 0, 255), -1)
            
            # Add head
            head_center = (person_x + person_w//2, person_y - 20)
            cv2.circle(image, head_center, 15, (0, 0, 255), -1)
        
        # Save image
        image_path = os.path.join(data_dir, "images", f"sample_{i:04d}.jpg")
        cv2.imwrite(image_path, image)
        
        # Add annotation
        bboxes = []
        class_names = []
        
        if has_person:
            bboxes.append((person_x, person_y, person_w, person_h))
            class_names.append("person")
        else:
            bboxes.append((0, 0, 640, 480))  # Full image bbox for negative samples
            class_names.append("background")
    
    print(f"✅ Created {num_samples} synthetic training samples")

def demo_training_pipeline():
    """Demonstrate the complete training pipeline."""
    print("🎯 MACHINE LEARNING TRAINING DEMO")
    print("=" * 50)
    print("This demo shows:")
    print("• Dataset management and annotation")
    print("• Model training with custom features")
    print("• Model evaluation and comparison")
    print("• Performance metrics and reporting")
    print("=" * 50)
    
    # Create training pipeline
    pipeline = TrainingPipeline("demo_training_data")
    
    # Create synthetic training data
    create_synthetic_training_data("demo_training_data", num_samples=50)
    
    # Add training data to pipeline
    print("\n📊 Adding training data to pipeline...")
    for i in range(50):
        image_path = f"demo_training_data/images/sample_{i:04d}.jpg"
        
        # Determine if image has person (based on our synthetic data creation)
        has_person = i % 2 == 0  # Simple pattern for demo
        
        if has_person:
            bboxes = [(100, 100, 80, 150)]  # Example bbox
            class_names = ["person"]
        else:
            bboxes = [(0, 0, 640, 480)]
            class_names = ["background"]
        
        pipeline.add_training_data(image_path, bboxes, class_names)
    
    print(f"✅ Added {len(pipeline.data_manager.annotations)} training samples")
    
    # Show dataset statistics
    stats = pipeline.data_manager.get_dataset_stats()
    print("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Train custom detector
    print("\n🤖 Training custom detection model...")
    try:
        training_results = pipeline.train_custom_detector(epochs=5)
        print("✅ Training completed!")
        print(f"  Accuracy: {training_results['accuracy']:.3f}")
        print(f"  Training samples: {training_results['training_samples']}")
        print(f"  Validation samples: {training_results['validation_samples']}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    try:
        # Use some of the training data as test data for demo
        test_data = pipeline.data_manager.annotations[:10]
        evaluation_results = pipeline.evaluate_all_models(test_data)
        
        for model_name, results in evaluation_results.items():
            print(f"\n{model_name} Results:")
            print(f"  Accuracy: {results['metrics']['accuracy']:.3f}")
            print(f"  Precision: {results['metrics']['precision']:.3f}")
            print(f"  Recall: {results['metrics']['recall']:.3f}")
            print(f"  Evaluation time: {results['evaluation_time']:.3f}s")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return
    
    # Generate training report
    print("\n📋 Generating training report...")
    try:
        report_path = pipeline.generate_training_report()
        print(f"✅ Training report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n🎉 Training demo completed!")

def demo_model_prediction():
    """Demonstrate model prediction on new images."""
    print("\n🎯 MODEL PREDICTION DEMO")
    print("=" * 30)
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a person-like shape
    cv2.rectangle(test_image, (200, 150), (280, 350), (0, 0, 255), -1)
    cv2.circle(test_image, (240, 120), 20, (0, 0, 255), -1)
    
    # Save test image
    cv2.imwrite("test_prediction.jpg", test_image)
    print("✅ Created test image: test_prediction.jpg")
    
    # Try to load and use trained model
    try:
        pipeline = TrainingPipeline("demo_training_data")
        
        # Check if we have a trained model
        if 'custom_detector' in pipeline.trainers:
            trainer = pipeline.trainers['custom_detector']
            
            if trainer.is_trained:
                # Make prediction
                has_person, confidence = trainer.predict(test_image)
                
                print(f"\n🔮 Prediction Results:")
                print(f"  Has person: {has_person}")
                print(f"  Confidence: {confidence:.3f}")
                
                # Visualize result
                result_image = test_image.copy()
                if has_person:
                    cv2.putText(result_image, f"PERSON DETECTED ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(result_image, f"NO PERSON ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imwrite("prediction_result.jpg", result_image)
                print("✅ Prediction result saved: prediction_result.jpg")
            else:
                print("❌ No trained model available")
        else:
            print("❌ No trained model available")
    
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def show_ml_capabilities():
    """Show ML capabilities and features."""
    print("🎯 MACHINE LEARNING CAPABILITIES")
    print("=" * 50)
    print("📊 FEATURES:")
    print("• Custom dataset management")
    print("• Feature extraction (HOG, SIFT, etc.)")
    print("• Model training and evaluation")
    print("• Performance metrics and reporting")
    print("• Model comparison and selection")
    print("• Real-time prediction")
    print("\n🤖 ALGORITHMS:")
    print("• Logistic Regression")
    print("• Support Vector Machines")
    print("• Random Forest")
    print("• Neural Networks (extensible)")
    print("\n📈 METRICS:")
    print("• Accuracy, Precision, Recall")
    print("• F1-Score, ROC-AUC")
    print("• Confusion Matrix")
    print("• Training/Validation Curves")
    print("\n🛠️ TOOLS:")
    print("• Scikit-learn integration")
    print("• OpenCV feature extraction")
    print("• Automated hyperparameter tuning")
    print("• Cross-validation")
    print("• Model persistence")
    print("=" * 50)

def main():
    """Main demo function."""
    print("🎯 MACHINE LEARNING TRAINING DEMO")
    print("=" * 50)
    print("Select a demo:")
    print("1. Complete Training Pipeline")
    print("2. Model Prediction Demo")
    print("3. Show ML Capabilities")
    print("0. Exit")
    print("=" * 50)
    
    while True:
        choice = input("Enter your choice (0-3): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            demo_training_pipeline()
        elif choice == '2':
            demo_model_prediction()
        elif choice == '3':
            show_ml_capabilities()
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
