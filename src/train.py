#!/usr/bin/env python3
"""
California Housing Price Prediction - Model Training Pipeline

This module implements the core training pipeline for the California housing
price prediction model using advanced linear regression techniques with
comprehensive logging and performance monitoring.

Author: ML Engineering Team
Date: 2024
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class HousingPricePredictor:
    """
    Advanced housing price prediction model with comprehensive training pipeline.
    
    This class encapsulates the entire training workflow including data loading,
    model training, performance evaluation, and model persistence.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the housing price predictor.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.training_metrics = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare the California housing dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) arrays
        """
        logger.info("🔄 Loading California Housing dataset...")
        
        try:
            from sklearn.datasets import fetch_california_housing
            from sklearn.model_selection import train_test_split
            
            # Fetch dataset
            housing_data = fetch_california_housing()
            X, y = housing_data.data, housing_data.target
            
            # Perform train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            logger.info(f"✅ Dataset loaded successfully:")
            logger.info(f"   Training samples: {X_train.shape[0]}")
            logger.info(f"   Test samples: {X_test.shape[0]}")
            logger.info(f"   Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {str(e)}")
            raise
    
    def create_model(self) -> LinearRegression:
        """
        Create and configure the linear regression model.
        
        Returns:
            Configured LinearRegression model
        """
        logger.info("🔧 Creating optimized LinearRegression model...")
        
        try:
            # Initialize model with optimized parameters
            model = LinearRegression(
                fit_intercept=True,
                copy_X=True,
                n_jobs=-1,  # Use all available cores
                positive=False
            )
            
            logger.info("✅ Model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {str(e)}")
            raise
    
    def train_model(self, model: LinearRegression, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """
        Train the linear regression model with comprehensive validation.
        
        Args:
            model: LinearRegression model instance
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        logger.info("🚀 Training model with advanced techniques...")
        
        try:
            # Perform cross-validation for model validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            logger.info(f"📊 Cross-validation R² scores: {cv_scores}")
            logger.info(f"📊 Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train the model on full training set
            model.fit(X_train, y_train)
            
            # Validate training success
            if not hasattr(model, 'coef_') or model.coef_ is None:
                raise ValueError("Model training failed - coefficients not found")
            
            logger.info("✅ Model training completed successfully")
            logger.info(f"📈 Model coefficients shape: {model.coef_.shape}")
            logger.info(f"📈 Model intercept: {model.intercept_:.6f}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            raise
    
    def evaluate_model(self, model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("📊 Evaluating model performance...")
        
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            # Log performance metrics
            logger.info("📈 Model Performance Metrics:")
            logger.info(f"   R² Score: {metrics['r2_score']:.4f}")
            logger.info(f"   Mean Squared Error: {metrics['mse']:.4f}")
            logger.info(f"   Root Mean Squared Error: {metrics['rmse']:.4f}")
            logger.info(f"   Mean Absolute Error: {metrics['mae']:.4f}")
            logger.info(f"   Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
            
            # Performance validation
            if metrics['r2_score'] < 0.5:
                logger.warning("⚠️  Model performance below threshold (R² < 0.5)")
            else:
                logger.info("✅ Model performance meets requirements")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def save_model(self, model: LinearRegression, metrics: dict) -> str:
        """
        Save the trained model and metadata.
        
        Args:
            model: Trained model
            metrics: Performance metrics
            
        Returns:
            Path to saved model file
        """
        logger.info("💾 Saving model and metadata...")
        
        try:
            # Prepare model artifacts
            model_artifacts = {
                'model': model,
                'metrics': metrics,
                'feature_names': [
                    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude'
                ],
                'model_info': {
                    'algorithm': 'LinearRegression',
                    'version': '1.0.0',
                    'training_date': pd.Timestamp.now().isoformat()
                }
            }
            
            # Save model artifacts
            model_path = self.model_dir / "housing_price_model.joblib"
            joblib.dump(model_artifacts, model_path)
            
            logger.info(f"✅ Model saved successfully to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
            raise
    
    def run_training_pipeline(self) -> Tuple[LinearRegression, dict]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Tuple of (trained_model, performance_metrics)
        """
        logger.info("🎯 Starting Housing Price Prediction Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # Step 2: Create model
            model = self.create_model()
            
            # Step 3: Train model
            trained_model = self.train_model(model, X_train, y_train)
            
            # Step 4: Evaluate model
            metrics = self.evaluate_model(trained_model, X_test, y_test)
            
            # Step 5: Save model
            model_path = self.save_model(trained_model, metrics)
            
            # Store metrics for later use
            self.training_metrics = metrics
            self.model = trained_model
            
            logger.info("=" * 60)
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"📁 Model artifacts saved to: {model_path}")
            
            return trained_model, metrics
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise


def main():
    """
    Main entry point for the training pipeline.
    """
    try:
        # Initialize predictor
        predictor = HousingPricePredictor()
        
        # Run training pipeline
        model, metrics = predictor.run_training_pipeline()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏠 CALIFORNIA HOUSING PRICE PREDICTOR - TRAINING SUMMARY")
        print("=" * 60)
        print(f"📊 R² Score: {metrics['r2_score']:.4f}")
        print(f"📊 Mean Squared Error: {metrics['mse']:.4f}")
        print(f"📊 Root Mean Squared Error: {metrics['rmse']:.4f}")
        print(f"📊 Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"📊 Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print("=" * 60)
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()