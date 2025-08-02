#!/usr/bin/env python3
"""
Real Estate Valuation System - Model Training Engine

This module implements the core training pipeline for the real estate
valuation system using advanced regression techniques with comprehensive
logging and performance monitoring.

Author: Data Science Team
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

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)


class RealEstateValuationEngine:
    """
    Advanced real estate valuation model with comprehensive training pipeline.
    
    This class encapsulates the entire training workflow including data loading,
    model training, performance evaluation, and model persistence.
    """
    
    def __init__(self, artifact_directory: str = "models"):
        """
        Initialize the real estate valuation engine.
        
        Args:
            artifact_directory: Directory to store trained models
        """
        self.artifact_directory = Path(artifact_directory)
        self.artifact_directory.mkdir(exist_ok=True)
        self.trained_model = None
        self.performance_statistics = {}
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the California housing dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) arrays
        """
        logger.info("🔄 Loading California Housing dataset...")
        
        try:
            from sklearn.datasets import fetch_california_housing
            from sklearn.model_selection import train_test_split
            
            # Fetch dataset
            housing_dataset = fetch_california_housing()
            feature_matrix, target_vector = housing_dataset.data, housing_dataset.target
            
            # Perform train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix, target_vector, test_size=0.2, random_state=42, shuffle=True
            )
            
            logger.info(f"✅ Dataset loaded successfully:")
            logger.info(f"   Training samples: {X_train.shape[0]}")
            logger.info(f"   Test samples: {X_test.shape[0]}")
            logger.info(f"   Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {str(e)}")
            raise
    
    def initialize_model(self) -> LinearRegression:
        """
        Create and configure the linear regression model.
        
        Returns:
            Configured LinearRegression model
        """
        logger.info("🔧 Creating optimized LinearRegression model...")
        
        try:
            # Initialize model with optimized parameters
            regression_model = LinearRegression(
                fit_intercept=True,
                copy_X=True,
                n_jobs=-1,  # Use all available cores
                positive=False
            )
            
            logger.info("✅ Model created successfully")
            return regression_model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {str(e)}")
            raise
    
    def execute_training(self, model: LinearRegression, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """
        Train the linear regression model with comprehensive validation.
        
        Args:
            model: LinearRegression model instance
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model
        """
        logger.info("Training model with advanced techniques...")
        
        try:
            # Perform cross-validation for model validation
            cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            logger.info(f"Cross-validation R² scores: {cv_results}")
            logger.info(f"Mean CV R²: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
            
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
    
    def assess_model_performance(self, model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
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
            predicted_values = model.predict(X_test)
            
            # Calculate comprehensive metrics
            performance_metrics = {
                'r2_score': r2_score(y_test, predicted_values),
                'mse': mean_squared_error(y_test, predicted_values),
                'rmse': np.sqrt(mean_squared_error(y_test, predicted_values)),
                'mae': mean_absolute_error(y_test, predicted_values),
                'mape': np.mean(np.abs((y_test - predicted_values) / y_test)) * 100
            }
            
            # Log performance metrics
            logger.info("📈 Model Performance Metrics:")
            logger.info(f"   R² Score: {performance_metrics['r2_score']:.4f}")
            logger.info(f"   Mean Squared Error: {performance_metrics['mse']:.4f}")
            logger.info(f"   Root Mean Squared Error: {performance_metrics['rmse']:.4f}")
            logger.info(f"   Mean Absolute Error: {performance_metrics['mae']:.4f}")
            logger.info(f"   Mean Absolute Percentage Error: {performance_metrics['mape']:.2f}%")
            
            # Performance validation
            if performance_metrics['r2_score'] < 0.5:
                logger.warning("⚠️  Model performance below threshold (R² < 0.5)")
            else:
                logger.info("✅ Model performance meets requirements")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def persist_model(self, model: LinearRegression, metrics: dict) -> str:
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
            model_filepath = self.artifact_directory / "real_estate_valuation_model.joblib"
            joblib.dump(model_artifacts, model_filepath)
            
            logger.info(f"✅ Model saved successfully to: {model_filepath}")
            return str(model_filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
            raise
    
    def execute_training_workflow(self) -> Tuple[LinearRegression, dict]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Tuple of (trained_model, performance_metrics)
        """
        logger.info("🎯 Starting Real Estate Valuation Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and preprocess data
            X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
            
            # Step 2: Initialize model
            model = self.initialize_model()
            
            # Step 3: Execute training
            trained_model = self.execute_training(model, X_train, y_train)
            
            # Step 4: Assess model performance
            metrics = self.assess_model_performance(trained_model, X_test, y_test)
            
            # Step 5: Persist model
            model_path = self.persist_model(trained_model, metrics)
            
            # Store metrics for later use
            self.performance_statistics = metrics
            self.trained_model = trained_model
            
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
        # Initialize valuation engine
        valuation_engine = RealEstateValuationEngine()
        
        # Execute training workflow
        model, metrics = valuation_engine.execute_training_workflow()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏠 REAL ESTATE VALUATION SYSTEM - TRAINING SUMMARY")
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