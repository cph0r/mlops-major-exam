#!/usr/bin/env python3
"""
Advanced Prediction Pipeline for California Housing Price Predictor

This module provides comprehensive prediction capabilities for the housing
price prediction model, including both original and quantized model inference
with detailed performance analysis and visualization.

Author: ML Engineering Team
Date: 2024
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import custom utilities
from utils import (
    ModelManager, MetricsCalculator, DataManager,
    QuantizationEngine, load_dataset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prediction.log')
    ]
)
logger = logging.getLogger(__name__)


class HousingPricePredictor:
    """
    Advanced housing price prediction engine with comprehensive analysis.
    
    This class provides sophisticated prediction capabilities including
    model loading, inference execution, performance analysis, and
    detailed reporting for both original and quantized models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the housing price predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.original_model = None
        self.quantized_params = None
        self.prediction_results = {}
        self.performance_metrics = {}
        
    def load_models(self, 
                   original_model_path: str = "models/housing_price_model.joblib",
                   quantized_model_path: str = "models/quantized_housing_model.joblib") -> None:
        """
        Load both original and quantized models with validation.
        
        Args:
            original_model_path: Path to original trained model
            quantized_model_path: Path to quantized model
            
        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If model loading fails
        """
        try:
            logger.info("🔄 Loading prediction models...")
            
            # Load original model
            if Path(original_model_path).exists():
                self.original_model, _ = ModelManager.load_model_artifacts(original_model_path)
                logger.info("✅ Original model loaded successfully")
            else:
                logger.warning("⚠️  Original model not found, skipping...")
            
            # Load quantized model
            if Path(quantized_model_path).exists():
                quantized_artifacts = joblib.load(quantized_model_path)
                self.quantized_params = quantized_artifacts['quantized_parameters']
                logger.info("✅ Quantized model loaded successfully")
            else:
                logger.warning("⚠️  Quantized model not found, skipping...")
            
            if self.original_model is None and self.quantized_params is None:
                raise ValueError("No models available for prediction")
                
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            raise
    
    def prepare_test_data(self, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare test data for prediction with validation.
        
        Args:
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Tuple of (X_test, y_test) arrays
        """
        try:
            logger.info("📊 Preparing test dataset...")
            
            # Load dataset
            X_train, X_test, y_train, y_test = DataManager.load_california_housing_dataset()
            
            # Validate data quality
            DataManager.validate_data_quality(X_train, X_test, y_train, y_test)
            
            # Sample data if requested
            if sample_size is not None and sample_size < len(X_test):
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
                logger.info(f"📊 Using {sample_size} test samples")
            else:
                logger.info(f"📊 Using all {len(X_test)} test samples")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare test data: {str(e)}")
            raise
    
    def predict_with_original_model(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the original model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted values
            
        Raises:
            ValueError: If original model is not available
        """
        try:
            if self.original_model is None:
                raise ValueError("Original model not available")
            
            logger.info("🔮 Generating predictions with original model...")
            predictions = self.original_model.predict(X_test)
            
            logger.info(f"✅ Original model predictions generated: {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Original model prediction failed: {str(e)}")
            raise
    
    def predict_with_quantized_model(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the quantized model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted values
            
        Raises:
            ValueError: If quantized model is not available
        """
        try:
            if self.quantized_params is None:
                raise ValueError("Quantized model not available")
            
            logger.info("🔮 Generating predictions with quantized model...")
            
            # Dequantize parameters
            quant_coef = self.quantized_params['quantized_coefficients']
            coef_metadata = self.quantized_params['coefficient_metadata']
            quant_intercept = self.quantized_params['quantized_intercept']
            intercept_metadata = self.quantized_params['intercept_metadata']
            
            # Dequantize coefficients and intercept
            dequant_coef = QuantizationEngine.dequantize_parameters(quant_coef, coef_metadata)
            dequant_intercept = QuantizationEngine.dequantize_parameters(
                np.array([quant_intercept]), intercept_metadata
            )[0]
            
            # Generate predictions manually
            predictions = X_test @ dequant_coef + dequant_intercept
            
            logger.info(f"✅ Quantized model predictions generated: {len(predictions)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Quantized model prediction failed: {str(e)}")
            raise
    
    def calculate_performance_metrics(self, 
                                    y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    model_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            logger.info(f"📊 Calculating performance metrics for {model_name}...")
            
            metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
            
            # Additional analysis
            residuals = y_true - y_pred
            metrics.update({
                'residual_std': np.std(residuals),
                'residual_mean': np.mean(residuals),
                'prediction_range': y_pred.max() - y_pred.min(),
                'true_range': y_true.max() - y_true.min(),
                'correlation': np.corrcoef(y_true, y_pred)[0, 1]
            })
            
            logger.info(f"✅ {model_name} metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics for {model_name}: {str(e)}")
            raise
    
    def generate_prediction_report(self, 
                                 y_true: np.ndarray,
                                 original_pred: Optional[np.ndarray] = None,
                                 quantized_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive prediction report with analysis.
        
        Args:
            y_true: True target values
            original_pred: Original model predictions
            quantized_pred: Quantized model predictions
            
        Returns:
            Dictionary containing comprehensive report
        """
        try:
            logger.info("📋 Generating comprehensive prediction report...")
            
            report = {
                'dataset_info': {
                    'total_samples': len(y_true),
                    'true_value_range': [y_true.min(), y_true.max()],
                    'true_value_mean': y_true.mean(),
                    'true_value_std': y_true.std()
                },
                'model_comparison': {},
                'sample_predictions': []
            }
            
            # Analyze original model if available
            if original_pred is not None:
                original_metrics = self.calculate_performance_metrics(y_true, original_pred, "Original Model")
                report['model_comparison']['original_model'] = original_metrics
                
                # Sample predictions for original model
                for i in range(min(10, len(y_true))):
                    report['sample_predictions'].append({
                        'sample_id': i,
                        'true_value': y_true[i],
                        'original_prediction': original_pred[i],
                        'original_error': abs(y_true[i] - original_pred[i])
                    })
            
            # Analyze quantized model if available
            if quantized_pred is not None:
                quantized_metrics = self.calculate_performance_metrics(y_true, quantized_pred, "Quantized Model")
                report['model_comparison']['quantized_model'] = quantized_metrics
                
                # Update sample predictions with quantized results
                for i, sample in enumerate(report['sample_predictions']):
                    if i < len(quantized_pred):
                        sample['quantized_prediction'] = quantized_pred[i]
                        sample['quantized_error'] = abs(y_true[i] - quantized_pred[i])
                        sample['prediction_difference'] = abs(original_pred[i] - quantized_pred[i]) if original_pred is not None else None
            
            # Model comparison analysis
            if original_pred is not None and quantized_pred is not None:
                comparison_metrics = self.calculate_performance_metrics(original_pred, quantized_pred, "Model Comparison")
                report['model_comparison']['comparison'] = comparison_metrics
            
            logger.info("✅ Prediction report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate prediction report: {str(e)}")
            raise
    
    def print_prediction_summary(self, report: Dict[str, Any]) -> None:
        """
        Print formatted prediction summary.
        
        Args:
            report: Comprehensive prediction report
        """
        print("\n" + "=" * 70)
        print("🏠 CALIFORNIA HOUSING PRICE PREDICTION SUMMARY")
        print("=" * 70)
        
        # Dataset information
        dataset_info = report['dataset_info']
        print(f"📊 Dataset Information:")
        print(f"   Total Samples: {dataset_info['total_samples']}")
        print(f"   True Value Range: [{dataset_info['true_value_range'][0]:.2f}, {dataset_info['true_value_range'][1]:.2f}]")
        print(f"   True Value Mean: {dataset_info['true_value_mean']:.2f}")
        print(f"   True Value Std: {dataset_info['true_value_std']:.2f}")
        
        # Model performance comparison
        model_comparison = report['model_comparison']
        
        if 'original_model' in model_comparison:
            orig_metrics = model_comparison['original_model']
            print(f"\n🔮 Original Model Performance:")
            print(f"   R² Score: {orig_metrics['r2_score']:.4f}")
            print(f"   Mean Squared Error: {orig_metrics['mse']:.4f}")
            print(f"   Root Mean Squared Error: {orig_metrics['rmse']:.4f}")
            print(f"   Mean Absolute Error: {orig_metrics['mae']:.4f}")
            print(f"   Mean Absolute Percentage Error: {orig_metrics['mape']:.2f}%")
        
        if 'quantized_model' in model_comparison:
            quant_metrics = model_comparison['quantized_model']
            print(f"\n⚡ Quantized Model Performance:")
            print(f"   R² Score: {quant_metrics['r2_score']:.4f}")
            print(f"   Mean Squared Error: {quant_metrics['mse']:.4f}")
            print(f"   Root Mean Squared Error: {quant_metrics['rmse']:.4f}")
            print(f"   Mean Absolute Error: {quant_metrics['mae']:.4f}")
            print(f"   Mean Absolute Percentage Error: {quant_metrics['mape']:.2f}%")
        
        # Sample predictions
        print(f"\n📋 Sample Predictions (First 10):")
        print(f"{'ID':<3} {'True':<8} {'Original':<10} {'Quantized':<10} {'Orig Err':<9} {'Quant Err':<9}")
        print("-" * 60)
        
        for sample in report['sample_predictions'][:10]:
            print(f"{sample['sample_id']:<3} {sample['true_value']:<8.2f} "
                  f"{sample.get('original_prediction', 'N/A'):<10.2f} "
                  f"{sample.get('quantized_prediction', 'N/A'):<10.2f} "
                  f"{sample.get('original_error', 'N/A'):<9.2f} "
                  f"{sample.get('quantized_error', 'N/A'):<9.2f}")
        
        print("=" * 70)
    
    def run_prediction_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the complete prediction pipeline.
        
        Args:
            sample_size: Number of test samples to use
            
        Returns:
            Comprehensive prediction report
        """
        logger.info("🎯 Starting Housing Price Prediction Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load models
            self.load_models()
            
            # Step 2: Prepare test data
            X_test, y_test = self.prepare_test_data(sample_size)
            
            # Step 3: Generate predictions
            original_pred = None
            quantized_pred = None
            
            if self.original_model is not None:
                original_pred = self.predict_with_original_model(X_test)
            
            if self.quantized_params is not None:
                quantized_pred = self.predict_with_quantized_model(X_test)
            
            # Step 4: Generate comprehensive report
            report = self.generate_prediction_report(y_test, original_pred, quantized_pred)
            
            # Step 5: Print summary
            self.print_prediction_summary(report)
            
            logger.info("=" * 60)
            logger.info("🎉 Prediction pipeline completed successfully!")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Prediction pipeline failed: {str(e)}")
            raise


def main():
    """
    Main entry point for the prediction pipeline.
    """
    try:
        # Initialize predictor
        predictor = HousingPricePredictor()
        
        # Run prediction pipeline
        report = predictor.run_prediction_pipeline()
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()