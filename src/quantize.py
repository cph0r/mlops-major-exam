#!/usr/bin/env python3
"""
Advanced Model Quantization Pipeline for California Housing Predictor

This module implements sophisticated 8-bit quantization techniques for
model compression while maintaining prediction accuracy within acceptable
tolerance levels.

Author: ML Engineering Team
Date: 2024
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# Import custom utilities
from utils import (
    ModelManager, MetricsCalculator, QuantizationEngine,
    DataManager, load_dataset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quantization.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Advanced model quantization engine with comprehensive validation.
    
    This class implements sophisticated quantization techniques for linear
    regression models, including parameter extraction, compression, and
    accuracy validation.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the model quantizer.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.original_model = None
        self.quantized_params = {}
        self.quantization_metrics = {}
        
    def load_trained_model(self, model_path: str = "models/housing_price_model.joblib") -> LinearRegression:
        """
        Load the trained model with comprehensive validation.
        
        Args:
            model_path: Path to the trained model file
            
        Returns:
            Loaded LinearRegression model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        try:
            logger.info("🔄 Loading trained model...")
            
            # Load model artifacts
            model, metadata = ModelManager.load_model_artifacts(model_path)
            self.original_model = model
            
            # Extract and validate model parameters
            coef = model.coef_
            intercept = model.intercept_
            
            logger.info(f"✅ Model loaded successfully:")
            logger.info(f"   Coefficients shape: {coef.shape}")
            logger.info(f"   Intercept value: {intercept:.8f}")
            logger.info(f"   Model metadata: {metadata}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def extract_model_parameters(self) -> Dict[str, np.ndarray]:
        """
        Extract and validate model parameters for quantization.
        
        Returns:
            Dictionary containing model parameters
            
        Raises:
            ValueError: If model parameters are invalid
        """
        try:
            if self.original_model is None:
                raise ValueError("No model loaded")
            
            # Extract parameters
            coef = self.original_model.coef_
            intercept = self.original_model.intercept_
            
            # Validate parameters
            if coef is None or intercept is None:
                raise ValueError("Model parameters are None")
            
            if not isinstance(coef, np.ndarray):
                raise ValueError("Coefficients must be numpy array")
            
            if not isinstance(intercept, (float, np.floating)):
                raise ValueError("Intercept must be a float")
            
            # Check for NaN or infinite values
            if np.isnan(coef).any() or np.isinf(coef).any():
                raise ValueError("Coefficients contain NaN or infinite values")
            
            if np.isnan(intercept) or np.isinf(intercept):
                raise ValueError("Intercept contains NaN or infinite values")
            
            parameters = {
                'coefficients': coef,
                'intercept': intercept,
                'feature_count': coef.shape[0]
            }
            
            logger.info("✅ Model parameters extracted successfully:")
            logger.info(f"   Feature count: {parameters['feature_count']}")
            logger.info(f"   Coefficient range: [{coef.min():.6f}, {coef.max():.6f}]")
            logger.info(f"   Intercept: {intercept:.8f}")
            
            return parameters
            
        except Exception as e:
            logger.error(f"❌ Parameter extraction failed: {str(e)}")
            raise
    
    def quantize_model_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Quantize model parameters using advanced techniques.
        
        Args:
            parameters: Dictionary containing model parameters
            
        Returns:
            Dictionary containing quantized parameters and metadata
        """
        try:
            logger.info("⚡ Starting model quantization process...")
            
            coef = parameters['coefficients']
            intercept = parameters['intercept']
            
            # Quantize coefficients with adaptive scaling
            logger.info("📊 Quantizing coefficients...")
            quant_coef, coef_metadata = QuantizationEngine.quantize_parameters(
                coef, method='adaptive'
            )
            
            # Quantize intercept with individual scaling
            logger.info("📊 Quantizing intercept...")
            quant_intercept, intercept_metadata = QuantizationEngine.quantize_parameters(
                np.array([intercept]), method='adaptive'
            )
            
            # Prepare quantized parameters
            quantized_params = {
                'quantized_coefficients': quant_coef,
                'coefficient_metadata': coef_metadata,
                'quantized_intercept': quant_intercept[0],  # Extract single value
                'intercept_metadata': intercept_metadata,
                'original_shape': coef.shape,
                'quantization_info': {
                    'method': 'adaptive_8bit',
                    'compression_ratio': self._calculate_compression_ratio(coef, quant_coef),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            logger.info("✅ Quantization completed successfully")
            logger.info(f"   Compression ratio: {quantized_params['quantization_info']['compression_ratio']:.1f}%")
            
            return quantized_params
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {str(e)}")
            raise
    
    def _calculate_compression_ratio(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Calculate compression ratio between original and quantized parameters.
        
        Args:
            original: Original float parameters
            quantized: Quantized uint8 parameters
            
        Returns:
            Compression ratio as percentage
        """
        original_size = original.nbytes
        quantized_size = quantized.nbytes
        compression_ratio = ((original_size - quantized_size) / original_size) * 100
        return compression_ratio
    
    def validate_quantization_accuracy(self, quantized_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate quantization accuracy through comprehensive testing.
        
        Args:
            quantized_params: Quantized parameters
            
        Returns:
            Dictionary containing accuracy metrics
        """
        try:
            logger.info("🔍 Validating quantization accuracy...")
            
            # Load test data
            X_train, X_test, y_train, y_test = load_dataset()
            
            # Dequantize parameters
            dequant_coef = QuantizationEngine.dequantize_parameters(
                quantized_params['quantized_coefficients'],
                quantized_params['coefficient_metadata']
            )
            
            dequant_intercept = QuantizationEngine.dequantize_parameters(
                np.array([quantized_params['quantized_intercept']]),
                quantized_params['intercept_metadata']
            )[0]
            
            # Calculate parameter errors
            original_coef = self.original_model.coef_
            original_intercept = self.original_model.intercept_
            
            coef_error = np.abs(original_coef - dequant_coef)
            intercept_error = np.abs(original_intercept - dequant_intercept)
            
            # Generate predictions
            original_pred = self.original_model.predict(X_test[:100])  # Use subset for speed
            
            # Manual prediction with dequantized parameters
            dequant_pred = X_test[:100] @ dequant_coef + dequant_intercept
            
            # Calculate prediction errors
            prediction_errors = np.abs(original_pred - dequant_pred)
            
            # Calculate metrics
            accuracy_metrics = {
                'max_coefficient_error': coef_error.max(),
                'mean_coefficient_error': coef_error.mean(),
                'intercept_error': intercept_error,
                'max_prediction_error': prediction_errors.max(),
                'mean_prediction_error': prediction_errors.mean(),
                'prediction_r2': r2_score(original_pred, dequant_pred),
                'prediction_mse': mean_squared_error(original_pred, dequant_pred)
            }
            
            # Log accuracy metrics
            logger.info("📊 Quantization Accuracy Metrics:")
            logger.info(f"   Max coefficient error: {accuracy_metrics['max_coefficient_error']:.8f}")
            logger.info(f"   Mean coefficient error: {accuracy_metrics['mean_coefficient_error']:.8f}")
            logger.info(f"   Intercept error: {accuracy_metrics['intercept_error']:.8f}")
            logger.info(f"   Max prediction error: {accuracy_metrics['max_prediction_error']:.8f}")
            logger.info(f"   Mean prediction error: {accuracy_metrics['mean_prediction_error']:.8f}")
            logger.info(f"   Prediction R²: {accuracy_metrics['prediction_r2']:.8f}")
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"❌ Accuracy validation failed: {str(e)}")
            raise
    
    def save_quantized_model(self, quantized_params: Dict[str, Any]) -> str:
        """
        Save quantized model with comprehensive metadata.
        
        Args:
            quantized_params: Quantized parameters and metadata
            
        Returns:
            Path to saved quantized model file
        """
        try:
            logger.info("💾 Saving quantized model...")
            
            # Prepare complete model artifacts
            model_artifacts = {
                'quantized_parameters': quantized_params,
                'original_model_info': {
                    'algorithm': 'LinearRegression',
                    'feature_count': quantized_params['original_shape'][0],
                    'original_coefficients_shape': quantized_params['original_shape']
                },
                'quantization_metadata': {
                    'version': '1.0.0',
                    'quantization_date': pd.Timestamp.now().isoformat(),
                    'compression_ratio': quantized_params['quantization_info']['compression_ratio']
                }
            }
            
            # Save to file
            quantized_model_path = self.model_dir / "quantized_housing_model.joblib"
            joblib.dump(model_artifacts, quantized_model_path)
            
            logger.info(f"✅ Quantized model saved to: {quantized_model_path}")
            return str(quantized_model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save quantized model: {str(e)}")
            raise
    
    def run_quantization_pipeline(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute the complete quantization pipeline.
        
        Returns:
            Tuple of (quantized_parameters, accuracy_metrics)
        """
        logger.info("🎯 Starting Model Quantization Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load trained model
            self.load_trained_model()
            
            # Step 2: Extract model parameters
            parameters = self.extract_model_parameters()
            
            # Step 3: Quantize parameters
            quantized_params = self.quantize_model_parameters(parameters)
            
            # Step 4: Validate accuracy
            accuracy_metrics = self.validate_quantization_accuracy(quantized_params)
            
            # Step 5: Save quantized model
            model_path = self.save_quantized_model(quantized_params)
            
            # Store results
            self.quantized_params = quantized_params
            self.quantization_metrics = accuracy_metrics
            
            logger.info("=" * 60)
            logger.info("🎉 Quantization pipeline completed successfully!")
            logger.info(f"📁 Quantized model saved to: {model_path}")
            
            return quantized_params, accuracy_metrics
            
        except Exception as e:
            logger.error(f"❌ Quantization pipeline failed: {str(e)}")
            raise


def main():
    """
    Main entry point for the quantization pipeline.
    """
    try:
        # Initialize quantizer
        quantizer = ModelQuantizer()
        
        # Run quantization pipeline
        quantized_params, accuracy_metrics = quantizer.run_quantization_pipeline()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🔧 MODEL QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"📊 Compression Ratio: {quantized_params['quantization_info']['compression_ratio']:.1f}%")
        print(f"📊 Max Coefficient Error: {accuracy_metrics['max_coefficient_error']:.8f}")
        print(f"📊 Intercept Error: {accuracy_metrics['intercept_error']:.8f}")
        print(f"📊 Max Prediction Error: {accuracy_metrics['max_prediction_error']:.8f}")
        print(f"📊 Prediction R²: {accuracy_metrics['prediction_r2']:.8f}")
        
        # Quality assessment
        if accuracy_metrics['max_prediction_error'] < 0.001:
            print("✅ Quantization Quality: EXCELLENT")
        elif accuracy_metrics['max_prediction_error'] < 0.01:
            print("✅ Quantization Quality: GOOD")
        elif accuracy_metrics['max_prediction_error'] < 0.1:
            print("⚠️  Quantization Quality: ACCEPTABLE")
        else:
            print("❌ Quantization Quality: POOR")
        
        print("=" * 60)
        
        return quantized_params, accuracy_metrics
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()