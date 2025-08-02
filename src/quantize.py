#!/usr/bin/env python3
"""
Advanced Model Compression Pipeline for Real Estate Valuation System

This module implements sophisticated 8-bit compression techniques for
model optimization while maintaining prediction accuracy within acceptable
tolerance levels.

Author: Data Science Team
Date: 2024
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Import custom utilities
from utils import (
    ModelHandler, PerformanceAnalyzer, CompressionEngine,
    DatasetHandler, load_dataset
)

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_compression.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    Advanced model compression engine with comprehensive validation.
    
    This class implements sophisticated compression techniques for linear
    regression models, including parameter extraction, compression, and
    accuracy validation.
    """
    
    def __init__(self, artifact_directory: str = "models"):
        """
        Initialize the model compressor.
        
        Args:
            artifact_directory: Directory containing trained models
        """
        self.artifact_directory = Path(artifact_directory)
        self.artifact_directory.mkdir(exist_ok=True)
        self.original_model = None
        self.compressed_parameters = {}
        self.compression_metrics = {}
        
    def load_trained_model(self, model_path: str = "models/real_estate_valuation_model.joblib") -> LinearRegression:
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
            model, metadata = ModelHandler.load_model_artifacts(model_path)
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
        Extract and validate model parameters for compression.
        
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
    
    def compress_model_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compress model parameters using advanced techniques.
        
        Args:
            parameters: Dictionary containing model parameters
            
        Returns:
            Dictionary containing compressed parameters and metadata
        """
        try:
            logger.info("⚡ Starting model compression process...")
            
            coef = parameters['coefficients']
            intercept = parameters['intercept']
            
            # Compress coefficients with adaptive scaling
            logger.info("📊 Compressing coefficients...")
            comp_coef, coef_metadata = CompressionEngine.quantize_parameters(
                coef, method='adaptive'
            )
            
            # Compress intercept with individual scaling
            logger.info("📊 Compressing intercept...")
            comp_intercept, intercept_metadata = CompressionEngine.quantize_parameters(
                np.array([intercept]), method='adaptive'
            )
            
            # Prepare compressed parameters
            compressed_params = {
                'quantized_coefficients': comp_coef,
                'coefficient_metadata': coef_metadata,
                'quantized_intercept': comp_intercept[0],  # Extract single value
                'intercept_metadata': intercept_metadata,
                'original_shape': coef.shape,
                'compression_info': {
                    'method': 'adaptive_8bit',
                    'compression_ratio': self._calculate_compression_ratio(coef, comp_coef),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            logger.info("✅ Compression completed successfully")
            logger.info(f"   Compression ratio: {compressed_params['compression_info']['compression_ratio']:.1f}%")
            
            return compressed_params
            
        except Exception as e:
            logger.error(f"❌ Compression failed: {str(e)}")
            raise
    
    def _calculate_compression_ratio(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate compression ratio between original and compressed parameters.
        
        Args:
            original: Original float parameters
            compressed: Compressed uint8 parameters
            
        Returns:
            Compression ratio as percentage
        """
        original_size = original.nbytes
        compressed_size = compressed.nbytes
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        return compression_ratio
    
    def validate_compression_accuracy(self, compressed_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate compression accuracy through comprehensive testing.
        
        Args:
            compressed_params: Compressed parameters
            
        Returns:
            Dictionary containing accuracy metrics
        """
        try:
            logger.info("🔍 Validating compression accuracy...")
            
            # Load test data
            X_train, X_test, y_train, y_test = load_dataset()
            
            # Decompress parameters
            decomp_coef = CompressionEngine.dequantize_parameters(
                compressed_params['quantized_coefficients'],
                compressed_params['coefficient_metadata']
            )
            
            decomp_intercept = CompressionEngine.dequantize_parameters(
                np.array([compressed_params['quantized_intercept']]),
                compressed_params['intercept_metadata']
            )[0]
            
            # Calculate parameter errors
            original_coef = self.original_model.coef_
            original_intercept = self.original_model.intercept_
            
            coef_error = np.abs(original_coef - decomp_coef)
            intercept_error = np.abs(original_intercept - decomp_intercept)
            
            # Generate predictions
            original_pred = self.original_model.predict(X_test[:100])  # Use subset for speed
            
            # Manual prediction with decompressed parameters
            decomp_pred = X_test[:100] @ decomp_coef + decomp_intercept
            
            # Calculate prediction errors
            prediction_errors = np.abs(original_pred - decomp_pred)
            
            # Calculate metrics
            accuracy_metrics = {
                'max_coefficient_error': coef_error.max(),
                'mean_coefficient_error': coef_error.mean(),
                'intercept_error': intercept_error,
                'max_prediction_error': prediction_errors.max(),
                'mean_prediction_error': prediction_errors.mean(),
                'prediction_r2': r2_score(original_pred, decomp_pred),
                'prediction_mse': mean_squared_error(original_pred, decomp_pred)
            }
            
            # Log accuracy metrics
            logger.info("📊 Compression Accuracy Metrics:")
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
    
    def save_compressed_model(self, compressed_params: Dict[str, Any]) -> str:
        """
        Save compressed model with comprehensive metadata.
        
        Args:
            compressed_params: Compressed parameters and metadata
            
        Returns:
            Path to saved compressed model file
        """
        try:
            logger.info("💾 Saving compressed model...")
            
            # Prepare complete model artifacts
            model_artifacts = {
                'quantized_parameters': compressed_params,
                'original_model_info': {
                    'algorithm': 'LinearRegression',
                    'feature_count': compressed_params['original_shape'][0],
                    'original_coefficients_shape': compressed_params['original_shape']
                },
                'compression_metadata': {
                    'version': '1.0.0',
                    'compression_date': pd.Timestamp.now().isoformat(),
                    'compression_ratio': compressed_params['compression_info']['compression_ratio']
                }
            }
            
            # Save to file
            compressed_model_path = self.artifact_directory / "quantized_real_estate_model.joblib"
            joblib.dump(model_artifacts, compressed_model_path)
            
            logger.info(f"✅ Compressed model saved to: {compressed_model_path}")
            return str(compressed_model_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save compressed model: {str(e)}")
            raise
    
    def execute_compression_workflow(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Execute the complete compression pipeline.
        
        Returns:
            Tuple of (compressed_parameters, accuracy_metrics)
        """
        logger.info("🎯 Starting Model Compression Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load trained model
            self.load_trained_model()
            
            # Step 2: Extract model parameters
            parameters = self.extract_model_parameters()
            
            # Step 3: Compress parameters
            compressed_params = self.compress_model_parameters(parameters)
            
            # Step 4: Validate accuracy
            accuracy_metrics = self.validate_compression_accuracy(compressed_params)
            
            # Step 5: Save compressed model
            model_path = self.save_compressed_model(compressed_params)
            
            # Store results
            self.compressed_parameters = compressed_params
            self.compression_metrics = accuracy_metrics
            
            logger.info("=" * 60)
            logger.info("🎉 Compression pipeline completed successfully!")
            logger.info(f"📁 Compressed model saved to: {model_path}")
            
            return compressed_params, accuracy_metrics
            
        except Exception as e:
            logger.error(f"❌ Compression pipeline failed: {str(e)}")
            raise


def main():
    """
    Main entry point for the compression pipeline.
    """
    try:
        # Initialize compressor
        compressor = ModelCompressor()
        
        # Execute compression workflow
        compressed_params, accuracy_metrics = compressor.execute_compression_workflow()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🔧 MODEL COMPRESSION SUMMARY")
        print("=" * 60)
        print(f"📊 Compression Ratio: {compressed_params['compression_info']['compression_ratio']:.1f}%")
        print(f"📊 Max Coefficient Error: {accuracy_metrics['max_coefficient_error']:.8f}")
        print(f"📊 Intercept Error: {accuracy_metrics['intercept_error']:.8f}")
        print(f"📊 Max Prediction Error: {accuracy_metrics['max_prediction_error']:.8f}")
        print(f"📊 Prediction R²: {accuracy_metrics['prediction_r2']:.8f}")
        
        # Quality assessment
        if accuracy_metrics['max_prediction_error'] < 0.001:
            print("✅ Compression Quality: EXCELLENT")
        elif accuracy_metrics['max_prediction_error'] < 0.01:
            print("✅ Compression Quality: GOOD")
        elif accuracy_metrics['max_prediction_error'] < 0.1:
            print("⚠️  Compression Quality: ACCEPTABLE")
        else:
            print("❌ Compression Quality: POOR")
        
        print("=" * 60)
        
        return compressed_params, accuracy_metrics
        
    except Exception as e:
        logger.error(f"❌ Compression failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()