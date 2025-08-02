#!/usr/bin/env python3
"""
Utility functions for Real Estate Valuation System

This module provides comprehensive utility functions for data processing,
model management, and quantization operations used throughout the ML pipeline.

"""

import os
import logging
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Configure logging system
logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    Comprehensive dataset management utility for real estate data operations.
    """
    
    @staticmethod
    def load_california_housing_dataset(
        test_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split California Housing dataset with advanced preprocessing.
        
        Args:
            test_size: Proportion of dataset to include in test split
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) arrays
            
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info("Loading California Housing dataset...")
            
            # Fetch dataset
            housing_dataset = fetch_california_housing()
            feature_matrix, target_vector = housing_dataset.data, housing_dataset.target
            
            # Validate data integrity
            if feature_matrix is None or target_vector is None:
                raise ValueError("Dataset contains null values")
            
            if feature_matrix.shape[0] != target_vector.shape[0]:
                raise ValueError("Feature and target arrays have different lengths")
            
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix, target_vector, test_size=test_size, random_state=random_state, shuffle=shuffle
            )
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"   Total samples: {feature_matrix.shape[0]}")
            logger.info(f"   Training samples: {X_train.shape[0]}")
            logger.info(f"   Test samples: {X_test.shape[0]}")
            logger.info(f"   Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {str(e)}")
            raise RuntimeError(f"Dataset loading failed: {str(e)}")
    
    @staticmethod
    def validate_data_quality(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> bool:
        """
        Validate data quality and integrity.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            True if data quality checks pass
            
        Raises:
            ValueError: If data quality issues are detected
        """
        try:
            # Check for null values
            if np.isnan(X_train).any() or np.isnan(X_test).any():
                raise ValueError("Feature data contains NaN values")
            
            if np.isnan(y_train).any() or np.isnan(y_test).any():
                raise ValueError("Target data contains NaN values")
            
            # Check for infinite values
            if np.isinf(X_train).any() or np.isinf(X_test).any():
                raise ValueError("Feature data contains infinite values")
            
            if np.isinf(y_train).any() or np.isinf(y_test).any():
                raise ValueError("Target data contains infinite values")
            
            # Check data shapes
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Training and test features have different dimensions")
            
            if len(X_train) != len(y_train):
                raise ValueError("Training features and targets have different lengths")
            
            if len(X_test) != len(y_test):
                raise ValueError("Test features and targets have different lengths")
            
            # Check target value range
            if y_train.min() < 0 or y_test.min() < 0:
                logger.warning("⚠️  Negative target values detected")
            
            logger.info("Data quality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data quality validation failed: {str(e)}")
            raise ValueError(f"Data quality issues: {str(e)}")


class ModelHandler:
    """
    Advanced model management utilities for training and deployment.
    """
    
    @staticmethod
    def create_linear_regression_model(
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = -1,
        positive: bool = False
    ) -> LinearRegression:
        """
        Create and configure LinearRegression model with advanced parameters.
        
        Args:
            fit_intercept: Whether to calculate intercept
            copy_X: Whether to copy X or overwrite
            n_jobs: Number of jobs for parallel computation
            positive: Force coefficients to be positive
            
        Returns:
            Configured LinearRegression model
        """
        try:
            logger.info("Creating LinearRegression model...")
            
            regression_model = LinearRegression(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive
            )
            
            logger.info("Model created successfully")
            return regression_model
            
        except Exception as e:
            logger.error(f"❌ Failed to create model: {str(e)}")
            raise
    
    @staticmethod
    def save_model_artifacts(
        model: LinearRegression,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model with comprehensive metadata and error handling.
        
        Args:
            model: Trained model to save
            filepath: Path where to save the model
            metadata: Additional metadata to save with model
            
        Returns:
            Path to saved model file
            
        Raises:
            IOError: If model saving fails
        """
        try:
            filepath = Path(filepath)
            
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model artifacts
            model_artifacts = {
                'model': model,
                'metadata': metadata or {},
                'version': '1.0.0',
                'saved_at': pd.Timestamp.now().isoformat()
            }
            
            # Save model
            joblib.dump(model_artifacts, filepath)
            
            logger.info(f"Model saved successfully to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
            raise IOError(f"Model saving failed: {str(e)}")
    
    @staticmethod
    def load_model_artifacts(filepath: Union[str, Path]) -> Tuple[LinearRegression, Dict[str, Any]]:
        """
        Load model with metadata and validation.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Load model artifacts
            model_artifacts = joblib.load(filepath)
            
            # Validate loaded data
            if 'model' not in model_artifacts:
                raise ValueError("Invalid model file: 'model' key not found")
            
            model = model_artifacts['model']
            metadata = model_artifacts.get('metadata', {})
            
            # Validate model type
            if not isinstance(model, LinearRegression):
                raise ValueError("Loaded object is not a LinearRegression model")
            
            logger.info(f"Model loaded successfully from: {filepath}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis utilities.
    """
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing various regression metrics
        """
        try:
            # Validate inputs
            if y_true.shape != y_pred.shape:
                raise ValueError("True and predicted arrays have different shapes")
            
            if len(y_true) == 0:
                raise ValueError("Empty arrays provided")
            
            # Calculate metrics
            performance_metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': np.mean(np.abs(y_true - y_pred)),
                'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100,
                'explained_variance': np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
            }
            
            # Additional statistical metrics
            residuals = y_true - y_pred
            performance_metrics.update({
                'residual_std': np.std(residuals),
                'residual_mean': np.mean(residuals),
                'max_error': np.max(np.abs(residuals)),
                'median_absolute_error': np.median(np.abs(residuals))
            })
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics: {str(e)}")
            raise
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float]) -> None:
        """
        Print formatted metrics summary.
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        print("\n" + "=" * 50)
        print("📊 MODEL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"R² Score:                    {metrics['r2_score']:.4f}")
        print(f"Mean Squared Error:          {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error:     {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error:         {metrics['mae']:.4f}")
        print(f"Mean Absolute Percentage:    {metrics['mape']:.2f}%")
        print(f"Explained Variance:          {metrics['explained_variance']:.4f}")
        print(f"Residual Standard Deviation: {metrics['residual_std']:.4f}")
        print(f"Maximum Error:               {metrics['max_error']:.4f}")
        print("=" * 50)


class CompressionEngine:
    """
    Advanced compression utilities for model optimization.
    """
    
    @staticmethod
    def quantize_parameters(
        values: np.ndarray,
        scale_factor: Optional[float] = None,
        method: str = 'adaptive'
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Quantize float parameters to uint8 with advanced scaling.
        
        Args:
            values: Float values to quantize
            scale_factor: Optional scale factor (auto-calculated if None)
            method: Quantization method ('adaptive' or 'fixed')
            
        Returns:
            Tuple of (quantized_values, metadata)
        """
        try:
            if np.all(values == 0):
                return np.zeros(values.shape, dtype=np.uint8), {
                    'min_val': 0.0, 'max_val': 0.0, 'scale_factor': 1.0, 'method': method
                }
            
            if scale_factor is None:
                if method == 'adaptive':
                    abs_max = np.abs(values).max()
                    scale_factor = 200.0 / abs_max if abs_max > 0 else 1.0
                else:
                    scale_factor = 1.0
            
            # Apply scaling
            scaled_values = values * scale_factor
            min_val, max_val = scaled_values.min(), scaled_values.max()
            
            # Handle constant values
            if max_val == min_val:
                quantized = np.full(values.shape, 127, dtype=np.uint8)
                return quantized, {
                    'min_val': min_val, 'max_val': max_val, 
                    'scale_factor': scale_factor, 'method': method
                }
            
            # Normalize to 0-255 range
            value_range = max_val - min_val
            normalized = ((scaled_values - min_val) / value_range * 255)
            normalized = np.clip(normalized, 0, 255)
            quantized = normalized.astype(np.uint8)
            
            metadata = {
                'min_val': min_val,
                'max_val': max_val,
                'scale_factor': scale_factor,
                'method': method,
                'original_shape': values.shape
            }
            
            return quantized, metadata
            
        except Exception as e:
            logger.error(f"❌ Quantization failed: {str(e)}")
            raise
    
    @staticmethod
    def dequantize_parameters(
        quantized_values: np.ndarray,
        metadata: Dict[str, float]
    ) -> np.ndarray:
        """
        Dequantize uint8 parameters back to float.
        
        Args:
            quantized_values: Quantized uint8 values
            metadata: Quantization metadata
            
        Returns:
            Dequantized float values
        """
        try:
            min_val = metadata['min_val']
            max_val = metadata['max_val']
            scale_factor = metadata['scale_factor']
            
            # Handle constant values
            if max_val == min_val:
                return np.full(quantized_values.shape, min_val / scale_factor, dtype=np.float32)
            
            # Denormalize from 0-255 range
            value_range = max_val - min_val
            denormalized = (quantized_values.astype(np.float32) / 255.0) * value_range + min_val
            
            # Descale
            original_values = denormalized / scale_factor
            
            return original_values
            
        except Exception as e:
            logger.error(f"❌ Dequantization failed: {str(e)}")
            raise


# Legacy function aliases for backward compatibility
def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy function for loading dataset."""
    return DatasetHandler.load_california_housing_dataset()


def create_model() -> LinearRegression:
    """Legacy function for creating model."""
    return ModelHandler.create_linear_regression_model()


def save_model(model: LinearRegression, filepath: str) -> str:
    """Legacy function for saving model."""
    return ModelHandler.save_model_artifacts(model, filepath)


def load_model(filepath: str) -> LinearRegression:
    """Legacy function for loading model."""
    model, _ = ModelHandler.load_model_artifacts(filepath)
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Legacy function for calculating metrics."""
    metrics = PerformanceAnalyzer.calculate_regression_metrics(y_true, y_pred)
    return metrics['r2_score'], metrics['mse']


def quantize_to_uint8(values: np.ndarray, scale_factor: Optional[float] = None) -> Tuple[np.ndarray, float, float, float]:
    """Legacy function for quantization."""
    quantized, metadata = CompressionEngine.quantize_parameters(values, scale_factor)
    return quantized, metadata['min_val'], metadata['max_val'], metadata['scale_factor']


def dequantize_from_uint8(quantized_values: np.ndarray, min_val: float, max_val: float, scale_factor: float) -> np.ndarray:
    """Legacy function for dequantization."""
    metadata = {'min_val': min_val, 'max_val': max_val, 'scale_factor': scale_factor}
    return CompressionEngine.dequantize_parameters(quantized_values, metadata)