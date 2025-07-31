#!/usr/bin/env python3
"""
Comprehensive Test Suite for California Housing Price Predictor

This module provides extensive testing coverage for the housing price
prediction pipeline, including unit tests, integration tests, and
performance validation tests.

Author: ML Engineering Team
Date: 2024
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules to test
from utils import (
    DataManager, ModelManager, MetricsCalculator, QuantizationEngine,
    load_dataset, create_model, save_model, load_model, calculate_metrics
)


class TestDataManagement:
    """Comprehensive tests for data management functionality."""
    
    def test_dataset_loading(self):
        """Test dataset loading with comprehensive validation."""
        # Load dataset
        X_train, X_test, y_train, y_test = DataManager.load_california_housing_dataset()
        
        # Basic validation
        assert X_train is not None, "Training features should not be None"
        assert X_test is not None, "Test features should not be None"
        assert y_train is not None, "Training targets should not be None"
        assert y_test is not None, "Test targets should not be None"
        
        # Shape validation
        assert X_train.shape[1] == 8, "California housing should have 8 features"
        assert X_test.shape[1] == 8, "Test features should have 8 dimensions"
        assert len(X_train) == len(y_train), "Training features and targets should have same length"
        assert len(X_test) == len(y_test), "Test features and targets should have same length"
        
        # Data type validation
        assert isinstance(X_train, np.ndarray), "Training features should be numpy array"
        assert isinstance(X_test, np.ndarray), "Test features should be numpy array"
        assert isinstance(y_train, np.ndarray), "Training targets should be numpy array"
        assert isinstance(y_test, np.ndarray), "Test targets should be numpy array"
        
        # Train/test split ratio validation (approximately 80/20)
        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        assert 0.75 <= train_ratio <= 0.85, f"Train ratio {train_ratio:.2f} should be between 0.75 and 0.85"
        
        # Value range validation
        assert X_train.min() >= 0, "Feature values should be non-negative"
        assert y_train.min() >= 0, "Target values should be non-negative"
    
    def test_data_quality_validation(self):
        """Test data quality validation functionality."""
        # Load dataset
        X_train, X_test, y_train, y_test = DataManager.load_california_housing_dataset()
        
        # Test valid data
        assert DataManager.validate_data_quality(X_train, X_test, y_train, y_test) is True
        
        # Test with NaN values (should raise ValueError)
        X_train_nan = X_train.copy()
        X_train_nan[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="NaN values"):
            DataManager.validate_data_quality(X_train_nan, X_test, y_train, y_test)
        
        # Test with infinite values (should raise ValueError)
        X_train_inf = X_train.copy()
        X_train_inf[0, 0] = np.inf
        
        with pytest.raises(ValueError, match="infinite values"):
            DataManager.validate_data_quality(X_train_inf, X_test, y_train, y_test)
        
        # Test with mismatched shapes (should raise ValueError)
        with pytest.raises(ValueError, match="different dimensions"):
            DataManager.validate_data_quality(X_train, X_test[:, :7], y_train, y_test)


class TestModelManagement:
    """Comprehensive tests for model management functionality."""
    
    def test_model_creation(self):
        """Test model creation with various configurations."""
        # Test default model creation
        model = ModelManager.create_linear_regression_model()
        assert isinstance(model, LinearRegression), "Should create LinearRegression instance"
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert hasattr(model, 'predict'), "Model should have predict method"
        
        # Test model with custom parameters
        model_custom = ModelManager.create_linear_regression_model(
            fit_intercept=False,
            copy_X=False,
            n_jobs=1,
            positive=True
        )
        assert isinstance(model_custom, LinearRegression), "Should create LinearRegression with custom params"
    
    def test_model_save_load(self):
        """Test model saving and loading functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and train a model
            X_train, X_test, y_train, y_test = load_dataset()
            model = create_model()
            model.fit(X_train, y_train)
            
            # Test model saving
            model_path = os.path.join(temp_dir, "test_model.joblib")
            metadata = {"test_metadata": "test_value", "version": "1.0.0"}
            
            saved_path = ModelManager.save_model_artifacts(model, model_path, metadata)
            assert os.path.exists(saved_path), "Model file should be created"
            
            # Test model loading
            loaded_model, loaded_metadata = ModelManager.load_model_artifacts(saved_path)
            assert isinstance(loaded_model, LinearRegression), "Loaded object should be LinearRegression"
            assert loaded_metadata["test_metadata"] == "test_value", "Metadata should be preserved"
            
            # Test predictions consistency
            original_pred = model.predict(X_test[:5])
            loaded_pred = loaded_model.predict(X_test[:5])
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=10)
    
    def test_model_loading_errors(self):
        """Test model loading error handling."""
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            ModelManager.load_model_artifacts("non_existent_model.joblib")
        
        # Test loading invalid model file
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "invalid_model.joblib")
            
            # Save invalid data
            joblib.dump({"invalid": "data"}, invalid_path)
            
            with pytest.raises(ValueError, match="'model' key not found"):
                ModelManager.load_model_artifacts(invalid_path)


class TestMetricsCalculation:
    """Comprehensive tests for metrics calculation functionality."""
    
    def test_regression_metrics_calculation(self):
        """Test comprehensive regression metrics calculation."""
        # Create synthetic data
        np.random.seed(42)
        y_true = np.random.normal(0, 1, 100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add some noise
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Validate metric types and ranges
        assert isinstance(metrics['r2_score'], float), "R² score should be float"
        assert isinstance(metrics['mse'], float), "MSE should be float"
        assert isinstance(metrics['rmse'], float), "RMSE should be float"
        assert isinstance(metrics['mae'], float), "MAE should be float"
        assert isinstance(metrics['mape'], float), "MAPE should be float"
        
        # Validate metric ranges
        assert 0 <= metrics['r2_score'] <= 1, "R² score should be between 0 and 1"
        assert metrics['mse'] >= 0, "MSE should be non-negative"
        assert metrics['rmse'] >= 0, "RMSE should be non-negative"
        assert metrics['mae'] >= 0, "MAE should be non-negative"
        assert metrics['mape'] >= 0, "MAPE should be non-negative"
        
        # Validate additional metrics
        assert isinstance(metrics['residual_std'], float), "Residual std should be float"
        assert isinstance(metrics['residual_mean'], float), "Residual mean should be float"
        assert isinstance(metrics['max_error'], float), "Max error should be float"
        assert isinstance(metrics['median_absolute_error'], float), "Median absolute error should be float"
    
    def test_metrics_with_perfect_predictions(self):
        """Test metrics calculation with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect predictions
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Perfect predictions should have specific values
        assert metrics['r2_score'] == 1.0, "Perfect predictions should have R² = 1.0"
        assert metrics['mse'] == 0.0, "Perfect predictions should have MSE = 0.0"
        assert metrics['rmse'] == 0.0, "Perfect predictions should have RMSE = 0.0"
        assert metrics['mae'] == 0.0, "Perfect predictions should have MAE = 0.0"
        assert metrics['max_error'] == 0.0, "Perfect predictions should have max error = 0.0"
    
    def test_metrics_with_constant_predictions(self):
        """Test metrics calculation with constant predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant predictions
        
        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Constant predictions should have specific characteristics
        assert metrics['r2_score'] <= 0.0, "Constant predictions should have R² ≤ 0.0"
        assert metrics['mse'] > 0.0, "Constant predictions should have positive MSE"
        assert metrics['mae'] > 0.0, "Constant predictions should have positive MAE"
    
    def test_metrics_input_validation(self):
        """Test metrics calculation input validation."""
        # Test with different shapes
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different length
        
        with pytest.raises(ValueError, match="different shapes"):
            MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        
        # Test with empty arrays
        with pytest.raises(ValueError, match="Empty arrays"):
            MetricsCalculator.calculate_regression_metrics(np.array([]), np.array([]))


class TestQuantizationEngine:
    """Comprehensive tests for quantization functionality."""
    
    def test_basic_quantization(self):
        """Test basic quantization and dequantization."""
        # Create test data
        original_values = np.array([0.1, 0.5, 1.0, -0.3, 0.8])
        
        # Quantize
        quantized, metadata = QuantizationEngine.quantize_parameters(original_values)
        
        # Validate quantization results
        assert quantized.dtype == np.uint8, "Quantized values should be uint8"
        assert quantized.shape == original_values.shape, "Shape should be preserved"
        assert quantized.min() >= 0, "Quantized values should be non-negative"
        assert quantized.max() <= 255, "Quantized values should be ≤ 255"
        
        # Dequantize
        dequantized = QuantizationEngine.dequantize_parameters(quantized, metadata)
        
        # Validate dequantization
        assert dequantized.dtype == np.float32, "Dequantized values should be float32"
        assert dequantized.shape == original_values.shape, "Shape should be preserved"
        
        # Check accuracy (should be close to original)
        max_error = np.max(np.abs(original_values - dequantized))
        assert max_error < 0.1, f"Quantization error {max_error} should be small"
    
    def test_quantization_with_zeros(self):
        """Test quantization with zero values."""
        original_values = np.zeros(5)
        
        quantized, metadata = QuantizationEngine.quantize_parameters(original_values)
        dequantized = QuantizationEngine.dequantize_parameters(quantized, metadata)
        
        # Zeros should be preserved exactly
        np.testing.assert_array_almost_equal(original_values, dequantized, decimal=10)
    
    def test_quantization_with_constant_values(self):
        """Test quantization with constant values."""
        original_values = np.full(5, 0.5)
        
        quantized, metadata = QuantizationEngine.quantize_parameters(original_values)
        dequantized = QuantizationEngine.dequantize_parameters(quantized, metadata)
        
        # Constant values should be preserved approximately
        max_error = np.max(np.abs(original_values - dequantized))
        assert max_error < 0.1, f"Constant value quantization error {max_error} should be small"
    
    def test_quantization_with_large_values(self):
        """Test quantization with large values."""
        original_values = np.array([1000.0, -500.0, 750.0])
        
        quantized, metadata = QuantizationEngine.quantize_parameters(original_values)
        dequantized = QuantizationEngine.dequantize_parameters(quantized, metadata)
        
        # Large values should be quantized with reasonable accuracy
        max_error = np.max(np.abs(original_values - dequantized))
        relative_error = max_error / np.max(np.abs(original_values))
        assert relative_error < 0.01, f"Relative quantization error {relative_error} should be small"
    
    def test_quantization_methods(self):
        """Test different quantization methods."""
        original_values = np.array([0.1, 0.5, 1.0, -0.3, 0.8])
        
        # Test adaptive method
        quant_adaptive, meta_adaptive = QuantizationEngine.quantize_parameters(
            original_values, method='adaptive'
        )
        
        # Test with fixed scale factor
        quant_fixed, meta_fixed = QuantizationEngine.quantize_parameters(
            original_values, scale_factor=100.0, method='fixed'
        )
        
        # Both methods should produce valid results
        assert quant_adaptive.dtype == np.uint8
        assert quant_fixed.dtype == np.uint8
        assert meta_adaptive['method'] == 'adaptive'
        assert meta_fixed['method'] == 'fixed'


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_complete_training_pipeline(self):
        """Test the complete training pipeline integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load data
            X_train, X_test, y_train, y_test = load_dataset()
            
            # Create and train model
            model = create_model()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2, mse = calculate_metrics(y_test, y_pred)
            
            # Validate results
            assert isinstance(model, LinearRegression), "Should be LinearRegression model"
            assert hasattr(model, 'coef_'), "Model should have coefficients"
            assert hasattr(model, 'intercept_'), "Model should have intercept"
            assert model.coef_.shape == (8,), "Should have 8 coefficients"
            assert isinstance(r2, float), "R² score should be float"
            assert isinstance(mse, float), "MSE should be float"
            assert 0 <= r2 <= 1, "R² score should be between 0 and 1"
            assert mse >= 0, "MSE should be non-negative"
            
            # Test model persistence
            model_path = os.path.join(temp_dir, "test_model.joblib")
            save_model(model, model_path)
            
            # Test model loading
            loaded_model = load_model(model_path)
            loaded_pred = loaded_model.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(y_pred, loaded_pred, decimal=10)
    
    def test_model_performance_threshold(self):
        """Test that model performance meets minimum threshold."""
        # Load data and train model
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        r2, mse = calculate_metrics(y_test, y_pred)
        
        # Performance validation
        assert r2 > 0.5, f"R² score {r2:.4f} should be above 0.5"
        assert mse > 0, "MSE should be positive"
        
        print(f"✅ Model Performance: R² = {r2:.4f}, MSE = {mse:.4f}")


class TestLegacyFunctions:
    """Tests for legacy function compatibility."""
    
    def test_legacy_dataset_loading(self):
        """Test legacy dataset loading function."""
        X_train, X_test, y_train, y_test = load_dataset()
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert X_train.shape[1] == 8
    
    def test_legacy_model_creation(self):
        """Test legacy model creation function."""
        model = create_model()
        assert isinstance(model, LinearRegression)
    
    def test_legacy_metrics_calculation(self):
        """Test legacy metrics calculation function."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        r2, mse = calculate_metrics(y_true, y_pred)
        assert isinstance(r2, float)
        assert isinstance(mse, float)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])