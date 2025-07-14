"""
Tests for Lambda³ core functionality
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from core.lambda3_zeroshot_tensor_field import (
    L3Config,
    calc_lambda3_features,
    calculate_rho_t,
    detect_jumps,
    calculate_diff_and_threshold,
    Lambda3FinancialRegimeDetector
)

class TestLambda3Core:
    """Test core Lambda³ functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        np.random.seed(42)
        n_points = 200
        
        # Create data with known structural changes
        data = np.zeros(n_points)
        
        # Add trend
        data += np.linspace(0, 2, n_points)
        
        # Add noise
        data += np.random.randn(n_points) * 0.1
        
        # Add jumps at known positions
        jump_positions = [50, 100, 150]
        jump_sizes = [1.0, -0.8, 1.2]
        
        for pos, size in zip(jump_positions, jump_sizes):
            data[pos:] += size
        
        return data, jump_positions
    
    @pytest.fixture
    def multi_series_data(self):
        """Generate multiple time series"""
        np.random.seed(42)
        n_points = 300
        
        data_dict = {
            'series_1': np.cumsum(np.random.randn(n_points) * 0.1),
            'series_2': np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.randn(n_points) * 0.1,
            'series_3': np.cumsum(np.random.randn(n_points) * 0.2)
        }
        
        return data_dict
    
    def test_config_creation(self):
        """Test L3Config creation and defaults"""
        config = L3Config()
        
        assert config.T == 150
        assert config.window == 10
        assert config.draws == 8000
        assert config.tune == 8000
        assert config.hierarchical == True
    
    def test_calculate_rho_t(self, sample_data):
        """Test tension scalar calculation"""
        data, _ = sample_data
        window = 10
        
        rho_t = calculate_rho_t(data, window)
        
        # Check output shape
        assert len(rho_t) == len(data)
        
        # Check all values are non-negative
        assert np.all(rho_t >= 0)
        
        # Check that tension increases around jumps
        assert rho_t[55] > rho_t[45]  # After first jump
    
    def test_detect_jumps(self, sample_data):
        """Test jump detection"""
        data, jump_positions = sample_data
        
        # Calculate differences
        diff, threshold = calculate_diff_and_threshold(data, 95.0)
        
        # Detect jumps
        pos_jumps, neg_jumps = detect_jumps(diff, threshold)
        
        # Check that jumps are detected near expected positions
        for pos in jump_positions:
            # Check within small window
            window = range(max(0, pos-2), min(len(data), pos+3))
            assert any(pos_jumps[i] == 1 or neg_jumps[i] == 1 for i in window)
    
    def test_calc_lambda3_features(self, sample_data):
        """Test complete feature extraction"""
        data, _ = sample_data
        config = L3Config(hierarchical=False)
        
        features = calc_lambda3_features(data, config)
        
        # Check all required features are present
        required_keys = [
            'data', 'delta_LambdaC_pos', 'delta_LambdaC_neg',
            'rho_T', 'time_trend', 'local_jump_detect'
        ]
        
        for key in required_keys:
            assert key in features
            assert len(features[key]) == len(data)
    
    def test_hierarchical_features(self, sample_data):
        """Test hierarchical feature extraction"""
        data, _ = sample_data
        config = L3Config(hierarchical=True)
        
        features = calc_lambda3_features(data, config)
        
        # Check hierarchical features
        hierarchical_keys = [
            'local_pos', 'local_neg', 'global_pos', 'global_neg',
            'pure_local_pos', 'pure_local_neg',
            'pure_global_pos', 'pure_global_neg',
            'mixed_pos', 'mixed_neg'
        ]
        
        for key in hierarchical_keys:
            assert key in features
            assert len(features[key]) == len(data)
        
        # Check that hierarchical decomposition is valid
        total_pos = features['local_pos'] + features['global_pos']
        assert np.all(total_pos <= 2)  # Can be 0, 1, or 2
    
    def test_regime_detector(self, multi_series_data):
        """Test regime detection"""
        # Use first series
        data = list(multi_series_data.values())[0]
        
        # Extract features
        features = calc_lambda3_features(data, L3Config())
        
        # Detect regimes
        detector = Lambda3FinancialRegimeDetector(n_regimes=3)
        labels = detector.fit(features, data)
        
        # Check output
        assert len(labels) == len(data)
        assert set(labels).issubset({0, 1, 2})
        
        # Check regime features
        assert hasattr(detector, 'regime_features')
        assert len(detector.regime_features) == 3
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Very short series
        short_data = np.array([1.0, 2.0, 3.0])
        features = calc_lambda3_features(short_data, L3Config())
        assert len(features['rho_T']) == 3
        
        # Constant series
        const_data = np.ones(100)
        features = calc_lambda3_features(const_data, L3Config())
        assert np.sum(features['delta_LambdaC_pos']) == 0
        assert np.sum(features['delta_LambdaC_neg']) == 0
        
        # Series with NaN (should be handled)
        nan_data = np.array([1, 2, np.nan, 4, 5])
        with pytest.warns(RuntimeWarning):
            features = calc_lambda3_features(nan_data, L3Config())

class TestDataStructures:
    """Test data structures and utilities"""
    
    def test_empty_series_dict(self):
        """Test handling of empty series dictionary"""
        empty_dict = {}
        
        # Should handle gracefully
        from core.lambda3_zeroshot_tensor_field import run_lambda3_analysis
        
        with pytest.raises(Exception):
            results = run_lambda3_analysis(empty_dict, verbose=False)
    
    def test_series_alignment(self):
        """Test series of different lengths"""
        from utils.data_loader import align_series
        
        series_dict = {
            'short': np.array([1, 2, 3]),
            'medium': np.array([1, 2, 3, 4, 5]),
            'long': np.array([1, 2, 3, 4, 5, 6, 7])
        }
        
        # Test truncate method
        aligned = align_series(series_dict, method='truncate')
        assert all(len(s) == 3 for s in aligned.values())
        
        # Test pad method
        aligned = align_series(series_dict, method='pad')
        assert all(len(s) == 7 for s in aligned.values())


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        from core.lambda3_zeroshot_tensor_field import run_lambda3_analysis
        
        # Generate test data
        np.random.seed(42)
        test_data = {
            'A': np.cumsum(np.random.randn(100)),
            'B': np.cumsum(np.random.randn(100)),
            'C': np.cumsum(np.random.randn(100))
        }
        
        # Run analysis with minimal config for speed
        config = L3Config(
            draws=100,
            tune=100,
            hierarchical=False
        )
        
        results = run_lambda3_analysis(
            data_source=test_data,
            config=config,
            verbose=False
        )
        
        # Check results structure
        assert 'features_dict' in results
        assert 'series_names' in results
        assert len(results['series_names']) == 3
        
        # Check features extracted for each series
        for name in test_data.keys():
            assert name in results['features_dict']
