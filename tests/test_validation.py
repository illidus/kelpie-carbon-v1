"""
Test suite for the validation module (Task 2 - SKEMA Integration).

Tests the BC validation framework components:
- ValidationDataManager
- MockValidationGenerator
- ValidationMetrics
- FieldCampaignProtocols
"""

import pytest
from pathlib import Path
from datetime import datetime

from kelpie_carbon_v1.validation import (
    ValidationDataManager,
    MockValidationGenerator,
    ValidationMetrics,
    FieldCampaignProtocols,
)


def test_validation_data_manager(tmp_path):
    """Test ValidationDataManager basic functionality."""
    data_manager = ValidationDataManager(tmp_path)
    
    try:
        # Check directories created
        assert (tmp_path / "field_campaigns").exists()
        assert (tmp_path / "validation.db").exists()
    finally:
        # Clean up
        data_manager.close()
        # Force garbage collection to close any lingering DB connections
        import gc
        gc.collect()


def test_mock_validation_generator(tmp_path):
    """Test MockValidationGenerator."""
    data_manager = ValidationDataManager(tmp_path)
    generator = MockValidationGenerator(data_manager)
    
    try:
        # Test BC dataset creation
        campaign_id = generator.create_bc_validation_dataset("saanich_inlet")
        assert campaign_id.startswith("bc_saanich_inlet_")
        
        # Check measurements
        measurements = data_manager.get_ground_truth_for_campaign(campaign_id)
        assert len(measurements) == 50
        
        # Check kelp measurements exist and have spectral data
        kelp_measurements = [m for m in measurements if m.kelp_present]
        assert len(kelp_measurements) > 0
        
        # At least some kelp measurements should have spectral data
        kelp_with_spectra = [m for m in kelp_measurements if m.spectral_data]
        assert len(kelp_with_spectra) >= len(kelp_measurements)  # All kelp should have spectral data
    finally:
        # Clean up
        data_manager.close()
        import gc
        gc.collect()


def test_validation_metrics():
    """Test ValidationMetrics calculations."""
    metrics_calc = ValidationMetrics()
    
    # Test data
    ground_truth = [True, True, False, False, True]
    ndre_predictions = [True, True, False, True, True]
    ndvi_predictions = [True, False, False, False, False]
    
    results = metrics_calc.calculate_detection_metrics(
        ground_truth, ndre_predictions, ndvi_predictions
    )
    
    assert "ndre_metrics" in results
    assert "improvements" in results


def test_field_protocols():
    """Test FieldCampaignProtocols."""
    protocols = FieldCampaignProtocols()
    bc_protocols = protocols.get_bc_protocols()
    
    assert "gps_mapping" in bc_protocols
    assert "spectral_measurements" in bc_protocols


def test_integration_workflow(tmp_path):
    """Test complete validation workflow.""" 
    # Initialize components
    data_manager = ValidationDataManager(tmp_path)
    generator = MockValidationGenerator(data_manager)
    metrics_calc = ValidationMetrics()
    
    try:
        # Generate validation dataset
        campaign_id = generator.create_bc_validation_dataset("saanich_inlet")
        
        # Get validation data
        campaign = data_manager.get_campaign(campaign_id)
        measurements = data_manager.get_ground_truth_for_campaign(campaign_id)
        
        assert campaign is not None
        assert len(measurements) > 0
        
        # Simulate validation analysis
        ground_truth = [m.kelp_present for m in measurements]
        
        # Mock NDRE/NDVI predictions (NDRE slightly better)
        ndre_predictions = [gt if i % 5 != 0 else not gt for i, gt in enumerate(ground_truth)]
        ndvi_predictions = [gt if i % 6 != 0 else not gt for i, gt in enumerate(ground_truth)]
        
        # Calculate metrics
        validation_results = metrics_calc.calculate_detection_metrics(
            ground_truth, ndre_predictions, ndvi_predictions
        )
        
        # Generate report
        report = metrics_calc.generate_report(campaign_id, validation_results)
        
        # Verify complete workflow
        assert report["campaign_id"] == campaign_id
        assert "skema_score" in report
        assert "validation_results" in report
    finally:
        # Clean up
        data_manager.close()
        import gc
        gc.collect()


@pytest.fixture
def tmp_path():
    """Temporary directory fixture.""" 
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 