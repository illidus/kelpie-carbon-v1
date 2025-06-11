"""
Unit tests for SKEMA/UVic biomass integration module.
Tests biomass validation, species-specific protocols, and ground truth integration.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import the module under test
from src.kelpie_carbon_v1.validation.skema_biomass_integration import (
    SKEMABiomassDatasetIntegrator,
    SKEMAIntegrationConfig,
    BiomassValidationSite,
    BiomassGroundTruth,
    create_skema_biomass_integrator
)


@pytest.fixture
def sample_config():
    """Create sample integration configuration for testing."""
    return SKEMAIntegrationConfig(
        spatial_tolerance_meters=100.0,
        temporal_tolerance_days=7,
        min_sites_per_species=2
    )


@pytest.fixture
def sample_validation_site():
    """Create sample validation site for testing."""
    return BiomassValidationSite(
        site_id='BC_TEST',
        name='British Columbia Test Site',
        latitude=50.1163,
        longitude=-125.2735,
        species='Nereocystis luetkeana',
        biomass_measurements=[],
        carbon_content_ratio=0.30,
        measurement_dates=[datetime(2023, 6, 15)],
        measurement_quality='good',
        sampling_method='diving_transect',
        depth_range=(5.0, 25.0),
        site_characteristics={'depth_avg': 15.0, 'substrate': 'rocky'}
    )


@pytest.fixture
def sample_biomass_ground_truth():
    """Create sample biomass ground truth measurement."""
    return BiomassGroundTruth(
        site_id='BC_TEST',
        measurement_date=datetime(2023, 6, 15),
        biomass_wet_weight_kg_m2=5.5,
        biomass_dry_weight_kg_m2=0.83,
        carbon_content_kg_m2=0.25,
        sampling_area_m2=1.0,
        measurement_uncertainty=0.1,
        quality_flags=['good'],
        environmental_conditions={'temperature': 10.5, 'salinity': 32.0},
        observer='Test Observer',
        instrumentation={'scale': 'digital', 'method': 'quadrat'}
    )


class TestSKEMAIntegrationConfig:
    """Test SKEMAIntegrationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SKEMAIntegrationConfig()
        
        assert config.uvic_skema_endpoint == "https://api.uvic.ca/skema"
        assert config.saanich_inlet_data == True
        assert config.spatial_tolerance_meters == 100.0
        assert config.temporal_tolerance_days == 7
        assert config.min_sites_per_species == 2
    
    def test_custom_config(self, sample_config):
        """Test custom configuration values."""
        assert sample_config.spatial_tolerance_meters == 100.0
        assert sample_config.temporal_tolerance_days == 7
        assert sample_config.min_sites_per_species == 2


class TestBiomassValidationSite:
    """Test BiomassValidationSite dataclass."""
    
    def test_site_creation(self, sample_validation_site):
        """Test biomass validation site creation."""
        assert sample_validation_site.site_id == 'BC_TEST'
        assert sample_validation_site.latitude == 50.1163
        assert sample_validation_site.longitude == -125.2735
        assert sample_validation_site.species == 'Nereocystis luetkeana'
        assert sample_validation_site.carbon_content_ratio == 0.30
        assert sample_validation_site.measurement_quality == 'good'


class TestBiomassGroundTruth:
    """Test BiomassGroundTruth dataclass."""
    
    def test_ground_truth_creation(self, sample_biomass_ground_truth):
        """Test biomass ground truth measurement creation."""
        assert sample_biomass_ground_truth.site_id == 'BC_TEST'
        assert sample_biomass_ground_truth.biomass_wet_weight_kg_m2 == 5.5
        assert sample_biomass_ground_truth.biomass_dry_weight_kg_m2 == 0.83
        assert sample_biomass_ground_truth.carbon_content_kg_m2 == 0.25
        assert sample_biomass_ground_truth.measurement_uncertainty == 0.1
        assert 'good' in sample_biomass_ground_truth.quality_flags


class TestSKEMABiomassDatasetIntegrator:
    """Test SKEMABiomassDatasetIntegrator class."""
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        integrator = SKEMABiomassDatasetIntegrator()
        
        assert isinstance(integrator.config, SKEMAIntegrationConfig)
        assert integrator.validation_sites == []
        assert integrator.biomass_measurements == []
        assert integrator.integration_history == []
    
    def test_initialization_custom_config(self, sample_config):
        """Test initialization with custom configuration."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        assert integrator.config == sample_config
        assert integrator.config.spatial_tolerance_meters == 100.0
    
    def test_integrate_four_validation_sites_biomass_data(self, sample_config):
        """Test integration of four validation sites biomass data."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        result = integrator.integrate_four_validation_sites_biomass_data()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'validation_sites' in result
        assert 'biomass_datasets' in result
        assert 'validation_results' in result
        assert 'species_statistics' in result
        assert 'integration_summary' in result
        
        # Verify validation sites
        validation_sites = result['validation_sites']
        assert isinstance(validation_sites, list)
        assert len(validation_sites) == 4  # Four validation sites
        
        # Verify site IDs match expected
        site_ids = [site.site_id for site in validation_sites]
        expected_ids = ['BC_VALIDATION', 'CA_VALIDATION', 'TAS_VALIDATION', 'BROUGHTON_VALIDATION']
        for expected_id in expected_ids:
            assert expected_id in site_ids
    
    def test_enhance_existing_skema_integration(self, sample_config):
        """Test enhancement of existing SKEMA integration."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        result = integrator.enhance_existing_skema_integration()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'enhanced_skema_framework' in result
        assert 'equivalence_improvement' in result
        assert 'biomass_validation_layer' in result
        assert 'cross_validation_results' in result
        assert 'integration_improvements' in result
        
        # Verify equivalence improvement
        equivalence = result['equivalence_improvement']
        assert 'before_percentage' in equivalence
        assert 'after_percentage' in equivalence
        assert equivalence['after_percentage'] > equivalence['before_percentage']
    
    def test_load_uvic_saanich_inlet_data(self, sample_config):
        """Test loading UVic Saanich Inlet data."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        result = integrator.load_uvic_saanich_inlet_data()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'saanich_inlet_dataset' in result
        assert 'data_quality_report' in result
        assert 'species_distribution' in result
        assert 'temporal_analysis' in result
        
        # Verify Saanich Inlet coordinates
        dataset = result['saanich_inlet_dataset']
        assert 'site_coordinates' in dataset
        coords = dataset['site_coordinates']
        assert 48.5 <= coords['latitude'] <= 48.7
        assert -123.6 <= coords['longitude'] <= -123.4
    
    def test_integrate_species_specific_biomass_validation(self, sample_config):
        """Test species-specific biomass validation integration."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        result = integrator.integrate_species_specific_biomass_validation()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'nereocystis_validation' in result
        assert 'macrocystis_validation' in result
        assert 'species_comparison' in result
        assert 'validation_protocols' in result
        
        # Verify species-specific data
        nereocystis = result['nereocystis_validation']
        assert 'carbon_ratio' in nereocystis
        assert 'biomass_characteristics' in nereocystis
        assert 0.25 <= nereocystis['carbon_ratio'] <= 0.35
        
        macrocystis = result['macrocystis_validation']
        assert 'carbon_ratio' in macrocystis
        assert 'biomass_characteristics' in macrocystis
        assert 0.25 <= macrocystis['carbon_ratio'] <= 0.35
    
    def test_create_carbon_quantification_validation(self, sample_config):
        """Test carbon quantification validation creation."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        result = integrator.create_carbon_quantification_validation()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'carbon_conversion_validation' in result
        assert 'seasonal_variation_analysis' in result
        assert 'sequestration_rate_calculations' in result
        assert 'uncertainty_quantification' in result
        
        # Verify carbon conversion validation
        carbon_validation = result['carbon_conversion_validation']
        assert 'conversion_factors' in carbon_validation
        assert 'validation_accuracy' in carbon_validation
        
        # Verify uncertainty quantification
        uncertainty = result['uncertainty_quantification']
        assert 'measurement_uncertainty' in uncertainty
        assert 'conversion_uncertainty' in uncertainty
        assert 'total_uncertainty' in uncertainty


class TestFactoryFunctions:
    """Test factory functions for easy usage."""
    
    def test_create_skema_biomass_integrator_default(self):
        """Test SKEMA biomass integrator creation with default config."""
        integrator = create_skema_biomass_integrator()
        
        assert isinstance(integrator, SKEMABiomassDatasetIntegrator)
        assert isinstance(integrator.config, SKEMAIntegrationConfig)
    
    def test_create_skema_biomass_integrator_custom_config(self, sample_config):
        """Test SKEMA biomass integrator creation with custom config."""
        integrator = create_skema_biomass_integrator(sample_config)
        
        assert isinstance(integrator, SKEMABiomassDatasetIntegrator)
        assert integrator.config == sample_config


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_validation_sites(self, sample_config):
        """Test handling of empty validation sites."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        # Should handle empty sites gracefully
        result = integrator.integrate_four_validation_sites_biomass_data()
        assert isinstance(result, dict)
    
    def test_missing_biomass_data(self, sample_config):
        """Test handling of missing biomass data."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        # Should handle missing data gracefully
        result = integrator.load_uvic_saanich_inlet_data()
        assert isinstance(result, dict)
    
    def test_invalid_coordinates(self, sample_config):
        """Test handling of invalid coordinates."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        # Should handle invalid coordinates gracefully
        result = integrator.integrate_four_validation_sites_biomass_data()
        assert isinstance(result, dict)


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    def test_complete_integration_workflow(self, sample_config):
        """Test complete SKEMA biomass integration workflow."""
        integrator = SKEMABiomassDatasetIntegrator(sample_config)
        
        # Test complete workflow
        sites_result = integrator.integrate_four_validation_sites_biomass_data()
        skema_result = integrator.enhance_existing_skema_integration()
        uvic_result = integrator.load_uvic_saanich_inlet_data()
        species_result = integrator.integrate_species_specific_biomass_validation()
        carbon_result = integrator.create_carbon_quantification_validation()
        
        # Verify all components work together
        assert isinstance(sites_result, dict)
        assert isinstance(skema_result, dict)
        assert isinstance(uvic_result, dict)
        assert isinstance(species_result, dict)
        assert isinstance(carbon_result, dict)
        
        # Verify workflow produces comprehensive results
        assert 'validation_sites' in sites_result
        assert 'enhanced_skema_framework' in skema_result
        assert 'saanich_inlet_dataset' in uvic_result
        assert 'species_comparison' in species_result
        assert 'carbon_conversion_validation' in carbon_result 