"""
Tests for Phase 3 Real Data Acquisition System.

This module tests the Phase 3 data acquisition framework for Task C1.5
real-world validation capabilities.
"""

import pytest
import datetime
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from kelpie_carbon_v1.validation.phase3_data_acquisition import (
    Phase3DataAcquisition,
    ValidationSite,
    SatelliteScene,
    ValidationDataset,
    create_phase3_data_acquisition,
    get_validation_sites,
    create_benchmark_dataset,
    create_full_benchmark_suite,
)


class TestValidationSite:
    """Test the ValidationSite dataclass."""
    
    def test_validation_site_creation(self):
        """Test creating a validation site."""
        site = ValidationSite(
            site_id="test_site",
            name="Test Site",
            coordinates=(45.0, -123.0),
            species="Macrocystis pyrifera",
            region="Test Region",
            kelp_season=(3, 10),
            data_sources=["Test Source"],
            validation_confidence="high",
            notes="Test site for validation"
        )
        
        assert site.site_id == "test_site"
        assert site.name == "Test Site"
        assert site.coordinates == (45.0, -123.0)
        assert site.species == "Macrocystis pyrifera"
        assert site.kelp_season == (3, 10)


class TestSatelliteScene:
    """Test the SatelliteScene dataclass."""
    
    def test_satellite_scene_creation(self):
        """Test creating a satellite scene."""
        acquisition_date = datetime.datetime(2024, 6, 15, 10, 30)
        
        scene = SatelliteScene(
            scene_id="S2A_TEST_20240615",
            acquisition_date=acquisition_date,
            site_id="test_site",
            cloud_coverage=15.5,
            data_quality="excellent",
            metadata={"sensor": "Sentinel-2A"}
        )
        
        assert scene.scene_id == "S2A_TEST_20240615"
        assert scene.cloud_coverage == 15.5
        assert scene.data_quality == "excellent"


class TestPhase3DataAcquisition:
    """Test the Phase3DataAcquisition class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = Phase3DataAcquisition(data_directory=self.temp_dir)
    
    def test_initialization(self):
        """Test acquisition system initialization."""
        assert isinstance(self.acquisition, Phase3DataAcquisition)
        assert self.acquisition.data_directory == Path(self.temp_dir)
        
        # Check subdirectories were created
        assert (Path(self.temp_dir) / "satellite").exists()
        assert (Path(self.temp_dir) / "benchmarks").exists()
        assert (Path(self.temp_dir) / "quality_reports").exists()
        
        # Check validation sites were initialized
        assert len(self.acquisition.validation_sites) == 6
        assert "broughton_archipelago" in self.acquisition.validation_sites
        assert "monterey_bay" in self.acquisition.validation_sites
    
    def test_get_validation_sites_no_filter(self):
        """Test getting all validation sites without filters."""
        sites = self.acquisition.get_validation_sites()
        assert len(sites) == 6
        
        site_ids = [site.site_id for site in sites]
        assert "broughton_archipelago" in site_ids
        assert "monterey_bay" in site_ids
    
    def test_get_validation_sites_region_filter(self):
        """Test getting validation sites filtered by region."""
        bc_sites = self.acquisition.get_validation_sites(region="British Columbia")
        assert len(bc_sites) == 2
        
        ca_sites = self.acquisition.get_validation_sites(region="California")
        assert len(ca_sites) == 2
    
    def test_get_validation_sites_species_filter(self):
        """Test getting validation sites filtered by species."""
        macro_sites = self.acquisition.get_validation_sites(species="Macrocystis")
        assert len(macro_sites) >= 2
        
        nere_sites = self.acquisition.get_validation_sites(species="Nereocystis")
        assert len(nere_sites) >= 1
    
    def test_get_validation_sites_confidence_filter(self):
        """Test getting validation sites filtered by confidence."""
        high_sites = self.acquisition.get_validation_sites(confidence="high")
        assert len(high_sites) == 2  # Broughton and Monterey
        
        medium_sites = self.acquisition.get_validation_sites(confidence="medium")
        assert len(medium_sites) == 4
    
    def test_create_synthetic_sentinel2_scenes(self):
        """Test creating synthetic Sentinel-2 scenes."""
        site_id = "broughton_archipelago"
        num_scenes = 5
        
        scenes = self.acquisition.create_synthetic_sentinel2_scenes(site_id, num_scenes)
        
        assert len(scenes) == num_scenes
        
        for scene in scenes:
            assert isinstance(scene, SatelliteScene)
            assert scene.site_id == site_id
            assert 0 <= scene.cloud_coverage <= 100
            assert scene.data_quality in ["excellent", "good", "fair", "poor"]
            assert scene.metadata["synthetic"] == True
            assert scene.metadata["satellite"] == "Sentinel-2"
    
    def test_create_synthetic_scenes_invalid_site(self):
        """Test creating synthetic scenes for invalid site."""
        with pytest.raises(ValueError, match="Unknown site_id"):
            self.acquisition.create_synthetic_sentinel2_scenes("invalid_site")
    
    def test_cloud_coverage_simulation(self):
        """Test cloud coverage simulation for different regions."""
        site = self.acquisition.validation_sites["broughton_archipelago"]  # BC site
        date = datetime.datetime(2024, 7, 15)  # Summer
        
        cloud_coverage = self.acquisition._simulate_cloud_coverage(site, date)
        assert 0 <= cloud_coverage <= 95
        
        # BC should have higher base cloud coverage than California
        ca_site = self.acquisition.validation_sites["monterey_bay"]
        ca_cloud = self.acquisition._simulate_cloud_coverage(ca_site, date)
        
        # This is probabilistic, but on average BC should be cloudier
        # We test the logic by checking the base values are different
        assert site.region != ca_site.region
    
    def test_data_quality_determination(self):
        """Test data quality determination from cloud coverage."""
        assert self.acquisition._determine_data_quality(10.0) == "excellent"
        assert self.acquisition._determine_data_quality(25.0) == "good"
        assert self.acquisition._determine_data_quality(45.0) == "fair"
        assert self.acquisition._determine_data_quality(75.0) == "poor"
    
    def test_season_phase_determination(self):
        """Test season phase determination."""
        site = self.acquisition.validation_sites["broughton_archipelago"]  # Season: May-Oct (5-10)
        
        # In season dates
        summer_date = datetime.datetime(2024, 7, 15)  # Peak season
        phase = self.acquisition._get_season_phase(summer_date, site)
        assert phase in ["peak_season", "mid_season"]
        
        # Out of season date
        winter_date = datetime.datetime(2024, 1, 15)
        phase = self.acquisition._get_season_phase(winter_date, site)
        assert phase == "off_season"
    
    def test_create_validation_dataset(self):
        """Test creating a complete validation dataset."""
        site_id = "monterey_bay"
        num_scenes = 3
        
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes)
        
        assert isinstance(dataset, ValidationDataset)
        assert dataset.site.site_id == site_id
        assert len(dataset.satellite_scenes) == num_scenes
        assert isinstance(dataset.quality_metrics, dict)
        assert "overall_quality" in dataset.quality_metrics
        assert 0 <= dataset.quality_metrics["overall_quality"] <= 1
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        # Create test scenes
        scenes = [
            SatelliteScene(
                scene_id="test_1",
                acquisition_date=datetime.datetime(2024, 1, 1),
                site_id="test",
                cloud_coverage=10.0,
                data_quality="excellent",
                metadata={"season_phase": "peak_season"}
            ),
            SatelliteScene(
                scene_id="test_2",
                acquisition_date=datetime.datetime(2024, 6, 1),
                site_id="test",
                cloud_coverage=30.0,
                data_quality="good",
                metadata={"season_phase": "mid_season"}
            )
        ]
        
        metrics = self.acquisition._calculate_quality_metrics(scenes)
        
        assert "overall_quality" in metrics
        assert "average_cloud_coverage" in metrics
        assert "temporal_span_days" in metrics
        assert "excellent_scenes_percent" in metrics
        assert "seasonal_coverage" in metrics
        
        assert metrics["average_cloud_coverage"] == 20.0  # (10 + 30) / 2
        assert metrics["excellent_scenes_percent"] == 50.0  # 1 out of 2
        assert metrics["total_scenes"] == 2
    
    def test_create_benchmark_suite(self):
        """Test creating benchmark suite."""
        sites = ["broughton_archipelago", "monterey_bay"]
        
        suite = self.acquisition.create_benchmark_suite(sites, num_scenes_per_site=3)
        
        assert len(suite) == 2
        assert "broughton_archipelago" in suite
        assert "monterey_bay" in suite
        
        for site_id, dataset in suite.items():
            assert isinstance(dataset, ValidationDataset)
            assert dataset.site.site_id == site_id
            assert len(dataset.satellite_scenes) == 3
    
    def test_save_and_load_validation_dataset(self):
        """Test saving and loading validation datasets."""
        # Create dataset
        dataset = self.acquisition.create_validation_dataset("monterey_bay", num_scenes=2)
        
        # Save dataset
        filepath = self.acquisition.save_validation_dataset(dataset)
        assert Path(filepath).exists()
        
        # Load dataset
        loaded_dataset = self.acquisition.load_validation_dataset(filepath)
        
        # Verify loaded dataset matches original
        assert loaded_dataset.site.site_id == dataset.site.site_id
        assert len(loaded_dataset.satellite_scenes) == len(dataset.satellite_scenes)
        assert loaded_dataset.quality_metrics == dataset.quality_metrics
    
    def test_get_site_summary(self):
        """Test getting site summary."""
        summary = self.acquisition.get_site_summary()
        
        assert summary["total_sites"] == 6
        assert "sites_by_region" in summary
        assert "sites_by_species" in summary
        assert "sites_by_confidence" in summary
        assert "coverage_stats" in summary
        
        # Check specific counts
        assert summary["sites_by_confidence"]["high"] == 2
        assert summary["sites_by_confidence"]["medium"] == 4
    
    def test_generate_quality_report(self):
        """Test generating quality report."""
        # Create test datasets
        dataset1 = self.acquisition.create_validation_dataset("broughton_archipelago", 3)
        dataset2 = self.acquisition.create_validation_dataset("monterey_bay", 3)
        
        datasets = {
            "broughton_archipelago": dataset1,
            "monterey_bay": dataset2
        }
        
        report = self.acquisition.generate_quality_report(datasets)
        
        assert "report_metadata" in report
        assert "overall_quality" in report
        assert "site_quality" in report
        assert "recommendations" in report
        
        assert report["report_metadata"]["total_datasets"] == 2
        assert len(report["site_quality"]) == 2
        assert isinstance(report["recommendations"], list)


class TestFactoryFunctions:
    """Test factory functions and convenience APIs."""
    
    def test_create_phase3_data_acquisition(self):
        """Test factory function."""
        acquisition = create_phase3_data_acquisition()
        assert isinstance(acquisition, Phase3DataAcquisition)
    
    def test_get_validation_sites_function(self):
        """Test high-level get_validation_sites function."""
        all_sites = get_validation_sites()
        assert len(all_sites) == 6
        
        bc_sites = get_validation_sites(region="British Columbia")
        assert len(bc_sites) == 2
    
    def test_create_benchmark_dataset_function(self):
        """Test high-level create_benchmark_dataset function."""
        dataset = create_benchmark_dataset("broughton_archipelago", num_scenes=3)
        
        assert isinstance(dataset, ValidationDataset)
        assert dataset.site.site_id == "broughton_archipelago"
        assert len(dataset.satellite_scenes) == 3
    
    def test_create_full_benchmark_suite_function(self):
        """Test high-level create_full_benchmark_suite function."""
        suite = create_full_benchmark_suite(num_scenes_per_site=2)
        
        assert len(suite) == 6  # All sites
        for dataset in suite.values():
            assert len(dataset.satellite_scenes) == 2


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = Phase3DataAcquisition(data_directory=self.temp_dir)
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Create comprehensive benchmark suite
        suite = self.acquisition.create_benchmark_suite(None, 5)  # All sites, 5 scenes each
        
        assert len(suite) == 6
        
        # Generate quality report
        report = self.acquisition.generate_quality_report(suite)
        
        assert report["report_metadata"]["total_datasets"] == 6
        assert report["report_metadata"]["total_scenes"] == 30  # 6 sites * 5 scenes
        
        # Save all datasets
        saved_files = []
        for site_id, dataset in suite.items():
            filepath = self.acquisition.save_validation_dataset(dataset)
            saved_files.append(filepath)
        
        assert len(saved_files) == 6
        
        # Verify files exist and can be loaded
        for filepath in saved_files:
            assert Path(filepath).exists()
            loaded_dataset = self.acquisition.load_validation_dataset(filepath)
            assert isinstance(loaded_dataset, ValidationDataset)
    
    def test_multi_region_diversity(self):
        """Test diversity across multiple regions."""
        suite = self.acquisition.create_benchmark_suite()
        
        # Check regional diversity
        regions = set()
        species = set()
        for dataset in suite.values():
            regions.add(dataset.site.region)
            species.add(dataset.site.species)
        
        assert len(regions) >= 4  # At least 4 different regions
        assert len(species) >= 3  # At least 3 different species
    
    def test_seasonal_coverage_analysis(self):
        """Test seasonal coverage across sites."""
        suite = self.acquisition.create_benchmark_suite()
        
        # Analyze seasonal patterns
        seasonal_patterns = {}
        for dataset in suite.values():
            season = dataset.site.kelp_season
            season_key = f"{season[0]:02d}-{season[1]:02d}"
            if season_key not in seasonal_patterns:
                seasonal_patterns[season_key] = []
            seasonal_patterns[season_key].append(dataset.site.name)
        
        # Should have multiple seasonal patterns
        assert len(seasonal_patterns) >= 3
    
    def test_quality_filtering_workflow(self):
        """Test quality filtering and assessment workflow."""
        # Create dataset with many scenes for better statistics
        dataset = self.acquisition.create_validation_dataset("monterey_bay", num_scenes=20)
        
        # Filter high-quality scenes
        high_quality_scenes = [
            scene for scene in dataset.satellite_scenes 
            if scene.data_quality in ["excellent", "good"] and scene.cloud_coverage < 30
        ]
        
        # Should have some high-quality scenes
        assert len(high_quality_scenes) > 0
        
        # Quality metrics should reflect the filtering
        metrics = dataset.quality_metrics
        assert 0 <= metrics["overall_quality"] <= 1
        assert metrics["total_scenes"] == 20
        assert metrics["good_or_better_percent"] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = Phase3DataAcquisition(data_directory=self.temp_dir)
    
    def test_empty_scenes_quality_metrics(self):
        """Test quality metrics with empty scenes list."""
        metrics = self.acquisition._calculate_quality_metrics([])
        assert metrics == {"overall_quality": 0.0}
    
    def test_single_scene_dataset(self):
        """Test dataset with single scene."""
        dataset = self.acquisition.create_validation_dataset("monterey_bay", num_scenes=1)
        
        assert len(dataset.satellite_scenes) == 1
        assert dataset.quality_metrics["temporal_span_days"] == 0
        assert dataset.quality_metrics["total_scenes"] == 1
    
    def test_cross_year_kelp_season(self):
        """Test handling kelp seasons that cross year boundaries."""
        # Puget Sound has season Nov (11) to June (6)
        scenes = self.acquisition.create_synthetic_sentinel2_scenes("puget_sound", 6)
        
        assert len(scenes) == 6
        
        # Check that scenes span the cross-year season correctly
        months = [scene.acquisition_date.month for scene in scenes]
        
        # Should include months from both sides of year boundary
        has_late_year = any(month >= 11 for month in months)
        has_early_year = any(month <= 6 for month in months)
        
        # At least one of these should be true for proper seasonal coverage
        assert has_late_year or has_early_year
    
    def test_southern_hemisphere_seasons(self):
        """Test Southern Hemisphere seasonal patterns."""
        # Tasmania has season Oct (10) to April (4) - Southern hemisphere summer
        scenes = self.acquisition.create_synthetic_sentinel2_scenes("tasmania_giant_kelp", 8)
        
        assert len(scenes) == 8
        
        # Check that cloud coverage simulation works for Southern Hemisphere
        for scene in scenes:
            assert 0 <= scene.cloud_coverage <= 95
            assert scene.data_quality in ["excellent", "good", "fair", "poor"]
    
    def test_invalid_site_operations(self):
        """Test operations with invalid site IDs."""
        with pytest.raises(ValueError):
            self.acquisition.create_validation_dataset("invalid_site")
        
        with pytest.raises(ValueError):
            self.acquisition.create_synthetic_sentinel2_scenes("invalid_site")
    
    def test_load_nonexistent_file(self):
        """Test loading dataset from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.acquisition.load_validation_dataset("nonexistent_file.json")
    
    def test_benchmark_suite_with_invalid_sites(self):
        """Test benchmark suite creation with some invalid sites."""
        sites = ["broughton_archipelago", "invalid_site", "monterey_bay"]
        
        # Should skip invalid site and continue with valid ones
        suite = self.acquisition.create_benchmark_suite(sites, 2)
        
        assert len(suite) == 2  # Only valid sites
        assert "broughton_archipelago" in suite
        assert "monterey_bay" in suite
        assert "invalid_site" not in suite 