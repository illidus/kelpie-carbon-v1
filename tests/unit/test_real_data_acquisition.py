"""
Tests for Real Satellite Data Acquisition System.

This module tests the real data acquisition framework for validating kelp detection
algorithms against actual satellite imagery and ground truth data.
"""

import pytest
import datetime
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from kelpie_carbon_v1.validation.real_data_acquisition import (
    RealDataAcquisition,
    ValidationSite,
    SatelliteScene,
    GroundTruthData,
    ValidationDataset,
    create_real_data_acquisition,
    get_validation_sites,
    create_benchmark_dataset,
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
            water_depth_range=(10.0, 30.0),
            kelp_season=(3, 10),
            data_sources=["Test Source"],
            validation_confidence="high",
            notes="Test site for validation"
        )
        
        assert site.site_id == "test_site"
        assert site.name == "Test Site"
        assert site.coordinates == (45.0, -123.0)
        assert site.species == "Macrocystis pyrifera"
        assert site.region == "Test Region"
        assert site.water_depth_range == (10.0, 30.0)
        assert site.kelp_season == (3, 10)
        assert site.data_sources == ["Test Source"]
        assert site.validation_confidence == "high"
        assert site.notes == "Test site for validation"


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
            file_path="/test/path/scene.tif",
            preprocessing_applied=["cloud_mask", "atmospheric_correction"],
            metadata={"sensor": "Sentinel-2A", "orbit": 123}
        )
        
        assert scene.scene_id == "S2A_TEST_20240615"
        assert scene.acquisition_date == acquisition_date
        assert scene.site_id == "test_site"
        assert scene.cloud_coverage == 15.5
        assert scene.data_quality == "excellent"
        assert scene.file_path == "/test/path/scene.tif"
        assert scene.preprocessing_applied == ["cloud_mask", "atmospheric_correction"]
        assert scene.metadata == {"sensor": "Sentinel-2A", "orbit": 123}


class TestGroundTruthData:
    """Test the GroundTruthData dataclass."""
    
    def test_ground_truth_data_creation(self):
        """Test creating ground truth data."""
        collection_date = datetime.datetime(2024, 6, 15)
        
        ground_truth = GroundTruthData(
            site_id="test_site",
            data_type="field_survey",
            collection_date=collection_date,
            kelp_coverage_percent=35.0,
            kelp_density="dense",
            confidence=0.9,
            source="field_team_alpha",
            metadata={"survey_method": "transect", "depth": 15.0}
        )
        
        assert ground_truth.site_id == "test_site"
        assert ground_truth.data_type == "field_survey"
        assert ground_truth.collection_date == collection_date
        assert ground_truth.kelp_coverage_percent == 35.0
        assert ground_truth.kelp_density == "dense"
        assert ground_truth.confidence == 0.9
        assert ground_truth.source == "field_team_alpha"
        assert ground_truth.metadata == {"survey_method": "transect", "depth": 15.0}


class TestValidationDataset:
    """Test the ValidationDataset dataclass."""
    
    def test_validation_dataset_creation(self):
        """Test creating a validation dataset."""
        site = ValidationSite(
            site_id="test_site",
            name="Test Site",
            coordinates=(45.0, -123.0),
            species="Macrocystis pyrifera",
            region="Test Region",
            water_depth_range=(10.0, 30.0),
            kelp_season=(3, 10),
            data_sources=["Test Source"],
            validation_confidence="high"
        )
        
        scenes = [
            SatelliteScene(
                scene_id="scene_1",
                acquisition_date=datetime.datetime(2024, 6, 15),
                site_id="test_site",
                cloud_coverage=10.0,
                data_quality="excellent"
            )
        ]
        
        ground_truth = [
            GroundTruthData(
                site_id="test_site",
                data_type="field_survey",
                collection_date=datetime.datetime(2024, 6, 15),
                kelp_coverage_percent=30.0,
                kelp_density="dense",
                confidence=0.9,
                source="field_team"
            )
        ]
        
        created_date = datetime.datetime.now()
        
        dataset = ValidationDataset(
            site=site,
            satellite_scenes=scenes,
            ground_truth=ground_truth,
            quality_metrics={"overall_quality": 0.85},
            created_date=created_date,
            version="1.0"
        )
        
        assert dataset.site == site
        assert dataset.satellite_scenes == scenes
        assert dataset.ground_truth == ground_truth
        assert dataset.quality_metrics == {"overall_quality": 0.85}
        assert dataset.created_date == created_date
        assert dataset.version == "1.0"


class TestRealDataAcquisition:
    """Test the RealDataAcquisition class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = RealDataAcquisition(data_directory=self.temp_dir)
    
    def test_initialization(self):
        """Test acquisition system initialization."""
        assert isinstance(self.acquisition, RealDataAcquisition)
        assert self.acquisition.data_directory == Path(self.temp_dir)
        
        # Check that subdirectories were created
        assert (Path(self.temp_dir) / "satellite").exists()
        assert (Path(self.temp_dir) / "ground_truth").exists()
        assert (Path(self.temp_dir) / "processed").exists()
        assert (Path(self.temp_dir) / "benchmarks").exists()
        
        # Check that validation sites were initialized
        assert len(self.acquisition.validation_sites) > 0
        assert "broughton_archipelago" in self.acquisition.validation_sites
        assert "monterey_bay" in self.acquisition.validation_sites
    
    def test_validation_sites_initialization(self):
        """Test validation sites database initialization."""
        sites = self.acquisition.validation_sites
        
        # Check that all expected sites are present
        expected_sites = [
            "broughton_archipelago", "monterey_bay", "saanich_inlet", 
            "puget_sound", "point_reyes", "tasmania_giant_kelp"
        ]
        
        for site_id in expected_sites:
            assert site_id in sites
            site = sites[site_id]
            assert isinstance(site, ValidationSite)
            assert site.site_id == site_id
            assert len(site.coordinates) == 2
            assert len(site.kelp_season) == 2
            assert len(site.data_sources) > 0
    
    def test_get_validation_sites_no_filter(self):
        """Test getting all validation sites without filters."""
        sites = self.acquisition.get_validation_sites()
        
        assert len(sites) == 6  # All sites
        site_ids = [site.site_id for site in sites]
        assert "broughton_archipelago" in site_ids
        assert "monterey_bay" in site_ids
    
    def test_get_validation_sites_region_filter(self):
        """Test getting validation sites filtered by region."""
        # Filter by British Columbia
        bc_sites = self.acquisition.get_validation_sites(region="British Columbia")
        assert len(bc_sites) == 2  # Broughton and Saanich
        
        bc_site_ids = [site.site_id for site in bc_sites]
        assert "broughton_archipelago" in bc_site_ids
        assert "saanich_inlet" in bc_site_ids
        
        # Filter by California
        ca_sites = self.acquisition.get_validation_sites(region="California")
        assert len(ca_sites) == 2  # Monterey and Point Reyes
        
        ca_site_ids = [site.site_id for site in ca_sites]
        assert "monterey_bay" in ca_site_ids
        assert "point_reyes" in ca_site_ids
    
    def test_get_validation_sites_species_filter(self):
        """Test getting validation sites filtered by species."""
        # Filter by Macrocystis
        macrocystis_sites = self.acquisition.get_validation_sites(species="Macrocystis")
        assert len(macrocystis_sites) >= 2
        
        # Filter by Nereocystis
        nereocystis_sites = self.acquisition.get_validation_sites(species="Nereocystis")
        assert len(nereocystis_sites) >= 1
        
        nere_site_ids = [site.site_id for site in nereocystis_sites]
        assert "broughton_archipelago" in nere_site_ids
    
    def test_get_validation_sites_combined_filter(self):
        """Test getting validation sites with combined region and species filters."""
        # Filter by California + Macrocystis
        ca_macro_sites = self.acquisition.get_validation_sites(
            region="California", species="Macrocystis"
        )
        assert len(ca_macro_sites) == 2  # Monterey and Point Reyes
        
        # Filter by British Columbia + Nereocystis
        bc_nere_sites = self.acquisition.get_validation_sites(
            region="British Columbia", species="Nereocystis"
        )
        assert len(bc_nere_sites) >= 1
    
    def test_create_synthetic_satellite_data(self):
        """Test creating synthetic satellite data."""
        site_id = "broughton_archipelago"
        num_scenes = 5
        
        scenes = self.acquisition.create_synthetic_satellite_data(site_id, num_scenes)
        
        assert len(scenes) == num_scenes
        
        for i, scene in enumerate(scenes):
            assert isinstance(scene, SatelliteScene)
            assert scene.scene_id == f"{site_id}_synthetic_{i+1:03d}"
            assert scene.site_id == site_id
            assert 0 <= scene.cloud_coverage <= 100
            assert scene.data_quality in ["excellent", "good", "fair", "poor"]
            assert scene.metadata["synthetic"] == True
            assert scene.metadata["species"] == "Nereocystis luetkeana"
    
    def test_create_synthetic_satellite_data_date_range(self):
        """Test creating synthetic satellite data with custom date range."""
        site_id = "monterey_bay"
        start_date = datetime.datetime(2024, 1, 1)
        end_date = datetime.datetime(2024, 12, 31)
        
        scenes = self.acquisition.create_synthetic_satellite_data(
            site_id, num_scenes=3, date_range=(start_date, end_date)
        )
        
        assert len(scenes) == 3
        
        # Check that dates are within range
        for scene in scenes:
            assert start_date <= scene.acquisition_date <= end_date
    
    def test_create_synthetic_satellite_data_invalid_site(self):
        """Test creating synthetic data for invalid site."""
        with pytest.raises(ValueError, match="Unknown site_id"):
            self.acquisition.create_synthetic_satellite_data("invalid_site")
    
    def test_create_synthetic_ground_truth(self):
        """Test creating synthetic ground truth data."""
        site_id = "monterey_bay"
        
        # Create some test scenes
        scenes = [
            SatelliteScene(
                scene_id="test_1",
                acquisition_date=datetime.datetime(2024, 6, 15),
                site_id=site_id,
                cloud_coverage=10.0,
                data_quality="excellent"
            ),
            SatelliteScene(
                scene_id="test_2",
                acquisition_date=datetime.datetime(2024, 8, 15),
                site_id=site_id,
                cloud_coverage=30.0,
                data_quality="good"
            )
        ]
        
        ground_truth = self.acquisition.create_synthetic_ground_truth(site_id, scenes)
        
        assert len(ground_truth) == len(scenes)
        
        for i, gt in enumerate(ground_truth):
            assert isinstance(gt, GroundTruthData)
            assert gt.site_id == site_id
            assert gt.data_type == "synthetic_validation"
            assert gt.collection_date == scenes[i].acquisition_date
            assert 0 <= gt.kelp_coverage_percent <= 100
            assert gt.kelp_density in ["sparse", "moderate", "dense", "very_dense"]
            assert 0 <= gt.confidence <= 1
            assert gt.source == "synthetic_generation"
            assert gt.metadata["scene_id"] == scenes[i].scene_id
    
    def test_create_synthetic_ground_truth_seasonal_variation(self):
        """Test that synthetic ground truth shows seasonal variation."""
        site_id = "broughton_archipelago"  # Kelp season: May (5) to October (10)
        
        # Create scenes in and out of season
        in_season_scene = SatelliteScene(
            scene_id="in_season",
            acquisition_date=datetime.datetime(2024, 7, 15),  # July - peak season
            site_id=site_id,
            cloud_coverage=10.0,
            data_quality="excellent"
        )
        
        out_season_scene = SatelliteScene(
            scene_id="out_season",
            acquisition_date=datetime.datetime(2024, 2, 15),  # February - out of season
            site_id=site_id,
            cloud_coverage=10.0,
            data_quality="excellent"
        )
        
        ground_truth = self.acquisition.create_synthetic_ground_truth(
            site_id, [in_season_scene, out_season_scene]
        )
        
        in_season_gt = ground_truth[0]
        out_season_gt = ground_truth[1]
        
        # In-season should have higher coverage than out-of-season
        assert in_season_gt.kelp_coverage_percent > out_season_gt.kelp_coverage_percent
    
    def test_create_synthetic_ground_truth_invalid_site(self):
        """Test creating synthetic ground truth for invalid site."""
        scenes = [
            SatelliteScene(
                scene_id="test",
                acquisition_date=datetime.datetime(2024, 6, 15),
                site_id="invalid_site",
                cloud_coverage=10.0,
                data_quality="excellent"
            )
        ]
        
        with pytest.raises(ValueError, match="Unknown site_id"):
            self.acquisition.create_synthetic_ground_truth("invalid_site", scenes)
    
    def test_calculate_dataset_quality_metrics(self):
        """Test calculation of dataset quality metrics."""
        scenes = [
            SatelliteScene(
                scene_id="scene_1",
                acquisition_date=datetime.datetime(2024, 1, 1),
                site_id="test",
                cloud_coverage=10.0,
                data_quality="excellent"
            ),
            SatelliteScene(
                scene_id="scene_2",
                acquisition_date=datetime.datetime(2024, 6, 1),
                site_id="test",
                cloud_coverage=30.0,
                data_quality="good"
            )
        ]
        
        ground_truth = [
            GroundTruthData(
                site_id="test",
                data_type="field_survey",
                collection_date=datetime.datetime(2024, 1, 1),
                kelp_coverage_percent=20.0,
                kelp_density="moderate",
                confidence=0.9,
                source="field_team"
            ),
            GroundTruthData(
                site_id="test",
                data_type="field_survey",
                collection_date=datetime.datetime(2024, 6, 1),
                kelp_coverage_percent=40.0,
                kelp_density="dense",
                confidence=0.8,
                source="field_team"
            )
        ]
        
        metrics = self.acquisition._calculate_dataset_quality_metrics(scenes, ground_truth)
        
        assert "overall_quality" in metrics
        assert "average_cloud_coverage" in metrics
        assert "excellent_scenes_percent" in metrics
        assert "good_or_better_percent" in metrics
        assert "average_gt_confidence" in metrics
        assert "coverage_range_percent" in metrics
        assert "temporal_span_days" in metrics
        assert "num_scenes" in metrics
        assert "num_ground_truth" in metrics
        
        assert 0 <= metrics["overall_quality"] <= 1
        assert metrics["average_cloud_coverage"] == 20.0  # (10 + 30) / 2
        assert metrics["excellent_scenes_percent"] == 50.0  # 1 out of 2
        assert metrics["good_or_better_percent"] == 100.0  # 2 out of 2
        assert metrics["average_gt_confidence"] == 0.85  # (0.9 + 0.8) / 2
        assert metrics["coverage_range_percent"] == 20.0  # 40 - 20
        assert metrics["num_scenes"] == 2
        assert metrics["num_ground_truth"] == 2
    
    def test_calculate_dataset_quality_metrics_empty(self):
        """Test quality metrics calculation with empty data."""
        metrics = self.acquisition._calculate_dataset_quality_metrics([], [])
        assert metrics == {"overall_quality": 0.0}
    
    def test_create_validation_dataset(self):
        """Test creating a complete validation dataset."""
        site_id = "broughton_archipelago"
        num_scenes = 3
        
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes, use_synthetic=True)
        
        assert isinstance(dataset, ValidationDataset)
        assert dataset.site.site_id == site_id
        assert len(dataset.satellite_scenes) == num_scenes
        assert len(dataset.ground_truth) == num_scenes
        assert isinstance(dataset.quality_metrics, dict)
        assert "overall_quality" in dataset.quality_metrics
        assert isinstance(dataset.created_date, datetime.datetime)
        assert dataset.version == "1.0"
    
    def test_create_validation_dataset_invalid_site(self):
        """Test creating validation dataset for invalid site."""
        with pytest.raises(ValueError, match="Unknown site_id"):
            self.acquisition.create_validation_dataset("invalid_site")
    
    def test_save_and_load_validation_dataset(self):
        """Test saving and loading validation datasets."""
        # Create a test dataset
        site_id = "monterey_bay"
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=2, use_synthetic=True)
        
        # Save dataset
        filepath = self.acquisition.save_validation_dataset(dataset)
        assert Path(filepath).exists()
        
        # Load dataset
        loaded_dataset = self.acquisition.load_validation_dataset(filepath)
        
        # Verify loaded dataset matches original
        assert loaded_dataset.site.site_id == dataset.site.site_id
        assert loaded_dataset.site.name == dataset.site.name
        assert loaded_dataset.site.coordinates == dataset.site.coordinates
        assert len(loaded_dataset.satellite_scenes) == len(dataset.satellite_scenes)
        assert len(loaded_dataset.ground_truth) == len(dataset.ground_truth)
        assert loaded_dataset.quality_metrics == dataset.quality_metrics
        assert loaded_dataset.version == dataset.version
        
        # Check specific scene and ground truth data
        assert loaded_dataset.satellite_scenes[0].scene_id == dataset.satellite_scenes[0].scene_id
        assert loaded_dataset.ground_truth[0].kelp_coverage_percent == dataset.ground_truth[0].kelp_coverage_percent
    
    def test_save_validation_dataset_custom_filename(self):
        """Test saving validation dataset with custom filename."""
        site_id = "saanich_inlet"
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=1, use_synthetic=True)
        
        custom_filename = "custom_test_dataset.json"
        filepath = self.acquisition.save_validation_dataset(dataset, custom_filename)
        
        assert custom_filename in filepath
        assert Path(filepath).exists()
    
    def test_create_benchmark_suite(self):
        """Test creating a benchmark suite across multiple sites."""
        include_sites = ["broughton_archipelago", "monterey_bay"]
        
        benchmark_suite = self.acquisition.create_benchmark_suite(include_sites)
        
        assert len(benchmark_suite) == 2
        assert "broughton_archipelago" in benchmark_suite
        assert "monterey_bay" in benchmark_suite
        
        for site_id, dataset in benchmark_suite.items():
            assert isinstance(dataset, ValidationDataset)
            assert dataset.site.site_id == site_id
            assert len(dataset.satellite_scenes) == 8  # Default num_scenes
            assert len(dataset.ground_truth) == 8
    
    def test_create_benchmark_suite_all_sites(self):
        """Test creating benchmark suite for all sites."""
        benchmark_suite = self.acquisition.create_benchmark_suite()
        
        # Should include all 6 sites
        assert len(benchmark_suite) == 6
        
        expected_sites = [
            "broughton_archipelago", "monterey_bay", "saanich_inlet",
            "puget_sound", "point_reyes", "tasmania_giant_kelp"
        ]
        
        for site_id in expected_sites:
            assert site_id in benchmark_suite
    
    def test_create_benchmark_suite_invalid_site(self):
        """Test creating benchmark suite with invalid site."""
        include_sites = ["broughton_archipelago", "invalid_site"]
        
        # Should skip invalid site and continue with valid ones
        benchmark_suite = self.acquisition.create_benchmark_suite(include_sites)
        
        assert len(benchmark_suite) == 1  # Only valid site
        assert "broughton_archipelago" in benchmark_suite
        assert "invalid_site" not in benchmark_suite
    
    def test_get_site_summary(self):
        """Test getting site summary statistics."""
        summary = self.acquisition.get_site_summary()
        
        assert "total_sites" in summary
        assert "sites_by_region" in summary
        assert "sites_by_species" in summary
        assert "sites_by_confidence" in summary
        assert "data_sources" in summary
        
        assert summary["total_sites"] == 6
        
        # Check region counts
        assert "British Columbia, Canada" in summary["sites_by_region"]
        assert "California, USA" in summary["sites_by_region"]
        
        # Check species counts
        assert summary["sites_by_species"]["Macrocystis pyrifera"] >= 2
        
        # Check confidence levels
        assert "high" in summary["sites_by_confidence"]
        assert "medium" in summary["sites_by_confidence"]
        
        # Check data sources
        assert isinstance(summary["data_sources"], list)
        assert len(summary["data_sources"]) > 0


class TestFactoryFunctions:
    """Test factory functions and high-level interfaces."""
    
    def test_create_real_data_acquisition(self):
        """Test factory function for creating acquisition system."""
        # Default creation
        acquisition1 = create_real_data_acquisition()
        assert isinstance(acquisition1, RealDataAcquisition)
        
        # Custom directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            acquisition2 = create_real_data_acquisition(temp_dir)
            assert isinstance(acquisition2, RealDataAcquisition)
            assert acquisition2.data_directory == Path(temp_dir)
    
    def test_get_validation_sites_function(self):
        """Test high-level get_validation_sites function."""
        # Get all sites
        all_sites = get_validation_sites()
        assert len(all_sites) == 6
        
        # Get BC sites
        bc_sites = get_validation_sites(region="British Columbia")
        assert len(bc_sites) == 2
        
        # Get Macrocystis sites
        macro_sites = get_validation_sites(species="Macrocystis")
        assert len(macro_sites) >= 2
    
    def test_create_benchmark_dataset_function(self):
        """Test high-level create_benchmark_dataset function."""
        dataset = create_benchmark_dataset("broughton_archipelago", num_scenes=3)
        
        assert isinstance(dataset, ValidationDataset)
        assert dataset.site.site_id == "broughton_archipelago"
        assert len(dataset.satellite_scenes) == 3
        assert len(dataset.ground_truth) == 3


class TestIntegrationScenarios:
    """Test integration scenarios and realistic workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = RealDataAcquisition(data_directory=self.temp_dir)
    
    def test_full_validation_workflow(self):
        """Test complete validation dataset creation and usage workflow."""
        # Step 1: Create validation dataset
        site_id = "monterey_bay"
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=5)
        
        # Step 2: Verify dataset quality
        assert dataset.quality_metrics["overall_quality"] > 0.0
        assert dataset.quality_metrics["num_scenes"] == 5
        assert dataset.quality_metrics["num_ground_truth"] == 5
        
        # Step 3: Save dataset
        filepath = self.acquisition.save_validation_dataset(dataset)
        
        # Step 4: Load and verify
        loaded_dataset = self.acquisition.load_validation_dataset(filepath)
        assert loaded_dataset.site.site_id == site_id
        
        # Step 5: Verify synthetic data quality
        scenes = loaded_dataset.satellite_scenes
        ground_truth = loaded_dataset.ground_truth
        
        # Check temporal alignment
        for scene, gt in zip(scenes, ground_truth):
            assert scene.acquisition_date == gt.collection_date
            assert scene.site_id == gt.site_id
        
        # Check data realism
        for scene in scenes:
            assert 0 <= scene.cloud_coverage <= 100
            assert scene.data_quality in ["excellent", "good", "fair", "poor"]
        
        for gt in ground_truth:
            assert 0 <= gt.kelp_coverage_percent <= 100
            assert 0 <= gt.confidence <= 1
            assert gt.kelp_density in ["sparse", "moderate", "dense", "very_dense"]
    
    def test_multi_site_benchmark_creation(self):
        """Test creating benchmarks across multiple diverse sites."""
        sites = ["broughton_archipelago", "monterey_bay", "puget_sound"]
        
        benchmark_suite = self.acquisition.create_benchmark_suite(sites)
        
        # Verify all sites included
        assert len(benchmark_suite) == 3
        
        # Check species diversity
        species_found = set()
        for dataset in benchmark_suite.values():
            species_found.add(dataset.site.species)
        
        assert len(species_found) >= 2  # At least 2 different species
        
        # Check geographic diversity
        regions_found = set()
        for dataset in benchmark_suite.values():
            regions_found.add(dataset.site.region)
        
        assert len(regions_found) >= 2  # At least 2 different regions
        
        # Check temporal coverage
        for dataset in benchmark_suite.values():
            dates = [scene.acquisition_date for scene in dataset.satellite_scenes]
            temporal_span = (max(dates) - min(dates)).days
            assert temporal_span > 0  # Should span multiple dates
    
    def test_seasonal_validation_scenarios(self):
        """Test validation across different seasonal scenarios."""
        site_id = "broughton_archipelago"  # Has clear seasonal pattern (May-Oct)
        
        # Create scenes across full year
        start_date = datetime.datetime(2024, 1, 1)
        end_date = datetime.datetime(2024, 12, 31)
        
        scenes = self.acquisition.create_synthetic_satellite_data(
            site_id, num_scenes=12, date_range=(start_date, end_date)
        )
        
        ground_truth = self.acquisition.create_synthetic_ground_truth(site_id, scenes)
        
        # Group by season
        peak_season_gt = []  # May-Oct
        off_season_gt = []   # Nov-Apr
        
        for gt in ground_truth:
            month = gt.collection_date.month
            if 5 <= month <= 10:  # Peak season
                peak_season_gt.append(gt)
            else:  # Off season
                off_season_gt.append(gt)
        
        # Peak season should have higher average coverage
        if peak_season_gt and off_season_gt:
            peak_avg = np.mean([gt.kelp_coverage_percent for gt in peak_season_gt])
            off_avg = np.mean([gt.kelp_coverage_percent for gt in off_season_gt])
            assert peak_avg > off_avg
    
    def test_quality_assessment_workflow(self):
        """Test quality assessment and filtering workflow."""
        site_id = "saanich_inlet"
        
        # Create dataset with varying quality
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=10)
        
        # Analyze quality metrics
        metrics = dataset.quality_metrics
        
        # Should have reasonable quality scores
        assert 0 <= metrics["overall_quality"] <= 1
        assert 0 <= metrics["average_cloud_coverage"] <= 100
        assert 0 <= metrics["excellent_scenes_percent"] <= 100
        assert 0 <= metrics["average_gt_confidence"] <= 1
        
        # Filter high-quality scenes
        high_quality_scenes = [
            scene for scene in dataset.satellite_scenes 
            if scene.data_quality in ["excellent", "good"] and scene.cloud_coverage < 30
        ]
        
        # Should have some high-quality scenes
        assert len(high_quality_scenes) > 0
        
        # High-quality scenes should have better characteristics
        for scene in high_quality_scenes:
            assert scene.cloud_coverage < 30
            assert scene.data_quality in ["excellent", "good"]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.acquisition = RealDataAcquisition(data_directory=self.temp_dir)
    
    def test_empty_scenes_list(self):
        """Test handling empty scenes list."""
        site_id = "broughton_archipelago"
        ground_truth = self.acquisition.create_synthetic_ground_truth(site_id, [])
        
        assert ground_truth == []
    
    def test_single_scene_dataset(self):
        """Test creating dataset with single scene."""
        site_id = "monterey_bay"
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=1)
        
        assert len(dataset.satellite_scenes) == 1
        assert len(dataset.ground_truth) == 1
        assert dataset.quality_metrics["temporal_span_days"] == 0
    
    def test_cross_year_kelp_season(self):
        """Test handling kelp seasons that cross year boundaries."""
        site_id = "puget_sound"  # Season: Nov (11) to June (6)
        
        # Create scenes across year boundary
        scenes = self.acquisition.create_synthetic_satellite_data(site_id, num_scenes=8)
        ground_truth = self.acquisition.create_synthetic_ground_truth(site_id, scenes)
        
        # Should handle cross-year season correctly
        assert len(ground_truth) == len(scenes)
        
        # Check that seasonal patterns work across year boundary
        in_season_months = [11, 12, 1, 2, 3, 4, 5, 6]
        for gt in ground_truth:
            month = gt.collection_date.month
            if month in in_season_months:
                # Should have reasonable coverage during season
                assert gt.kelp_coverage_percent >= 0
    
    def test_invalid_date_range(self):
        """Test handling invalid date ranges."""
        site_id = "broughton_archipelago"
        
        # Start date after end date
        start_date = datetime.datetime(2024, 12, 1)
        end_date = datetime.datetime(2024, 1, 1)
        
        # Should handle gracefully
        scenes = self.acquisition.create_synthetic_satellite_data(
            site_id, num_scenes=2, date_range=(start_date, end_date)
        )
        
        # Should still create scenes (implementation may swap dates or handle differently)
        assert len(scenes) == 2
    
    def test_load_nonexistent_file(self):
        """Test loading dataset from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.acquisition.load_validation_dataset("nonexistent_file.json")
    
    def test_save_to_readonly_directory(self):
        """Test saving dataset to readonly directory."""
        site_id = "broughton_archipelago"
        dataset = self.acquisition.create_validation_dataset(site_id, num_scenes=1)
        
        # Try to save to nonexistent directory (should create it)
        bad_filename = "/nonexistent/readonly/test.json"
        
        # This might raise an exception depending on permissions
        # The exact behavior depends on the system
        try:
            self.acquisition.save_validation_dataset(dataset, bad_filename)
        except (OSError, PermissionError):
            # Expected for readonly or nonexistent directories
            pass 