"""
Unit tests for Field Survey Data Integration - Task C2.4

Tests field data ingestion, species validation analysis, and reporting functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, mock_open
from tempfile import NamedTemporaryFile

from src.kelpie_carbon_v1.validation.field_survey_integration import (
    FieldSurveyRecord,
    SpeciesValidationMetrics,
    FieldDataIngestor,
    SpeciesValidationAnalyzer,
    FieldSurveyReporter,
    create_field_data_ingestor,
    create_validation_analyzer,
    create_survey_reporter
)


class TestFieldSurveyRecord:
    """Test FieldSurveyRecord dataclass."""

    def test_field_survey_record_creation(self):
        """Test creating a field survey record."""
        record = FieldSurveyRecord(
            record_id="test_001",
            site_name="Test Site",
            timestamp=datetime.now(),
            lat=48.5,
            lng=-123.5,
            depth_m=10.0,
            observed_species=["nereocystis_luetkeana"],
            primary_species="nereocystis_luetkeana",
            species_confidence=0.85,
            biomass_kg_per_m2=8.5
        )
        
        assert record.record_id == "test_001"
        assert record.primary_species == "nereocystis_luetkeana"
        assert record.biomass_kg_per_m2 == 8.5
        assert record.species_confidence == 0.85

    def test_field_survey_record_defaults(self):
        """Test field survey record with default values."""
        record = FieldSurveyRecord(
            record_id="test_002",
            site_name="Test Site",
            timestamp=datetime.now(),
            lat=48.5,
            lng=-123.5,
            depth_m=10.0,
            observed_species=["unknown"],
            primary_species="unknown",
            species_confidence=0.5
        )
        
        assert record.biomass_measurement_method == "visual_estimate"
        assert record.biomass_confidence == "moderate"
        assert record.canopy_type == "surface"
        assert record.kelp_density == "moderate"
        assert len(record.equipment_used) == 0
        assert len(record.photo_references) == 0


class TestSpeciesValidationMetrics:
    """Test SpeciesValidationMetrics dataclass."""

    def test_validation_metrics_creation(self):
        """Test creating validation metrics."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={"nereocystis_luetkeana": 0.9},
            species_recall={"nereocystis_luetkeana": 0.8},
            species_f1_score={"nereocystis_luetkeana": 0.85},
            confusion_matrix={"nereocystis_luetkeana": {"nereocystis_luetkeana": 10}},
            biomass_mae=1.5,
            biomass_rmse=2.0,
            biomass_r2=0.8,
            biomass_accuracy_by_species={"nereocystis_luetkeana": {"mae": 1.2, "rmse": 1.8}},
            total_samples=25,
            samples_by_species={"nereocystis_luetkeana": 20},
            classification_confidence_distribution={"high": 15, "medium": 8, "low": 2},
            biomass_confidence_distribution={"high": 10, "moderate": 12, "low": 3}
        )
        
        assert metrics.species_accuracy == 0.85
        assert metrics.biomass_mae == 1.5
        assert metrics.total_samples == 25
        assert "nereocystis_luetkeana" in metrics.species_precision


class TestFieldDataIngestor:
    """Test FieldDataIngestor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ingestor = FieldDataIngestor()

    def test_ingestor_initialization(self):
        """Test ingestor initialization."""
        assert 'csv' in self.ingestor.supported_formats
        assert 'json' in self.ingestor.supported_formats
        assert 'excel' in self.ingestor.supported_formats

    def test_parse_species_list_from_string(self):
        """Test parsing species list from string."""
        result = self.ingestor._parse_species_list("nereocystis_luetkeana, macrocystis_pyrifera")
        assert len(result) == 2
        assert "nereocystis_luetkeana" in result
        assert "macrocystis_pyrifera" in result

    def test_parse_species_list_from_list(self):
        """Test parsing species list from list."""
        input_list = ["nereocystis_luetkeana", "macrocystis_pyrifera"]
        result = self.ingestor._parse_species_list(input_list)
        assert result == ["nereocystis_luetkeana", "macrocystis_pyrifera"]

    def test_parse_empty_species_list(self):
        """Test parsing empty species list."""
        result = self.ingestor._parse_species_list("")
        assert result == ['unknown']

    def test_convert_dict_to_record_success(self):
        """Test successful conversion of dict to record."""
        data = {
            'record_id': 'test_001',
            'site_name': 'Test Site',
            'timestamp': '2024-01-15 10:00:00',
            'lat': 48.5,
            'lng': -123.5,
            'depth_m': 10.0,
            'observed_species': ['nereocystis_luetkeana'],
            'primary_species': 'nereocystis_luetkeana',
            'species_confidence': 0.85,
            'biomass_kg_per_m2': 8.5
        }
        
        record = self.ingestor._convert_dict_to_record(data)
        
        assert record is not None
        assert record.record_id == 'test_001'
        assert record.primary_species == 'nereocystis_luetkeana'
        assert record.biomass_kg_per_m2 == 8.5

    def test_convert_dict_to_record_with_missing_fields(self):
        """Test conversion with missing optional fields."""
        data = {
            'lat': 48.5,
            'lng': -123.5,
            'depth_m': 10.0,
            'primary_species': 'nereocystis_luetkeana'
        }
        
        record = self.ingestor._convert_dict_to_record(data)
        
        assert record is not None
        assert record.primary_species == 'nereocystis_luetkeana'
        assert record.site_name == 'unknown'
        assert record.species_confidence == 0.5

    def test_convert_dict_to_record_with_invalid_data(self):
        """Test conversion with invalid data."""
        data = {
            'lat': 'invalid',  # Invalid latitude
            'lng': -123.5,
            'depth_m': 10.0
        }
        
        record = self.ingestor._convert_dict_to_record(data)
        assert record is None

    @patch('pandas.read_csv')
    def test_ingest_csv_survey_success(self, mock_read_csv):
        """Test successful CSV ingestion."""
        # Mock CSV data
        mock_df = pd.DataFrame({
            'record_id': ['001', '002'],
            'site_name': ['Site A', 'Site B'],
            'timestamp': ['2024-01-15 10:00:00', '2024-01-15 11:00:00'],
            'lat': [48.5, 48.6],
            'lng': [-123.5, -123.6],
            'depth_m': [10.0, 12.0],
            'primary_species': ['nereocystis_luetkeana', 'macrocystis_pyrifera'],
            'species_confidence': [0.85, 0.75],
            'biomass_kg_per_m2': [8.5, 12.0]
        })
        mock_read_csv.return_value = mock_df
        
        records = self.ingestor.ingest_csv_survey(Path('test.csv'))
        
        assert len(records) == 2
        assert records[0].primary_species == 'nereocystis_luetkeana'
        assert records[1].primary_species == 'macrocystis_pyrifera'

    @patch('pandas.read_csv')
    def test_ingest_csv_survey_error(self, mock_read_csv):
        """Test CSV ingestion with error."""
        mock_read_csv.side_effect = Exception("File not found")
        
        records = self.ingestor.ingest_csv_survey(Path('nonexistent.csv'))
        
        assert len(records) == 0

    def test_ingest_json_survey_success(self):
        """Test successful JSON ingestion."""
        json_data = [
            {
                'record_id': '001',
                'site_name': 'Site A',
                'timestamp': '2024-01-15 10:00:00',
                'lat': 48.5,
                'lng': -123.5,
                'depth_m': 10.0,
                'primary_species': 'nereocystis_luetkeana',
                'species_confidence': 0.85,
                'biomass_kg_per_m2': 8.5
            }
        ]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(json_data))):
            records = self.ingestor.ingest_json_survey(Path('test.json'))
        
        assert len(records) == 1
        assert records[0].primary_species == 'nereocystis_luetkeana'
        assert records[0].biomass_kg_per_m2 == 8.5

    def test_ingest_json_survey_error(self):
        """Test JSON ingestion with error."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            records = self.ingestor.ingest_json_survey(Path('nonexistent.json'))
        
        assert len(records) == 0


class TestSpeciesValidationAnalyzer:
    """Test SpeciesValidationAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SpeciesValidationAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.logger is not None

    def test_create_empty_metrics(self):
        """Test creating empty metrics."""
        metrics = self.analyzer._create_empty_metrics()
        
        assert metrics.species_accuracy == 0.0
        assert metrics.biomass_mae == 0.0
        assert metrics.total_samples == 0
        assert len(metrics.species_precision) == 0

    def test_calculate_species_classification_metrics(self):
        """Test species classification metrics calculation."""
        # Create mock matched pairs
        predictions = [
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.9},
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.8},
            {'primary_species': 'macrocystis_pyrifera', 'confidence': 0.7}
        ]
        
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime.now(),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85
            ),
            FieldSurveyRecord(
                record_id="002", site_name="Site B", timestamp=datetime.now(),
                lat=48.6, lng=-123.6, depth_m=12.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.75
            ),
            FieldSurveyRecord(
                record_id="003", site_name="Site C", timestamp=datetime.now(),
                lat=48.7, lng=-123.7, depth_m=8.0,
                observed_species=["macrocystis_pyrifera"], 
                primary_species="macrocystis_pyrifera", species_confidence=0.80
            )
        ]
        
        matched_pairs = list(zip(predictions, field_records))
        metrics = self.analyzer._calculate_species_classification_metrics(matched_pairs)
        
        assert metrics['accuracy'] == 1.0  # All predictions correct
        assert 'nereocystis_luetkeana' in metrics['precision']
        assert 'macrocystis_pyrifera' in metrics['precision']
        assert metrics['confidence_dist']['high'] == 2  # Two predictions >= 0.8
        assert metrics['confidence_dist']['medium'] == 1  # One prediction 0.5-0.8

    def test_calculate_biomass_estimation_metrics(self):
        """Test biomass estimation metrics calculation."""
        # Create mock matched pairs with biomass data
        predictions = [
            {'biomass_estimate_kg_per_m2': 8.0},
            {'biomass_estimate_kg_per_m2': 9.5},
            {'biomass_estimate_kg_per_m2': 12.0}
        ]
        
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime.now(),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85,
                biomass_kg_per_m2=8.5, biomass_confidence="high"
            ),
            FieldSurveyRecord(
                record_id="002", site_name="Site B", timestamp=datetime.now(),
                lat=48.6, lng=-123.6, depth_m=12.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.75,
                biomass_kg_per_m2=10.0, biomass_confidence="moderate"
            ),
            FieldSurveyRecord(
                record_id="003", site_name="Site C", timestamp=datetime.now(),
                lat=48.7, lng=-123.7, depth_m=8.0,
                observed_species=["macrocystis_pyrifera"], 
                primary_species="macrocystis_pyrifera", species_confidence=0.80,
                biomass_kg_per_m2=11.5, biomass_confidence="high"
            )
        ]
        
        matched_pairs = list(zip(predictions, field_records))
        metrics = self.analyzer._calculate_biomass_estimation_metrics(matched_pairs)
        
        assert metrics['mae'] > 0  # Should have some error
        assert metrics['rmse'] > 0
        assert -1 <= metrics['r2'] <= 1  # R² should be in valid range
        assert metrics['confidence_dist']['high'] == 2
        assert metrics['confidence_dist']['moderate'] == 1

    def test_compare_predictions_to_field_data_empty(self):
        """Test comparison with empty data."""
        metrics = self.analyzer.compare_predictions_to_field_data([], [])
        
        assert metrics.total_samples == 0
        assert metrics.species_accuracy == 0.0
        assert metrics.biomass_mae == 0.0

    def test_compare_predictions_to_field_data_success(self):
        """Test successful comparison of predictions to field data."""
        predictions = [
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.9, 'biomass_estimate_kg_per_m2': 8.0}
        ]
        
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime.now(),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85,
                biomass_kg_per_m2=8.5
            )
        ]
        
        metrics = self.analyzer.compare_predictions_to_field_data(predictions, field_records)
        
        assert metrics.total_samples == 1
        assert metrics.species_accuracy == 1.0


class TestFieldSurveyReporter:
    """Test FieldSurveyReporter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = FieldSurveyReporter()

    def test_reporter_initialization(self):
        """Test reporter initialization."""
        assert self.reporter.logger is not None

    def test_assess_data_quality_excellent(self):
        """Test data quality assessment - excellent."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={}, species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=1.0, biomass_rmse=1.5, biomass_r2=0.8,
            biomass_accuracy_by_species={}, total_samples=60, samples_by_species={},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        quality = self.reporter._assess_data_quality(metrics)
        assert quality == "Excellent"

    def test_assess_data_quality_limited(self):
        """Test data quality assessment - limited."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.45,
            species_precision={}, species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=1.0, biomass_rmse=1.5, biomass_r2=0.8,
            biomass_accuracy_by_species={}, total_samples=5, samples_by_species={},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        quality = self.reporter._assess_data_quality(metrics)
        assert quality == "Limited"

    def test_assess_biomass_performance_excellent(self):
        """Test biomass performance assessment - excellent."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={}, species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=1.5, biomass_rmse=1.8, biomass_r2=0.85,
            biomass_accuracy_by_species={}, total_samples=50, samples_by_species={},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        performance = self.reporter._assess_biomass_performance(metrics)
        assert performance == "Excellent"

    def test_assess_biomass_performance_poor(self):
        """Test biomass performance assessment - poor."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={}, species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=8.0, biomass_rmse=10.0, biomass_r2=0.2,
            biomass_accuracy_by_species={}, total_samples=50, samples_by_species={},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        performance = self.reporter._assess_biomass_performance(metrics)
        assert performance == "Poor"

    def test_generate_recommendations_good_performance(self):
        """Test recommendations for good performance."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={}, species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=2.0, biomass_rmse=2.5, biomass_r2=0.8,
            biomass_accuracy_by_species={}, total_samples=50, samples_by_species={},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        recommendations = self.reporter._generate_recommendations(metrics)
        assert len(recommendations) == 1
        assert "quality standards" in recommendations[0]

    def test_generate_recommendations_poor_performance(self):
        """Test recommendations for poor performance."""
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.45,  # Below 70%
            species_precision={'nereocystis_luetkeana': 0.5}, 
            species_recall={}, species_f1_score={},
            confusion_matrix={}, biomass_mae=8.0, biomass_rmse=10.0, biomass_r2=0.2,
            biomass_accuracy_by_species={}, total_samples=50, 
            samples_by_species={'nereocystis_luetkeana': 10},
            classification_confidence_distribution={}, biomass_confidence_distribution={}
        )
        
        recommendations = self.reporter._generate_recommendations(metrics)
        assert len(recommendations) >= 3  # Should have multiple recommendations
        assert any("accuracy is below 70%" in rec for rec in recommendations)
        assert any("error is high" in rec for rec in recommendations)
        assert any("Low correlation" in rec for rec in recommendations)

    def test_generate_field_survey_summary_empty(self):
        """Test field survey summary with empty data."""
        summary = self.reporter._generate_field_survey_summary([])
        assert summary['status'] == 'No field records available'

    def test_generate_field_survey_summary_with_data(self):
        """Test field survey summary with data."""
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime(2024, 1, 15, 10, 0),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85,
                biomass_kg_per_m2=8.5
            ),
            FieldSurveyRecord(
                record_id="002", site_name="Site B", timestamp=datetime(2024, 1, 16, 11, 0),
                lat=48.6, lng=-123.6, depth_m=12.0,
                observed_species=["macrocystis_pyrifera"], 
                primary_species="macrocystis_pyrifera", species_confidence=0.75,
                biomass_kg_per_m2=12.0
            )
        ]
        
        summary = self.reporter._generate_field_survey_summary(field_records)
        
        assert summary['total_records'] == 2
        assert len(summary['sites_surveyed']) == 2
        assert 'nereocystis_luetkeana' in summary['species_distribution']
        assert 'macrocystis_pyrifera' in summary['species_distribution']
        assert summary['biomass_statistics']['measurements_count'] == 2
        assert summary['biomass_statistics']['mean_kg_per_m2'] == 10.25

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create test data
        metrics = SpeciesValidationMetrics(
            species_accuracy=0.85,
            species_precision={'nereocystis_luetkeana': 0.9}, 
            species_recall={'nereocystis_luetkeana': 0.8}, 
            species_f1_score={'nereocystis_luetkeana': 0.85},
            confusion_matrix={'nereocystis_luetkeana': {'nereocystis_luetkeana': 10}}, 
            biomass_mae=2.0, biomass_rmse=2.5, biomass_r2=0.8,
            biomass_accuracy_by_species={'nereocystis_luetkeana': {'mae': 1.8, 'rmse': 2.2}}, 
            total_samples=20, 
            samples_by_species={'nereocystis_luetkeana': 18},
            classification_confidence_distribution={'high': 15, 'medium': 3, 'low': 2}, 
            biomass_confidence_distribution={'high': 8, 'moderate': 10, 'low': 2}
        )
        
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime.now(),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85,
                biomass_kg_per_m2=8.5
            )
        ]
        
        predictions = [
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.9, 'biomass_estimate_kg_per_m2': 8.0}
        ]
        
        report = self.reporter.generate_comprehensive_report(
            "test_campaign", metrics, field_records, predictions
        )
        
        assert report['campaign_id'] == "test_campaign"
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'species_classification' in report
        assert 'biomass_estimation' in report
        assert 'field_survey_summary' in report
        assert 'model_performance' in report
        assert 'recommendations' in report


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_field_data_ingestor(self):
        """Test field data ingestor factory function."""
        ingestor = create_field_data_ingestor()
        assert isinstance(ingestor, FieldDataIngestor)
        assert 'csv' in ingestor.supported_formats

    def test_create_validation_analyzer(self):
        """Test validation analyzer factory function."""
        analyzer = create_validation_analyzer()
        assert isinstance(analyzer, SpeciesValidationAnalyzer)

    def test_create_survey_reporter(self):
        """Test survey reporter factory function."""
        reporter = create_survey_reporter()
        assert isinstance(reporter, FieldSurveyReporter)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_end_to_end_field_survey_processing(self):
        """Test end-to-end field survey data processing."""
        # Create components
        ingestor = create_field_data_ingestor()
        analyzer = create_validation_analyzer()
        reporter = create_survey_reporter()
        
        # Create test data
        test_data = {
            'record_id': '001',
            'site_name': 'Test Site',
            'timestamp': '2024-01-15 10:00:00',
            'lat': 48.5,
            'lng': -123.5,
            'depth_m': 10.0,
            'primary_species': 'nereocystis_luetkeana',
            'species_confidence': 0.85,
            'biomass_kg_per_m2': 8.5
        }
        
        # Convert to field record
        field_record = ingestor._convert_dict_to_record(test_data)
        assert field_record is not None
        
        # Create mock prediction
        prediction = {
            'primary_species': 'nereocystis_luetkeana',
            'confidence': 0.9,
            'biomass_estimate_kg_per_m2': 8.0
        }
        
        # Analyze validation
        metrics = analyzer.compare_predictions_to_field_data([prediction], [field_record])
        assert metrics.total_samples == 1
        assert metrics.species_accuracy == 1.0
        
        # Generate report
        report = reporter.generate_comprehensive_report(
            "integration_test", metrics, [field_record], [prediction]
        )
        assert report['campaign_id'] == "integration_test"
        assert report['summary']['overall_species_accuracy'] == "100.0%"

    def test_mixed_species_validation_scenario(self):
        """Test validation scenario with mixed species."""
        analyzer = create_validation_analyzer()
        
        # Create mixed species predictions and field data
        predictions = [
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.9, 'biomass_estimate_kg_per_m2': 8.0},
            {'primary_species': 'macrocystis_pyrifera', 'confidence': 0.8, 'biomass_estimate_kg_per_m2': 12.0},
            {'primary_species': 'nereocystis_luetkeana', 'confidence': 0.7, 'biomass_estimate_kg_per_m2': 9.0}
        ]
        
        field_records = [
            FieldSurveyRecord(
                record_id="001", site_name="Site A", timestamp=datetime.now(),
                lat=48.5, lng=-123.5, depth_m=10.0,
                observed_species=["nereocystis_luetkeana"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.85,
                biomass_kg_per_m2=8.5
            ),
            FieldSurveyRecord(
                record_id="002", site_name="Site B", timestamp=datetime.now(),
                lat=48.6, lng=-123.6, depth_m=12.0,
                observed_species=["macrocystis_pyrifera"], 
                primary_species="macrocystis_pyrifera", species_confidence=0.80,
                biomass_kg_per_m2=11.5
            ),
            FieldSurveyRecord(
                record_id="003", site_name="Site C", timestamp=datetime.now(),
                lat=48.7, lng=-123.7, depth_m=8.0,
                observed_species=["mixed_species"], 
                primary_species="nereocystis_luetkeana", species_confidence=0.75,
                biomass_kg_per_m2=9.2
            )
        ]
        
        metrics = analyzer.compare_predictions_to_field_data(predictions, field_records)
        
        # Should detect both species
        assert len(metrics.samples_by_species) >= 2
        assert metrics.total_samples == 3
        
        # Both species should have precision/recall metrics
        assert 'nereocystis_luetkeana' in metrics.species_precision
        assert 'macrocystis_pyrifera' in metrics.species_precision

    def test_performance_degradation_scenario(self):
        """Test scenario with performance degradation patterns."""
        reporter = create_survey_reporter()
        
        # Create metrics showing poor performance
        poor_metrics = SpeciesValidationMetrics(
            species_accuracy=0.45,  # Poor accuracy
            species_precision={'nereocystis_luetkeana': 0.4, 'macrocystis_pyrifera': 0.3}, 
            species_recall={'nereocystis_luetkeana': 0.5, 'macrocystis_pyrifera': 0.4}, 
            species_f1_score={'nereocystis_luetkeana': 0.44, 'macrocystis_pyrifera': 0.34},
            confusion_matrix={}, 
            biomass_mae=8.5,  # High error
            biomass_rmse=12.0, 
            biomass_r2=0.15,  # Poor correlation
            biomass_accuracy_by_species={}, 
            total_samples=30, 
            samples_by_species={'nereocystis_luetkeana': 20, 'macrocystis_pyrifera': 10},
            classification_confidence_distribution={'high': 5, 'medium': 10, 'low': 15}, 
            biomass_confidence_distribution={'high': 2, 'moderate': 8, 'low': 20}
        )
        
        recommendations = reporter._generate_recommendations(poor_metrics)
        
        # Should generate multiple recommendations for improvement
        assert len(recommendations) >= 3
        assert any("accuracy is below 70%" in rec for rec in recommendations)
        assert any("error is high" in rec for rec in recommendations)
        assert any("Low correlation" in rec for rec in recommendations)
        assert any("Low precision for nereocystis_luetkeana" in rec for rec in recommendations)
        assert any("Low precision for macrocystis_pyrifera" in rec for rec in recommendations) 