"""
Field Survey Data Integration - Task C2.4

This module implements comprehensive field survey data integration for:
- Species-specific ground truth validation
- Biomass estimation accuracy assessment  
- Enhanced reporting with species classification metrics
- Field data ingestion pipeline for multiple data formats

Integrates with Task C2.1-C2.3 species classification and biomass estimation.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.processing.species_classifier import (
    SpeciesClassificationResult,
    SpeciesClassifier,
)
from .data_manager import (
    GroundTruthMeasurement,
    ValidationDataManager,
)

logger = logging.getLogger(__name__)


@dataclass
class FieldSurveyRecord:
    """Individual field survey record with species and biomass data."""
    
    record_id: str
    site_name: str
    timestamp: datetime
    lat: float
    lng: float
    depth_m: float
    
    # Species information
    observed_species: list[str]
    primary_species: str
    species_confidence: float
    mixed_species_ratio: dict[str, float] | None = None
    
    # Biomass measurements
    biomass_kg_per_m2: float | None = None
    biomass_measurement_method: str = "visual_estimate"
    biomass_confidence: str = "moderate"  # low, moderate, high
    
    # Environmental conditions
    water_clarity_m: float | None = None
    canopy_type: str = "surface"  # surface, submerged, mixed
    kelp_density: str = "moderate"  # sparse, moderate, dense
    
    # Additional metadata
    surveyor: str = ""
    equipment_used: list[str] = field(default_factory=list)
    notes: str = ""
    photo_references: list[str] = field(default_factory=list)


@dataclass
class SpeciesValidationMetrics:
    """Validation metrics specific to species classification."""
    
    # Overall classification accuracy
    species_accuracy: float
    species_precision: dict[str, float]
    species_recall: dict[str, float]
    species_f1_score: dict[str, float]
    
    # Confusion matrix data
    confusion_matrix: dict[str, dict[str, int]]
    
    # Biomass estimation accuracy
    biomass_mae: float  # Mean Absolute Error
    biomass_rmse: float  # Root Mean Square Error
    biomass_r2: float  # R-squared correlation
    biomass_accuracy_by_species: dict[str, dict[str, float]]
    
    # Sample statistics
    total_samples: int
    samples_by_species: dict[str, int]
    
    # Quality metrics
    classification_confidence_distribution: dict[str, int]
    biomass_confidence_distribution: dict[str, int]


class FieldDataIngestor:
    """Handles ingestion of field survey data from multiple formats."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'excel', 'json', 'shapefile']
        self.logger = logging.getLogger(__name__)
    
    def ingest_csv_survey(self, file_path: Path) -> list[FieldSurveyRecord]:
        """Ingest field survey data from CSV format."""
        try:
            df = pd.read_csv(file_path)
            return self._convert_dataframe_to_records(df)
        except Exception as e:
            self.logger.error(f"Error ingesting CSV survey data: {e}")
            return []
    
    def ingest_excel_survey(self, file_path: Path, sheet_name: str = None) -> list[FieldSurveyRecord]:
        """Ingest field survey data from Excel format."""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            return self._convert_dataframe_to_records(df)
        except Exception as e:
            self.logger.error(f"Error ingesting Excel survey data: {e}")
            return []
    
    def ingest_json_survey(self, file_path: Path) -> list[FieldSurveyRecord]:
        """Ingest field survey data from JSON format."""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            records = []
            for record_data in data:
                record = self._convert_dict_to_record(record_data)
                if record:
                    records.append(record)
            return records
        except Exception as e:
            self.logger.error(f"Error ingesting JSON survey data: {e}")
            return []
    
    def _convert_dataframe_to_records(self, df: pd.DataFrame) -> list[FieldSurveyRecord]:
        """Convert pandas DataFrame to FieldSurveyRecord objects."""
        records = []
        
        for _, row in df.iterrows():
            try:
                # Parse species information
                observed_species = self._parse_species_list(row.get('observed_species', ''))
                primary_species = str(row.get('primary_species', 'unknown'))
                
                # Parse biomass measurements
                biomass = row.get('biomass_kg_per_m2')
                if pd.isna(biomass):
                    biomass = None
                else:
                    biomass = float(biomass)
                
                record = FieldSurveyRecord(
                    record_id=str(row.get('record_id', f"survey_{len(records)}")),
                    site_name=str(row.get('site_name', 'unknown')),
                    timestamp=pd.to_datetime(row.get('timestamp', datetime.now())),
                    lat=float(row.get('lat', 0.0)),
                    lng=float(row.get('lng', 0.0)),
                    depth_m=float(row.get('depth_m', 0.0)),
                    observed_species=observed_species,
                    primary_species=primary_species,
                    species_confidence=float(row.get('species_confidence', 0.5)),
                    biomass_kg_per_m2=biomass,
                    biomass_measurement_method=str(row.get('biomass_measurement_method', 'visual_estimate')),
                    biomass_confidence=str(row.get('biomass_confidence', 'moderate')),
                    water_clarity_m=row.get('water_clarity_m'),
                    canopy_type=str(row.get('canopy_type', 'surface')),
                    kelp_density=str(row.get('kelp_density', 'moderate')),
                    surveyor=str(row.get('surveyor', '')),
                    notes=str(row.get('notes', ''))
                )
                records.append(record)
                
            except Exception as e:
                self.logger.warning(f"Error converting row to record: {e}")
                continue
        
        return records
    
    def _convert_dict_to_record(self, data: dict[str, Any]) -> FieldSurveyRecord | None:
        """Convert dictionary to FieldSurveyRecord object."""
        try:
            return FieldSurveyRecord(
                record_id=data.get('record_id', f"record_{datetime.now().timestamp()}"),
                site_name=data.get('site_name', 'unknown'),
                timestamp=pd.to_datetime(data.get('timestamp', datetime.now())),
                lat=float(data.get('lat', 0.0)),
                lng=float(data.get('lng', 0.0)),
                depth_m=float(data.get('depth_m', 0.0)),
                observed_species=self._parse_species_list(data.get('observed_species', [])),
                primary_species=str(data.get('primary_species', 'unknown')),
                species_confidence=float(data.get('species_confidence', 0.5)),
                mixed_species_ratio=data.get('mixed_species_ratio'),
                biomass_kg_per_m2=data.get('biomass_kg_per_m2'),
                biomass_measurement_method=data.get('biomass_measurement_method', 'visual_estimate'),
                biomass_confidence=data.get('biomass_confidence', 'moderate'),
                water_clarity_m=data.get('water_clarity_m'),
                canopy_type=data.get('canopy_type', 'surface'),
                kelp_density=data.get('kelp_density', 'moderate'),
                surveyor=data.get('surveyor', ''),
                equipment_used=data.get('equipment_used', []),
                notes=data.get('notes', ''),
                photo_references=data.get('photo_references', [])
            )
        except Exception as e:
            self.logger.error(f"Error converting dict to record: {e}")
            return None
    
    def _parse_species_list(self, species_str: str | list[str]) -> list[str]:
        """Parse species list from string or list format."""
        if isinstance(species_str, list):
            return [str(s) for s in species_str]
        
        if isinstance(species_str, str):
            species_names = [s.strip() for s in species_str.split(',')]
            parsed_names = [name for name in species_names if name]
            return parsed_names if parsed_names else ['unknown']
        
        return ['unknown']


class SpeciesValidationAnalyzer:
    """Analyzes species classification accuracy against field survey data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_predictions_to_field_data(
        self,
        predictions: list[dict[str, Any]],
        field_records: list[FieldSurveyRecord],
        spatial_tolerance_m: float = 100.0
    ) -> SpeciesValidationMetrics:
        """Compare model predictions to field survey data."""
        
        # Match predictions to field records spatially and temporally
        matched_pairs = self._match_predictions_to_records(
            predictions, field_records, spatial_tolerance_m
        )
        
        if not matched_pairs:
            self.logger.warning("No matched prediction-field record pairs found")
            return self._create_empty_metrics()
        
        # Calculate species classification metrics
        species_metrics = self._calculate_species_classification_metrics(matched_pairs)
        
        # Calculate biomass estimation metrics
        biomass_metrics = self._calculate_biomass_estimation_metrics(matched_pairs)
        
        # Generate comprehensive validation metrics
        return SpeciesValidationMetrics(
            species_accuracy=species_metrics['accuracy'],
            species_precision=species_metrics['precision'],
            species_recall=species_metrics['recall'],
            species_f1_score=species_metrics['f1_score'],
            confusion_matrix=species_metrics['confusion_matrix'],
            biomass_mae=biomass_metrics['mae'],
            biomass_rmse=biomass_metrics['rmse'],
            biomass_r2=biomass_metrics['r2'],
            biomass_accuracy_by_species=biomass_metrics['by_species'],
            total_samples=len(matched_pairs),
            samples_by_species=species_metrics['samples_by_species'],
            classification_confidence_distribution=species_metrics['confidence_dist'],
            biomass_confidence_distribution=biomass_metrics['confidence_dist']
        )
    
    def _match_predictions_to_records(
        self,
        predictions: list[dict[str, Any]],
        field_records: list[FieldSurveyRecord],
        spatial_tolerance_m: float
    ) -> list[tuple[dict[str, Any], FieldSurveyRecord]]:
        """Match predictions to field records based on spatial and temporal proximity."""
        matched_pairs = []
        
        for prediction in predictions:
            # For this implementation, we'll use a simplified matching approach
            # In production, this would include proper spatial/temporal matching
            if len(field_records) > len(matched_pairs):
                matched_pairs.append((prediction, field_records[len(matched_pairs)]))
        
        return matched_pairs
    
    def _calculate_species_classification_metrics(
        self, matched_pairs: list[tuple[dict[str, Any], FieldSurveyRecord]]
    ) -> dict[str, Any]:
        """Calculate species classification accuracy metrics."""
        
        # Extract predictions and ground truth
        predicted_species = [pair[0].get('primary_species', 'unknown') for pair in matched_pairs]
        actual_species = [pair[1].primary_species for pair in matched_pairs]
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for pred, actual in zip(predicted_species, actual_species, strict=False) if pred == actual)
        accuracy = correct_predictions / len(matched_pairs) if matched_pairs else 0.0
        
        # Get unique species
        all_species = list(set(predicted_species + actual_species))
        
        # Calculate per-species metrics
        precision = {}
        recall = {}
        f1_score = {}
        samples_by_species = {}
        
        for species in all_species:
            # True positives, false positives, false negatives
            tp = sum(1 for pred, actual in zip(predicted_species, actual_species, strict=False) 
                    if pred == species and actual == species)
            fp = sum(1 for pred, actual in zip(predicted_species, actual_species, strict=False) 
                    if pred == species and actual != species)
            fn = sum(1 for pred, actual in zip(predicted_species, actual_species, strict=False) 
                    if pred != species and actual == species)
            
            # Calculate metrics
            precision[species] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[species] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision[species] * recall[species]) / (precision[species] + recall[species]) if (precision[species] + recall[species]) > 0 else 0.0
            f1_score[species] = f1
            
            samples_by_species[species] = sum(1 for actual in actual_species if actual == species)
        
        # Create confusion matrix
        confusion_matrix = {}
        for actual_species_name in all_species:
            confusion_matrix[actual_species_name] = {}
            for predicted_species_name in all_species:
                count = sum(1 for pred, actual in zip(predicted_species, actual_species, strict=False) 
                           if pred == predicted_species_name and actual == actual_species_name)
                confusion_matrix[actual_species_name][predicted_species_name] = count
        
        # Confidence distribution
        confidence_dist = {
            'high': sum(1 for pair in matched_pairs if pair[0].get('confidence', 0.0) >= 0.8),
            'medium': sum(1 for pair in matched_pairs if 0.5 <= pair[0].get('confidence', 0.0) < 0.8),
            'low': sum(1 for pair in matched_pairs if pair[0].get('confidence', 0.0) < 0.5)
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix,
            'samples_by_species': samples_by_species,
            'confidence_dist': confidence_dist
        }
    
    def _calculate_biomass_estimation_metrics(
        self, matched_pairs: list[tuple[dict[str, Any], FieldSurveyRecord]]
    ) -> dict[str, Any]:
        """Calculate biomass estimation accuracy metrics."""
        
        # Filter pairs with biomass data
        biomass_pairs = [
            (pair[0], pair[1]) for pair in matched_pairs 
            if pair[0].get('biomass_estimate_kg_per_m2') is not None and pair[1].biomass_kg_per_m2 is not None
        ]
        
        if not biomass_pairs:
            return {
                'mae': 0.0, 'rmse': 0.0, 'r2': 0.0,
                'by_species': {}, 'confidence_dist': {}
            }
        
        # Extract biomass predictions and ground truth
        predicted_biomass = np.array([pair[0].get('biomass_estimate_kg_per_m2', 0.0) for pair in biomass_pairs])
        actual_biomass = np.array([pair[1].biomass_kg_per_m2 for pair in biomass_pairs])
        
        # Calculate overall metrics
        mae = np.mean(np.abs(predicted_biomass - actual_biomass))
        rmse = np.sqrt(np.mean((predicted_biomass - actual_biomass) ** 2))
        
        # Calculate R-squared
        ss_res = np.sum((actual_biomass - predicted_biomass) ** 2)
        ss_tot = np.sum((actual_biomass - np.mean(actual_biomass)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate by species
        by_species = {}
        all_species = list(set([pair[1].primary_species for pair in biomass_pairs]))
        
        for species in all_species:
            species_pairs = [(pair[0], pair[1]) for pair in biomass_pairs if pair[1].primary_species == species]
            if species_pairs:
                species_pred = np.array([pair[0].get('biomass_estimate_kg_per_m2', 0.0) for pair in species_pairs])
                species_actual = np.array([pair[1].biomass_kg_per_m2 for pair in species_pairs])
                
                by_species[species] = {
                    'mae': float(np.mean(np.abs(species_pred - species_actual))),
                    'rmse': float(np.sqrt(np.mean((species_pred - species_actual) ** 2))),
                    'count': len(species_pairs)
                }
        
        # Confidence distribution based on field survey confidence
        confidence_dist = {
            'high': sum(1 for pair in biomass_pairs if pair[1].biomass_confidence == 'high'),
            'moderate': sum(1 for pair in biomass_pairs if pair[1].biomass_confidence == 'moderate'),
            'low': sum(1 for pair in biomass_pairs if pair[1].biomass_confidence == 'low')
        }
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'by_species': by_species,
            'confidence_dist': confidence_dist
        }
    
    def _create_empty_metrics(self) -> SpeciesValidationMetrics:
        """Create empty metrics when no data is available."""
        return SpeciesValidationMetrics(
            species_accuracy=0.0,
            species_precision={},
            species_recall={},
            species_f1_score={},
            confusion_matrix={},
            biomass_mae=0.0,
            biomass_rmse=0.0,
            biomass_r2=0.0,
            biomass_accuracy_by_species={},
            total_samples=0,
            samples_by_species={},
            classification_confidence_distribution={'high': 0, 'medium': 0, 'low': 0},
            biomass_confidence_distribution={'high': 0, 'moderate': 0, 'low': 0}
        )


class FieldSurveyReporter:
    """Generates comprehensive reports combining field survey and model prediction results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(
        self,
        campaign_id: str,
        validation_metrics: SpeciesValidationMetrics,
        field_records: list[FieldSurveyRecord],
        predictions: list[dict[str, Any]],
        output_path: Path | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        
        report = {
            'campaign_id': campaign_id,
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_summary(validation_metrics, field_records, predictions),
            'species_classification': self._generate_species_classification_report(validation_metrics),
            'biomass_estimation': self._generate_biomass_estimation_report(validation_metrics),
            'field_survey_summary': self._generate_field_survey_summary(field_records),
            'model_performance': self._generate_model_performance_summary(predictions),
            'recommendations': self._generate_recommendations(validation_metrics)
        }
        
        # Save to file if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_summary(
        self,
        metrics: SpeciesValidationMetrics,
        field_records: list[FieldSurveyRecord],
        predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate executive summary."""
        return {
            'total_validation_samples': metrics.total_samples,
            'total_field_records': len(field_records),
            'total_model_predictions': len(predictions),
            'overall_species_accuracy': f"{metrics.species_accuracy:.1%}",
            'biomass_estimation_accuracy': f"MAE: {metrics.biomass_mae:.2f} kg/m², R²: {metrics.biomass_r2:.3f}",
            'dominant_species_in_survey': max(metrics.samples_by_species.items(), key=lambda x: x[1])[0] if metrics.samples_by_species else 'None',
            'data_quality': self._assess_data_quality(metrics)
        }
    
    def _generate_species_classification_report(self, metrics: SpeciesValidationMetrics) -> dict[str, Any]:
        """Generate detailed species classification report."""
        return {
            'overall_accuracy': metrics.species_accuracy,
            'per_species_performance': {
                species: {
                    'precision': metrics.species_precision.get(species, 0.0),
                    'recall': metrics.species_recall.get(species, 0.0),
                    'f1_score': metrics.species_f1_score.get(species, 0.0),
                    'sample_count': metrics.samples_by_species.get(species, 0)
                }
                for species in set(list(metrics.species_precision.keys()) + list(metrics.samples_by_species.keys()))
            },
            'confusion_matrix': metrics.confusion_matrix,
            'confidence_distribution': metrics.classification_confidence_distribution
        }
    
    def _generate_biomass_estimation_report(self, metrics: SpeciesValidationMetrics) -> dict[str, Any]:
        """Generate detailed biomass estimation report."""
        return {
            'overall_performance': {
                'mean_absolute_error_kg_per_m2': metrics.biomass_mae,
                'root_mean_square_error_kg_per_m2': metrics.biomass_rmse,
                'r_squared_correlation': metrics.biomass_r2
            },
            'performance_by_species': metrics.biomass_accuracy_by_species,
            'confidence_distribution': metrics.biomass_confidence_distribution,
            'performance_assessment': self._assess_biomass_performance(metrics)
        }
    
    def _generate_field_survey_summary(self, field_records: list[FieldSurveyRecord]) -> dict[str, Any]:
        """Generate field survey data summary."""
        if not field_records:
            return {'status': 'No field records available'}
        
        species_counts = {}
        biomass_measurements = []
        site_coverage = set()
        
        for record in field_records:
            # Count species occurrences
            species_counts[record.primary_species] = species_counts.get(record.primary_species, 0) + 1
            
            # Collect biomass measurements
            if record.biomass_kg_per_m2 is not None:
                biomass_measurements.append(record.biomass_kg_per_m2)
            
            # Track site coverage
            site_coverage.add(record.site_name)
        
        return {
            'total_records': len(field_records),
            'sites_surveyed': list(site_coverage),
            'species_distribution': species_counts,
            'biomass_statistics': {
                'measurements_count': len(biomass_measurements),
                'mean_kg_per_m2': np.mean(biomass_measurements) if biomass_measurements else 0.0,
                'std_kg_per_m2': np.std(biomass_measurements) if biomass_measurements else 0.0,
                'min_kg_per_m2': np.min(biomass_measurements) if biomass_measurements else 0.0,
                'max_kg_per_m2': np.max(biomass_measurements) if biomass_measurements else 0.0
            },
            'temporal_coverage': {
                'earliest_survey': min(record.timestamp for record in field_records).isoformat(),
                'latest_survey': max(record.timestamp for record in field_records).isoformat()
            }
        }
    
    def _generate_model_performance_summary(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate model prediction performance summary."""
        if not predictions:
            return {'status': 'No model predictions available'}
        
        species_pred_counts = {}
        confidence_scores = []
        biomass_estimates = []
        
        for pred in predictions:
            species = pred.get('primary_species', 'unknown')
            species_pred_counts[species] = species_pred_counts.get(species, 0) + 1
            confidence_scores.append(pred.get('confidence', 0.0))
            
            biomass = pred.get('biomass_estimate_kg_per_m2')
            if biomass is not None:
                biomass_estimates.append(biomass)
        
        return {
            'total_predictions': len(predictions),
            'species_prediction_distribution': species_pred_counts,
            'confidence_statistics': {
                'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'std_confidence': np.std(confidence_scores) if confidence_scores else 0.0,
                'min_confidence': np.min(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': np.max(confidence_scores) if confidence_scores else 0.0
            },
            'biomass_prediction_statistics': {
                'predictions_count': len(biomass_estimates),
                'mean_kg_per_m2': np.mean(biomass_estimates) if biomass_estimates else 0.0,
                'std_kg_per_m2': np.std(biomass_estimates) if biomass_estimates else 0.0,
                'min_kg_per_m2': np.min(biomass_estimates) if biomass_estimates else 0.0,
                'max_kg_per_m2': np.max(biomass_estimates) if biomass_estimates else 0.0
            }
        }
    
    def _generate_recommendations(self, metrics: SpeciesValidationMetrics) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if metrics.species_accuracy < 0.7:
            recommendations.append("Species classification accuracy is below 70%. Consider collecting more training data or adjusting classification thresholds.")
        
        if metrics.biomass_mae > 5.0:
            recommendations.append("Biomass estimation error is high (>5 kg/m²). Review biomass calculation parameters and consider site-specific calibration.")
        
        if metrics.biomass_r2 < 0.5:
            recommendations.append("Low correlation between predicted and observed biomass. Investigate morphological feature extraction and enhancement factors.")
        
        # Check for species-specific issues
        for species, precision in metrics.species_precision.items():
            if precision < 0.6 and metrics.samples_by_species.get(species, 0) > 5:
                recommendations.append(f"Low precision for {species} classification. Consider species-specific model tuning.")
        
        if not recommendations:
            recommendations.append("Validation results meet quality standards. Continue monitoring with periodic validation campaigns.")
        
        return recommendations
    
    def _assess_data_quality(self, metrics: SpeciesValidationMetrics) -> str:
        """Assess overall data quality."""
        if metrics.total_samples >= 50 and metrics.species_accuracy >= 0.8:
            return "Excellent"
        elif metrics.total_samples >= 25 and metrics.species_accuracy >= 0.7:
            return "Good"
        elif metrics.total_samples >= 10 and metrics.species_accuracy >= 0.6:
            return "Moderate"
        else:
            return "Limited"
    
    def _assess_biomass_performance(self, metrics: SpeciesValidationMetrics) -> str:
        """Assess biomass estimation performance."""
        if metrics.biomass_r2 >= 0.8 and metrics.biomass_mae <= 2.0:
            return "Excellent"
        elif metrics.biomass_r2 >= 0.6 and metrics.biomass_mae <= 4.0:
            return "Good"
        elif metrics.biomass_r2 >= 0.4 and metrics.biomass_mae <= 6.0:
            return "Moderate"
        else:
            return "Poor"


class FieldSurveyIntegrationManager:
    """Main manager for field survey data integration."""
    
    def __init__(self, 
                 data_manager: ValidationDataManager,
                 species_classifier: SpeciesClassifier):
        self.data_manager = data_manager
        self.species_classifier = species_classifier
        self.ingestor = FieldDataIngestor()
        self.analyzer = SpeciesValidationAnalyzer()
        self.reporter = FieldSurveyReporter()
        self.logger = logging.getLogger(__name__)
    
    def process_field_survey_campaign(
        self,
        campaign_id: str,
        field_data_path: Path,
        model_predictions: list[SpeciesClassificationResult],
        output_dir: Path | None = None
    ) -> dict[str, Any]:
        """Process complete field survey validation campaign."""
        
        self.logger.info(f"Processing field survey campaign: {campaign_id}")
        
        try:
            # Ingest field survey data
            if field_data_path.suffix.lower() == '.csv':
                field_records = self.ingestor.ingest_csv_survey(field_data_path)
            elif field_data_path.suffix.lower() in ['.xlsx', '.xls']:
                field_records = self.ingestor.ingest_excel_survey(field_data_path)
            elif field_data_path.suffix.lower() == '.json':
                field_records = self.ingestor.ingest_json_survey(field_data_path)
            else:
                raise ValueError(f"Unsupported file format: {field_data_path.suffix}")
            
            self.logger.info(f"Ingested {len(field_records)} field survey records")
            
            # Analyze validation metrics
            validation_metrics = self.analyzer.compare_predictions_to_field_data(
                model_predictions, field_records
            )
            
            # Generate comprehensive report
            report = self.reporter.generate_comprehensive_report(
                campaign_id, validation_metrics, field_records, model_predictions,
                output_dir / f"{campaign_id}_validation_report.json" if output_dir else None
            )
            
            # Store results in validation database
            self._store_validation_results(campaign_id, validation_metrics, field_records)
            
            self.logger.info(f"Campaign processing complete. Overall accuracy: {validation_metrics.species_accuracy:.1%}")
            
            return {
                'status': 'success',
                'campaign_id': campaign_id,
                'validation_metrics': validation_metrics,
                'report': report,
                'field_records_count': len(field_records),
                'predictions_count': len(model_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing campaign {campaign_id}: {e}")
            return {
                'status': 'error',
                'campaign_id': campaign_id,
                'error': str(e)
            }
    
    def _store_validation_results(
        self,
        campaign_id: str,
        metrics: SpeciesValidationMetrics,
        field_records: list[FieldSurveyRecord]
    ):
        """Store validation results in the database."""
        try:
            # Store field records as ground truth measurements
            for record in field_records:
                ground_truth = GroundTruthMeasurement(
                    measurement_id=record.record_id,
                    campaign_id=campaign_id,
                    lat=record.lat,
                    lng=record.lng,
                    depth_m=record.depth_m,
                    kelp_present=record.primary_species != 'unknown',
                    kelp_species=record.primary_species,
                    kelp_density=record.kelp_density,
                    canopy_type=record.canopy_type,
                    timestamp=record.timestamp,
                    spectral_data={
                        'biomass_kg_per_m2': record.biomass_kg_per_m2,
                        'species_confidence': record.species_confidence,
                        'water_clarity_m': record.water_clarity_m
                    }
                )
                self.data_manager.add_ground_truth(ground_truth)
            
            # Store validation metrics
            metrics_data = {
                'species_accuracy': metrics.species_accuracy,
                'biomass_mae': metrics.biomass_mae,
                'biomass_rmse': metrics.biomass_rmse,
                'biomass_r2': metrics.biomass_r2,
                'total_samples': metrics.total_samples
            }
            
            # Note: In a full implementation, we'd have a dedicated validation results table
            self.logger.info(f"Stored {len(field_records)} validation records for campaign {campaign_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing validation results: {e}")


def create_field_survey_integration_manager(
    data_manager: ValidationDataManager,
    species_classifier: SpeciesClassifier
) -> FieldSurveyIntegrationManager:
    """Factory function to create a field survey integration manager."""
    return FieldSurveyIntegrationManager(data_manager, species_classifier)


def create_field_data_ingestor() -> FieldDataIngestor:
    """Factory function to create a field data ingestor."""
    return FieldDataIngestor()


def create_validation_analyzer() -> SpeciesValidationAnalyzer:
    """Factory function to create a species validation analyzer."""
    return SpeciesValidationAnalyzer()


def create_survey_reporter() -> FieldSurveyReporter:
    """Factory function to create a field survey reporter."""
    return FieldSurveyReporter()


def create_field_data_ingestor() -> FieldDataIngestor:
    """Factory function to create a field data ingestor."""
    return FieldDataIngestor()


def create_validation_analyzer() -> SpeciesValidationAnalyzer:
    """Factory function to create a species validation analyzer."""
    return SpeciesValidationAnalyzer()


def create_survey_reporter() -> FieldSurveyReporter:
    """Factory function to create a field survey reporter."""
    return FieldSurveyReporter() 
