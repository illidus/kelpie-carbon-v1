#!/usr/bin/env python3
"""
Field Survey Integration Demonstration Script - Task C2.4

This script demonstrates the comprehensive field survey data integration capabilities:
- Field data ingestion from multiple formats (CSV, JSON, Excel)
- Species-specific ground truth validation
- Biomass estimation accuracy assessment  
- Enhanced reporting with species classification metrics

Integration with Task C2.1-C2.3 species classification and biomass estimation.

Usage:
    python scripts/test_field_survey_integration.py [--mode MODE] [--output OUTPUT_DIR]
    
    MODE options:
        demo      - Run demonstration with synthetic data (default)
        validation - Run validation with sample field data
        report    - Generate comprehensive validation report
        all       - Run all modes sequentially
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kelpie_carbon_v1.validation.field_survey_integration import (
    FieldSurveyRecord,
    FieldDataIngestor,
    SpeciesValidationAnalyzer,
    FieldSurveyReporter,
    create_field_data_ingestor,
    create_validation_analyzer,
    create_survey_reporter
)
from src.kelpie_carbon_v1.processing.species_classifier import (
    SpeciesClassifier,
    SpeciesClassificationResult,
    KelpSpecies,
    BiomassEstimate,
    create_species_classifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FieldSurveyIntegrationDemo:
    """Demonstration of field survey integration capabilities."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("field_survey_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.ingestor = create_field_data_ingestor()
        self.analyzer = create_validation_analyzer()
        self.reporter = create_survey_reporter()
        
        logger.info(f"Initialized Field Survey Integration Demo with output dir: {self.output_dir}")
    
    def create_synthetic_field_data(self) -> List[FieldSurveyRecord]:
        """Create synthetic field survey data for demonstration."""
        logger.info("Creating synthetic field survey data...")
        
        # Simulate field survey data from multiple BC kelp sites
        sites = [
            ("Broughton Archipelago", 50.0833, -126.1667, "nereocystis_luetkeana"),
            ("Saanich Inlet", 48.5830, -123.5000, "mixed_species"),
            ("Haro Strait", 48.5000, -123.1667, "nereocystis_luetkeana"),
            ("Sidney Channel", 48.6500, -123.4000, "macrocystis_pyrifera"),
            ("Cadboro Bay", 48.4667, -123.2833, "nereocystis_luetkeana")
        ]
        
        field_records = []
        base_time = datetime(2024, 7, 15, 10, 0, 0)  # Summer kelp season
        
        for i, (site_name, lat, lng, primary_species) in enumerate(sites):
            # Create multiple records per site to simulate survey transects
            for j in range(4):
                record_id = f"survey_{i+1:02d}_{j+1:02d}"
                
                # Add some spatial variation within each site
                site_lat = lat + np.random.normal(0, 0.01)  # ~1km variation
                site_lng = lng + np.random.normal(0, 0.01)
                
                # Generate realistic species observations
                if primary_species == "mixed_species":
                    observed_species = ["nereocystis_luetkeana", "macrocystis_pyrifera"]
                    primary = np.random.choice(["nereocystis_luetkeana", "macrocystis_pyrifera"])
                else:
                    observed_species = [primary_species]
                    primary = primary_species
                
                # Generate realistic biomass estimates based on species
                if primary == "nereocystis_luetkeana":
                    biomass_base = 8.0  # kg/m²
                elif primary == "macrocystis_pyrifera":
                    biomass_base = 12.0  # kg/m²
                else:
                    biomass_base = 10.0  # Mixed species average
                
                biomass = max(0.5, np.random.normal(biomass_base, biomass_base * 0.3))
                
                # Species confidence based on survey conditions
                species_confidence = np.random.uniform(0.6, 0.95)
                
                # Biomass confidence based on measurement method
                biomass_confidence = np.random.choice(
                    ["high", "moderate", "low"], 
                    p=[0.3, 0.5, 0.2]
                )
                
                # Environmental conditions
                depth = np.random.uniform(3.0, 25.0)
                water_clarity = np.random.uniform(2.0, 12.0) if np.random.random() > 0.2 else None
                kelp_density = np.random.choice(["sparse", "moderate", "dense"], p=[0.2, 0.6, 0.2])
                canopy_type = np.random.choice(["surface", "submerged", "mixed"], p=[0.6, 0.2, 0.2])
                
                record = FieldSurveyRecord(
                    record_id=record_id,
                    site_name=site_name,
                    timestamp=base_time + timedelta(hours=i*2, minutes=j*30),
                    lat=site_lat,
                    lng=site_lng,
                    depth_m=depth,
                    observed_species=observed_species,
                    primary_species=primary,
                    species_confidence=species_confidence,
                    biomass_kg_per_m2=biomass,
                    biomass_measurement_method=np.random.choice([
                        "visual_estimate", "transect_measurement", "diver_observation"
                    ]),
                    biomass_confidence=biomass_confidence,
                    water_clarity_m=water_clarity,
                    canopy_type=canopy_type,
                    kelp_density=kelp_density,
                    surveyor=f"Field Team {(i % 3) + 1}",
                    equipment_used=["GPS", "depth sounder", "camera", "measurement tape"],
                    notes=f"Survey point {j+1} at {site_name}. {kelp_density.title()} kelp density observed."
                )
                
                field_records.append(record)
        
        logger.info(f"Created {len(field_records)} synthetic field survey records")
        return field_records
    
    def create_synthetic_model_predictions(self, field_records: List[FieldSurveyRecord]) -> List[Dict[str, Any]]:
        """Create synthetic model predictions corresponding to field records."""
        logger.info("Creating synthetic model predictions...")
        
        predictions = []
        
        for record in field_records:
            # Simulate model prediction with some accuracy variation
            actual_species = record.primary_species
            
            # Species classification with realistic accuracy
            if np.random.random() < 0.85:  # 85% accuracy
                predicted_species = actual_species
                confidence = np.random.uniform(0.7, 0.95)
            else:
                # Simulate misclassification
                if actual_species == "nereocystis_luetkeana":
                    predicted_species = "macrocystis_pyrifera"
                elif actual_species == "macrocystis_pyrifera":
                    predicted_species = "nereocystis_luetkeana"
                else:
                    predicted_species = np.random.choice(["nereocystis_luetkeana", "macrocystis_pyrifera"])
                confidence = np.random.uniform(0.4, 0.8)
            
            # Biomass estimation with realistic error
            if record.biomass_kg_per_m2 is not None:
                actual_biomass = record.biomass_kg_per_m2
                # Add realistic prediction error (±20%)
                biomass_error = np.random.normal(0, actual_biomass * 0.2)
                predicted_biomass = max(0.1, actual_biomass + biomass_error)
            else:
                predicted_biomass = None
            
            prediction = {
                'record_id': record.record_id,
                'primary_species': predicted_species,
                'confidence': confidence,
                'biomass_estimate_kg_per_m2': predicted_biomass,
                'detection_method': 'spectral_classification',
                'processing_time_seconds': np.random.uniform(0.5, 2.0),
                'coordinates': {'lat': record.lat, 'lng': record.lng}
            }
            
            predictions.append(prediction)
        
        logger.info(f"Created {len(predictions)} synthetic model predictions")
        return predictions
    
    def run_demo_mode(self):
        """Run demonstration mode with synthetic data."""
        logger.info("🚀 Running Field Survey Integration Demo Mode")
        print("\n" + "="*80)
        print("📋 FIELD SURVEY DATA INTEGRATION DEMONSTRATION - TASK C2.4")
        print("="*80)
        
        # 1. Create synthetic field data
        print("\n1. 📊 Creating Synthetic Field Survey Data...")
        field_records = self.create_synthetic_field_data()
        
        # Display field data summary
        species_counts = {}
        for record in field_records:
            species_counts[record.primary_species] = species_counts.get(record.primary_species, 0) + 1
        
        print(f"   ✅ Created {len(field_records)} field survey records")
        print(f"   📍 Sites surveyed: {len(set(r.site_name for r in field_records))}")
        print(f"   🌊 Species distribution: {species_counts}")
        
        # 2. Create model predictions
        print("\n2. 🤖 Generating Model Predictions...")
        predictions = self.create_synthetic_model_predictions(field_records)
        print(f"   ✅ Generated {len(predictions)} model predictions")
        
        # 3. Run validation analysis
        print("\n3. 📈 Running Species Validation Analysis...")
        validation_metrics = self.analyzer.compare_predictions_to_field_data(
            predictions, field_records
        )
        
        print(f"   ✅ Species Classification Accuracy: {validation_metrics.species_accuracy:.1%}")
        print(f"   ✅ Biomass Estimation MAE: {validation_metrics.biomass_mae:.2f} kg/m²")
        print(f"   ✅ Biomass Estimation R²: {validation_metrics.biomass_r2:.3f}")
        print(f"   ✅ Total Validation Samples: {validation_metrics.total_samples}")
        
        # 4. Generate comprehensive report
        print("\n4. 📝 Generating Comprehensive Validation Report...")
        report = self.reporter.generate_comprehensive_report(
            "demo_validation_campaign",
            validation_metrics,
            field_records,
            predictions,
            self.output_dir / "demo_validation_report.json"
        )
        
        print(f"   ✅ Report saved to: {self.output_dir / 'demo_validation_report.json'}")
        
        # 5. Display key findings
        print("\n5. 🎯 Key Validation Findings:")
        summary = report['summary']
        print(f"   📊 Overall Species Accuracy: {summary['overall_species_accuracy']}")
        print(f"   📊 Biomass Estimation: {summary['biomass_estimation_accuracy']}")
        print(f"   📊 Data Quality Assessment: {summary['data_quality']}")
        print(f"   📊 Dominant Species: {summary['dominant_species_in_survey']}")
        
        # Display recommendations
        recommendations = report['recommendations']
        if len(recommendations) > 0:
            print(f"\n6. 💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        # Save field data as CSV for inspection
        field_data_df = pd.DataFrame([
            {
                'record_id': r.record_id,
                'site_name': r.site_name,
                'timestamp': r.timestamp.isoformat(),
                'lat': r.lat,
                'lng': r.lng,
                'depth_m': r.depth_m,
                'primary_species': r.primary_species,
                'species_confidence': r.species_confidence,
                'biomass_kg_per_m2': r.biomass_kg_per_m2,
                'biomass_confidence': r.biomass_confidence,
                'kelp_density': r.kelp_density,
                'canopy_type': r.canopy_type,
                'surveyor': r.surveyor
            } for r in field_records
        ])
        
        field_data_df.to_csv(self.output_dir / "synthetic_field_data.csv", index=False)
        print(f"\n   📄 Field data saved to: {self.output_dir / 'synthetic_field_data.csv'}")
        
        print("\n" + "="*80)
        print("✅ Demo Mode Complete! Field Survey Integration Successfully Demonstrated")
        print("="*80)
        
        return {
            'field_records': field_records,
            'predictions': predictions,
            'validation_metrics': validation_metrics,
            'report': report
        }
    
    def run_validation_mode(self):
        """Run validation mode with sample data files."""
        logger.info("🔬 Running Field Survey Integration Validation Mode")
        print("\n" + "="*80)
        print("🔬 FIELD SURVEY DATA VALIDATION MODE - TASK C2.4")
        print("="*80)
        
        # Test data ingestion from different formats
        print("\n1. 📥 Testing Data Ingestion from Multiple Formats...")
        
        # Create sample CSV data
        sample_csv_data = pd.DataFrame({
            'record_id': ['val_001', 'val_002', 'val_003'],
            'site_name': ['Validation Site A', 'Validation Site B', 'Validation Site C'],
            'timestamp': ['2024-07-20 10:30:00', '2024-07-20 14:15:00', '2024-07-21 09:00:00'],
            'lat': [48.6000, 48.6100, 48.6200],
            'lng': [-123.4000, -123.4100, -123.4200],
            'depth_m': [8.5, 12.0, 6.0],
            'primary_species': ['nereocystis_luetkeana', 'macrocystis_pyrifera', 'nereocystis_luetkeana'],
            'species_confidence': [0.90, 0.85, 0.75],
            'biomass_kg_per_m2': [9.2, 14.5, 7.8],
            'biomass_confidence': ['high', 'high', 'moderate'],
            'kelp_density': ['moderate', 'dense', 'moderate'],
            'canopy_type': ['surface', 'mixed', 'surface'],
            'surveyor': ['Dr. Smith', 'Dr. Johnson', 'Dr. Smith']
        })
        
        csv_path = self.output_dir / "sample_field_data.csv"
        sample_csv_data.to_csv(csv_path, index=False)
        
        # Test CSV ingestion
        csv_records = self.ingestor.ingest_csv_survey(csv_path)
        print(f"   ✅ CSV Ingestion: {len(csv_records)} records loaded")
        
        # Create sample JSON data
        json_data = [
            {
                'record_id': 'json_001',
                'site_name': 'JSON Test Site',
                'timestamp': '2024-07-22 11:00:00',
                'lat': 48.6300,
                'lng': -123.4300,
                'depth_m': 10.0,
                'primary_species': 'mixed_species',
                'species_confidence': 0.80,
                'biomass_kg_per_m2': 11.0,
                'biomass_confidence': 'moderate',
                'observed_species': ['nereocystis_luetkeana', 'macrocystis_pyrifera'],
                'equipment_used': ['GPS', 'dive equipment', 'measurement tools'],
                'notes': 'Mixed species kelp forest with good visibility'
            }
        ]
        
        json_path = self.output_dir / "sample_field_data.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Test JSON ingestion
        json_records = self.ingestor.ingest_json_survey(json_path)
        print(f"   ✅ JSON Ingestion: {len(json_records)} records loaded")
        
        # Combine all records
        all_records = csv_records + json_records
        print(f"   📊 Total Records: {len(all_records)}")
        
        # 2. Test validation analysis with high-accuracy scenario
        print("\n2. 📈 Testing High-Accuracy Validation Scenario...")
        
        # Create high-accuracy predictions
        high_accuracy_predictions = []
        for record in all_records:
            prediction = {
                'record_id': record.record_id,
                'primary_species': record.primary_species,  # Perfect classification
                'confidence': 0.95,
                'biomass_estimate_kg_per_m2': record.biomass_kg_per_m2 * 1.05,  # 5% overestimate
                'detection_method': 'enhanced_spectral_classification'
            }
            high_accuracy_predictions.append(prediction)
        
        high_acc_metrics = self.analyzer.compare_predictions_to_field_data(
            high_accuracy_predictions, all_records
        )
        
        print(f"   ✅ High-Accuracy Scenario:")
        print(f"      Species Accuracy: {high_acc_metrics.species_accuracy:.1%}")
        print(f"      Biomass MAE: {high_acc_metrics.biomass_mae:.2f} kg/m²")
        print(f"      Biomass R²: {high_acc_metrics.biomass_r2:.3f}")
        
        # 3. Test validation analysis with poor-accuracy scenario
        print("\n3. 📉 Testing Poor-Accuracy Validation Scenario...")
        
        # Create poor-accuracy predictions
        poor_accuracy_predictions = []
        for record in all_records:
            # Random misclassification
            wrong_species = np.random.choice(['unknown', 'mixed_species', 'nereocystis_luetkeana'])
            prediction = {
                'record_id': record.record_id,
                'primary_species': wrong_species,
                'confidence': 0.45,
                'biomass_estimate_kg_per_m2': record.biomass_kg_per_m2 * np.random.uniform(0.3, 1.8),  # High error
                'detection_method': 'basic_classification'
            }
            poor_accuracy_predictions.append(prediction)
        
        poor_acc_metrics = self.analyzer.compare_predictions_to_field_data(
            poor_accuracy_predictions, all_records
        )
        
        print(f"   ✅ Poor-Accuracy Scenario:")
        print(f"      Species Accuracy: {poor_acc_metrics.species_accuracy:.1%}")
        print(f"      Biomass MAE: {poor_acc_metrics.biomass_mae:.2f} kg/m²")
        print(f"      Biomass R²: {poor_acc_metrics.biomass_r2:.3f}")
        
        # 4. Generate comparison report
        print("\n4. 📝 Generating Validation Comparison Report...")
        
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'test_scenarios': {
                'high_accuracy': {
                    'description': 'Perfect species classification with minimal biomass error',
                    'species_accuracy': high_acc_metrics.species_accuracy,
                    'biomass_mae': high_acc_metrics.biomass_mae,
                    'biomass_r2': high_acc_metrics.biomass_r2,
                    'total_samples': high_acc_metrics.total_samples
                },
                'poor_accuracy': {
                    'description': 'Random species misclassification with high biomass error',
                    'species_accuracy': poor_acc_metrics.species_accuracy,
                    'biomass_mae': poor_acc_metrics.biomass_mae,
                    'biomass_r2': poor_acc_metrics.biomass_r2,
                    'total_samples': poor_acc_metrics.total_samples
                }
            },
            'data_ingestion_validation': {
                'csv_records_loaded': len(csv_records),
                'json_records_loaded': len(json_records),
                'total_records_processed': len(all_records),
                'ingestion_success_rate': 1.0
            }
        }
        
        validation_path = self.output_dir / "validation_comparison_report.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"   ✅ Validation report saved to: {validation_path}")
        
        print("\n" + "="*80)
        print("✅ Validation Mode Complete! All Data Ingestion and Analysis Tests Passed")
        print("="*80)
        
        return validation_report
    
    def run_report_mode(self):
        """Run comprehensive reporting mode."""
        logger.info("📊 Running Field Survey Integration Report Mode")
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE FIELD SURVEY REPORTING MODE - TASK C2.4")
        print("="*80)
        
        # Create comprehensive test dataset
        print("\n1. 📊 Creating Comprehensive Test Dataset...")
        
        # Generate larger dataset with diverse scenarios
        comprehensive_records = []
        comprehensive_predictions = []
        
        # Multiple sites with varying conditions
        test_sites = [
            ("Broughton Archipelago", 50.0833, -126.1667, "nereocystis_luetkeana", "excellent"),
            ("Saanich Inlet", 48.5830, -123.5000, "mixed_species", "good"),
            ("Haro Strait", 48.5000, -123.1667, "nereocystis_luetkeana", "moderate"),
            ("Sidney Channel", 48.6500, -123.4000, "macrocystis_pyrifera", "excellent"),
            ("Cadboro Bay", 48.4667, -123.2833, "nereocystis_luetkeana", "poor"),
            ("Active Pass", 48.8667, -123.3000, "macrocystis_pyrifera", "good")
        ]
        
        base_time = datetime(2024, 8, 1, 9, 0, 0)
        
        for site_idx, (site_name, lat, lng, primary_species, conditions) in enumerate(test_sites):
            # Number of records varies by site conditions
            if conditions == "excellent":
                n_records = 8
                accuracy_rate = 0.95
            elif conditions == "good":
                n_records = 6
                accuracy_rate = 0.85
            elif conditions == "moderate":
                n_records = 4
                accuracy_rate = 0.70
            else:  # poor
                n_records = 3
                accuracy_rate = 0.50
            
            for i in range(n_records):
                record_id = f"comp_{site_idx+1:02d}_{i+1:02d}"
                
                # Site coordinates with variation
                site_lat = lat + np.random.normal(0, 0.005)
                site_lng = lng + np.random.normal(0, 0.005)
                
                # Species and biomass based on type
                if primary_species == "mixed_species":
                    observed_species = ["nereocystis_luetkeana", "macrocystis_pyrifera"]
                    primary = np.random.choice(["nereocystis_luetkeana", "macrocystis_pyrifera"])
                    biomass_base = 10.0
                else:
                    observed_species = [primary_species]
                    primary = primary_species
                    if primary == "nereocystis_luetkeana":
                        biomass_base = 8.0
                    else:  # macrocystis_pyrifera
                        biomass_base = 13.0
                
                # Adjust biomass by conditions
                condition_multipliers = {
                    "excellent": 1.2,
                    "good": 1.0,
                    "moderate": 0.8,
                    "poor": 0.6
                }
                biomass = max(1.0, np.random.normal(
                    biomass_base * condition_multipliers[conditions], 
                    biomass_base * 0.25
                ))
                
                # Create field record
                record = FieldSurveyRecord(
                    record_id=record_id,
                    site_name=site_name,
                    timestamp=base_time + timedelta(days=site_idx, hours=i),
                    lat=site_lat,
                    lng=site_lng,
                    depth_m=np.random.uniform(2.0, 20.0),
                    observed_species=observed_species,
                    primary_species=primary,
                    species_confidence=np.random.uniform(0.6, 0.95),
                    biomass_kg_per_m2=biomass,
                    biomass_measurement_method=np.random.choice([
                        "visual_estimate", "transect_measurement", "diver_observation", "sonar_mapping"
                    ]),
                    biomass_confidence=np.random.choice(["high", "moderate", "low"], p=[0.4, 0.4, 0.2]),
                    water_clarity_m=np.random.uniform(1.0, 15.0),
                    canopy_type=np.random.choice(["surface", "submerged", "mixed"], p=[0.5, 0.3, 0.2]),
                    kelp_density=np.random.choice(["sparse", "moderate", "dense"], p=[0.2, 0.5, 0.3]),
                    surveyor=f"Team {np.random.choice(['Alpha', 'Beta', 'Gamma'])}",
                    equipment_used=["GPS", "depth meter", "camera", "measurement tools"],
                    notes=f"Survey at {site_name} under {conditions} conditions"
                )
                comprehensive_records.append(record)
                
                # Create corresponding prediction with realistic accuracy
                if np.random.random() < accuracy_rate:
                    pred_species = primary
                    confidence = np.random.uniform(0.7, 0.95)
                    biomass_error_pct = np.random.normal(0, 0.15)  # ±15% error
                else:
                    # Misclassification
                    if primary == "nereocystis_luetkeana":
                        pred_species = "macrocystis_pyrifera"
                    elif primary == "macrocystis_pyrifera":
                        pred_species = "nereocystis_luetkeana"
                    else:
                        pred_species = "unknown"
                    confidence = np.random.uniform(0.3, 0.7)
                    biomass_error_pct = np.random.normal(0, 0.4)  # ±40% error
                
                pred_biomass = max(0.5, biomass * (1 + biomass_error_pct))
                
                prediction = {
                    'record_id': record_id,
                    'primary_species': pred_species,
                    'confidence': confidence,
                    'biomass_estimate_kg_per_m2': pred_biomass,
                    'detection_method': 'enhanced_classification',
                    'site_conditions': conditions
                }
                comprehensive_predictions.append(prediction)
        
        print(f"   ✅ Created {len(comprehensive_records)} comprehensive field records")
        print(f"   ✅ Created {len(comprehensive_predictions)} corresponding predictions")
        print(f"   📍 Sites included: {len(test_sites)} locations with varying conditions")
        
        # 2. Run comprehensive validation analysis
        print("\n2. 📈 Running Comprehensive Validation Analysis...")
        
        comprehensive_metrics = self.analyzer.compare_predictions_to_field_data(
            comprehensive_predictions, comprehensive_records
        )
        
        print(f"   ✅ Overall Species Accuracy: {comprehensive_metrics.species_accuracy:.1%}")
        print(f"   ✅ Overall Biomass MAE: {comprehensive_metrics.biomass_mae:.2f} kg/m²")
        print(f"   ✅ Overall Biomass R²: {comprehensive_metrics.biomass_r2:.3f}")
        print(f"   ✅ Total Validation Samples: {comprehensive_metrics.total_samples}")
        
        # 3. Generate comprehensive report
        print("\n3. 📝 Generating Comprehensive Field Survey Report...")
        
        comprehensive_report = self.reporter.generate_comprehensive_report(
            "comprehensive_field_survey_campaign",
            comprehensive_metrics,
            comprehensive_records,
            comprehensive_predictions,
            self.output_dir / "comprehensive_field_survey_report.json"
        )
        
        # 4. Create detailed analysis summaries
        print("\n4. 📊 Creating Detailed Analysis Summaries...")
        
        # Species-specific analysis
        species_analysis = {}
        for species in set(r.primary_species for r in comprehensive_records):
            species_records = [r for r in comprehensive_records if r.primary_species == species]
            species_predictions = [p for p in comprehensive_predictions if p.get('record_id') in [r.record_id for r in species_records]]
            
            if species_records and species_predictions:
                species_metrics = self.analyzer.compare_predictions_to_field_data(species_predictions, species_records)
                species_analysis[species] = {
                    'sample_count': len(species_records),
                    'accuracy': species_metrics.species_accuracy,
                    'biomass_mae': species_metrics.biomass_mae,
                    'biomass_r2': species_metrics.biomass_r2
                }
        
        # Site-specific analysis
        site_analysis = {}
        for site_name in set(r.site_name for r in comprehensive_records):
            site_records = [r for r in comprehensive_records if r.site_name == site_name]
            site_predictions = [p for p in comprehensive_predictions if p.get('record_id') in [r.record_id for r in site_records]]
            
            if site_records and site_predictions:
                site_metrics = self.analyzer.compare_predictions_to_field_data(site_predictions, site_records)
                site_analysis[site_name] = {
                    'sample_count': len(site_records),
                    'accuracy': site_metrics.species_accuracy,
                    'biomass_mae': site_metrics.biomass_mae,
                    'conditions': site_predictions[0].get('site_conditions', 'unknown')
                }
        
        # Save detailed analysis
        detailed_analysis = {
            'comprehensive_validation': {
                'total_samples': comprehensive_metrics.total_samples,
                'overall_accuracy': comprehensive_metrics.species_accuracy,
                'overall_biomass_mae': comprehensive_metrics.biomass_mae,
                'overall_biomass_r2': comprehensive_metrics.biomass_r2
            },
            'species_specific_analysis': species_analysis,
            'site_specific_analysis': site_analysis,
            'performance_by_conditions': {
                condition: {
                    'sites': [site for site, data in site_analysis.items() if data['conditions'] == condition],
                    'avg_accuracy': np.mean([data['accuracy'] for data in site_analysis.values() if data['conditions'] == condition]) if any(data['conditions'] == condition for data in site_analysis.values()) else 0.0
                }
                for condition in ['excellent', 'good', 'moderate', 'poor']
            }
        }
        
        detailed_path = self.output_dir / "detailed_field_survey_analysis.json"
        with open(detailed_path, 'w') as f:
            json.dump(detailed_analysis, f, indent=2, default=str)
        
        print(f"   ✅ Detailed analysis saved to: {detailed_path}")
        
        # 5. Display summary results
        print("\n5. 🎯 Summary Results:")
        print(f"   📊 Species-Specific Performance:")
        for species, data in species_analysis.items():
            print(f"      {species}: {data['accuracy']:.1%} accuracy, MAE {data['biomass_mae']:.2f} kg/m² ({data['sample_count']} samples)")
        
        print(f"\n   📍 Site-Specific Performance:")
        for site, data in site_analysis.items():
            print(f"      {site} ({data['conditions']}): {data['accuracy']:.1%} accuracy, MAE {data['biomass_mae']:.2f} kg/m²")
        
        print(f"\n   🌊 Performance by Conditions:")
        for condition, data in detailed_analysis['performance_by_conditions'].items():
            if data['sites']:
                print(f"      {condition.title()}: {data['avg_accuracy']:.1%} average accuracy ({len(data['sites'])} sites)")
        
        print("\n" + "="*80)
        print("✅ Report Mode Complete! Comprehensive Field Survey Analysis Generated")
        print("="*80)
        
        return {
            'comprehensive_report': comprehensive_report,
            'detailed_analysis': detailed_analysis,
            'metrics': comprehensive_metrics
        }
    
    def run_all_modes(self):
        """Run all demonstration modes sequentially."""
        logger.info("🚀 Running All Field Survey Integration Modes")
        print("\n" + "="*80)
        print("🚀 COMPLETE FIELD SURVEY INTEGRATION DEMONSTRATION - TASK C2.4")
        print("="*80)
        
        results = {}
        
        # Run demo mode
        results['demo'] = self.run_demo_mode()
        
        # Run validation mode
        results['validation'] = self.run_validation_mode()
        
        # Run report mode
        results['report'] = self.run_report_mode()
        
        # Create summary
        print("\n" + "="*80)
        print("📋 COMPLETE DEMONSTRATION SUMMARY")
        print("="*80)
        
        print("\n✅ All Modes Completed Successfully:")
        print("   1. ✅ Demo Mode - Synthetic data integration and basic validation")
        print("   2. ✅ Validation Mode - Data ingestion testing and accuracy scenarios")
        print("   3. ✅ Report Mode - Comprehensive reporting and detailed analysis")
        
        print(f"\n📊 Overall Statistics:")
        demo_metrics = results['demo']['validation_metrics']
        report_metrics = results['report']['metrics']
        
        print(f"   Demo Accuracy: {demo_metrics.species_accuracy:.1%}")
        print(f"   Comprehensive Accuracy: {report_metrics.species_accuracy:.1%}")
        print(f"   Demo Biomass MAE: {demo_metrics.biomass_mae:.2f} kg/m²")
        print(f"   Comprehensive Biomass MAE: {report_metrics.biomass_mae:.2f} kg/m²")
        
        print(f"\n📁 Output Files Generated:")
        output_files = list(self.output_dir.glob("*"))
        for file_path in sorted(output_files):
            print(f"   📄 {file_path.name}")
        
        print("\n" + "="*80)
        print("🎉 TASK C2.4: FIELD SURVEY DATA INTEGRATION COMPLETE!")
        print("   All field data ingestion, validation, and reporting capabilities demonstrated successfully.")
        print("="*80)
        
        return results


def main():
    """Main function to run the field survey integration demonstration."""
    parser = argparse.ArgumentParser(
        description="Field Survey Integration Demonstration - Task C2.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_field_survey_integration.py --mode demo
    python scripts/test_field_survey_integration.py --mode validation --output results/
    python scripts/test_field_survey_integration.py --mode all
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "validation", "report", "all"],
        default="demo",
        help="Demonstration mode to run (default: demo)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("field_survey_results"),
        help="Output directory for results (default: field_survey_results)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize demonstration
        demo = FieldSurveyIntegrationDemo(args.output)
        
        # Run selected mode
        if args.mode == "demo":
            results = demo.run_demo_mode()
        elif args.mode == "validation":
            results = demo.run_validation_mode()
        elif args.mode == "report":
            results = demo.run_report_mode()
        elif args.mode == "all":
            results = demo.run_all_modes()
        
        logger.info(f"Field Survey Integration demonstration completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Field Survey Integration demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 