#!/usr/bin/env python3
"""
Field Survey Integration Simple Demo - Task C2.4
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.kelpie_carbon_v1.validation.field_survey_integration import (
        FieldSurveyRecord,
        create_field_data_ingestor,
        create_validation_analyzer,
        create_survey_reporter
    )
    
    def demo_field_survey_integration():
        """Simple demonstration of field survey integration."""
        print("="*60)
        print("FIELD SURVEY INTEGRATION DEMO - TASK C2.4")
        print("="*60)
        
        # Create components
        ingestor = create_field_data_ingestor()
        analyzer = create_validation_analyzer()
        reporter = create_survey_reporter()
        
        print("\n✅ Components initialized successfully")
        
        # Create sample field record
        field_record = FieldSurveyRecord(
            record_id="demo_001",
            site_name="Saanich Inlet",
            timestamp=datetime(2024, 7, 15, 10, 0),
            lat=48.5830,
            lng=-123.5000,
            depth_m=10.0,
            observed_species=["nereocystis_luetkeana"],
            primary_species="nereocystis_luetkeana",
            species_confidence=0.85,
            biomass_kg_per_m2=8.5
        )
        
        print(f"\n✅ Created field record: {field_record.record_id}")
        print(f"   Site: {field_record.site_name}")
        print(f"   Species: {field_record.primary_species}")
        print(f"   Biomass: {field_record.biomass_kg_per_m2} kg/m²")
        
        # Create sample prediction
        prediction = {
            'record_id': 'demo_001',
            'primary_species': 'nereocystis_luetkeana',
            'confidence': 0.90,
            'biomass_estimate_kg_per_m2': 8.2
        }
        
        print(f"\n✅ Created model prediction")
        print(f"   Predicted species: {prediction['primary_species']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Predicted biomass: {prediction['biomass_estimate_kg_per_m2']} kg/m²")
        
        # Run validation
        metrics = analyzer.compare_predictions_to_field_data([prediction], [field_record])
        
        print(f"\n✅ Validation Analysis Complete")
        print(f"   Species accuracy: {metrics.species_accuracy:.1%}")
        print(f"   Biomass MAE: {metrics.biomass_mae:.2f} kg/m²")
        print(f"   Total samples: {metrics.total_samples}")
        
        # Generate report
        report = reporter.generate_comprehensive_report(
            "simple_demo", metrics, [field_record], [prediction]
        )
        
        print(f"\n✅ Report Generated")
        print(f"   Campaign ID: {report['campaign_id']}")
        print(f"   Data quality: {report['summary']['data_quality']}")
        
        print("\n" + "="*60)
        print("🎉 TASK C2.4 DEMONSTRATION SUCCESSFUL!")
        print("   All field survey integration components working")
        print("="*60)
        
        return True
    
    if __name__ == "__main__":
        try:
            demo_field_survey_integration()
        except Exception as e:
            print(f"Demo failed: {e}")
            exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Field survey integration module may not be available")
    exit(1) 