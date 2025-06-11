"""
Validation and testing framework for SKEMA kelp detection.

This module provides comprehensive validation capabilities including:
- Real-world validation against field surveys and expert annotations
- Environmental robustness testing across different conditions
- Species-level classification validation
- Field survey data integration and analysis
- Temporal validation and environmental driver analysis

The validation framework supports multiple validation approaches:
1. Ground-truth comparison validation
2. Environmental robustness testing
3. Species-specific validation
4. Field survey integration
5. Temporal persistence validation
"""

from .real_world_validation import (
    ValidationSite,
    ValidationResult,
    RealWorldValidator,
    create_real_world_validator,
    run_comprehensive_validation,
    validate_primary_sites,
    validate_with_controls,
)

from .data_manager import (
    ValidationCampaign,
    ValidationDataManager,
    GroundTruthMeasurement,
)

from .environmental_testing import (
    EnvironmentalRobustnessValidator,
    create_environmental_validator,
    run_comprehensive_environmental_testing,
)

# Import species classifier from processing module
from ..processing.species_classifier import (
    SpeciesClassifier,
    SpeciesClassificationResult,
    BiomassEstimate,
    create_species_classifier,
    run_species_classification,
)

from .field_survey_integration import (
    FieldSurveyRecord,
    FieldDataIngestor,
    SpeciesValidationAnalyzer,
    FieldSurveyReporter,
    create_field_data_ingestor,
    create_validation_analyzer,
    create_survey_reporter,
)

from .temporal_validation import (
    TemporalValidator,
    TemporalDataPoint,
    SeasonalPattern,
    TemporalValidationResult,
    create_temporal_validator,
    run_broughton_temporal_validation,
    run_comprehensive_temporal_analysis,
)

__all__ = [
    # Real-world validation
    "ValidationSite",
    "ValidationResult",
    "ValidationCampaign", 
    "RealWorldValidator",
    "ValidationDataManager",
    "GroundTruthMeasurement",
    "create_real_world_validator",
    "run_comprehensive_validation",
    "validate_primary_sites",
    "validate_with_controls",
    
    # Environmental testing
    "EnvironmentalRobustnessValidator",
    "create_environmental_validator",
    "run_comprehensive_environmental_testing",
    
    # Species classification
    "SpeciesClassifier",
    "SpeciesClassificationResult",
    "BiomassEstimate",
    "create_species_classifier", 
    "run_species_classification",
    
    # Field survey integration
    "FieldSurveyRecord",
    "FieldDataIngestor",
    "SpeciesValidationAnalyzer",
    "FieldSurveyReporter",
    "create_field_data_ingestor",
    "create_validation_analyzer",
    "create_survey_reporter",
    
    # Temporal validation
    "TemporalValidator",
    "TemporalDataPoint",
    "SeasonalPattern",
    "TemporalValidationResult",
    "create_temporal_validator",
    "run_broughton_temporal_validation",
    "run_comprehensive_temporal_analysis",
]
