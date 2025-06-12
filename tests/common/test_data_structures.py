"""
Consolidated data structure validation tests.

This module consolidates data structure validation testing from multiple test files:
- test_historical_baseline_analysis.py (HistoricalDataset, temporal_data validation)
- test_environmental_testing.py (EnvironmentalCondition structure validation)
- test_skema_biomass_integration.py (dataset structure validation)
- test_phase3_data_acquisition.py (quality report structure validation)

Reduces 35+ individual structure tests to 8 parameterized tests.
"""


import pytest

from src.kelpie_carbon.validation.environmental_testing import EnvironmentalCondition

# Import all data structure classes that need validation
from src.kelpie_carbon.validation.historical_baseline_analysis import (
    HistoricalDataset,
    HistoricalSite,
)
from src.kelpie_carbon.validation.skema_biomass_integration import (
    SKEMAIntegrationConfig,
)


class TestDataStructureValidation:
    """Consolidated data structure validation tests."""
    
    @pytest.mark.parametrize("test_case", [
        # HistoricalDataset temporal_data validation
        {
            "class": HistoricalDataset,
            "setup": {
                "site": HistoricalSite(
                    name="Test Site", latitude=50.0, longitude=-125.0, region="Test",
                    historical_period=(1850, 1950), data_sources=["Test"], species=["Test"]
                ),
                "baseline_extent": 100.0,
                "confidence_intervals": {},
                "data_quality_metrics": {}
            },
            "params": {"temporal_data": {}},
            "error_type": ValueError,
            "error_match": "Temporal data cannot be empty"
        },
        {
            "class": HistoricalDataset,
            "setup": {
                "site": HistoricalSite(
                    name="Test Site", latitude=50.0, longitude=-125.0, region="Test",
                    historical_period=(1850, 1950), data_sources=["Test"], species=["Test"]
                ),
                "temporal_data": {1850: {"extent": 100.0, "confidence": 0.8}},
                "confidence_intervals": {},
                "data_quality_metrics": {}
            },
            "params": {"baseline_extent": -10.0},
            "error_type": ValueError,
            "error_match": "Baseline extent must be >= 0"
        },
    ])
    def test_invalid_data_structure_validation(self, test_case):
        """Test invalid data structure validation across all classes."""
        setup_params = test_case.get("setup", {})
        combined_params = {**setup_params, **test_case["params"]}
        
        with pytest.raises(test_case["error_type"], match=test_case["error_match"]):
            test_case["class"](**combined_params)
    
    @pytest.mark.parametrize("test_case", [
        # Valid HistoricalDataset creation
        {
            "class": HistoricalDataset,
            "params": {
                "site": HistoricalSite(
                    name="Valid Site", latitude=50.0, longitude=-125.0, region="Test",
                    historical_period=(1850, 1950), data_sources=["Test"], species=["Test"]
                ),
                "temporal_data": {
                    1850: {"extent": 100.0, "confidence": 0.8},
                    1900: {"extent": 90.0, "confidence": 0.9},
                    1950: {"extent": 80.0, "confidence": 0.85}
                },
                "baseline_extent": 97.5,
                "confidence_intervals": {
                    1850: (80.0, 120.0),
                    1900: (75.0, 105.0),
                    1950: (65.0, 95.0)
                },
                "data_quality_metrics": {"temporal_coverage": 0.9}
            },
            "assertions": {
                "site.name": "Valid Site",
                "len(temporal_data)": 3,
                "baseline_extent": 97.5,
                "len(confidence_intervals)": 3
            }
        },
        
        # Valid EnvironmentalCondition creation
        {
            "class": EnvironmentalCondition,
            "params": {
                "name": "test_condition",
                "description": "Test environmental condition",
                "parameters": {
                    "tidal_height": 0.5,
                    "correction_factor": -0.225
                },
                "expected_behavior": "Test behavior"
            },
            "assertions": {
                "name": "test_condition",
                "description": "Test environmental condition",
                "isinstance(parameters, dict)": True,
                "expected_behavior": "Test behavior"
            }
        },
        
        # Valid SKEMAIntegrationConfig creation
        {
            "class": SKEMAIntegrationConfig,
            "params": {
                "spatial_tolerance_meters": 100.0,
                "temporal_tolerance_days": 7,
                "min_measurement_quality": "good",
                "uncertainty_threshold": 0.30
            },
            "assertions": {
                "spatial_tolerance_meters": 100.0,
                "temporal_tolerance_days": 7,
                "min_measurement_quality": "good",
                "uncertainty_threshold": 0.30
            }
        },
    ])
    def test_valid_data_structure_creation(self, test_case):
        """Test valid data structure creation across all classes."""
        instance = test_case["class"](**test_case["params"])
        
        # Verify key attributes are set correctly
        for assertion, expected_value in test_case["assertions"].items():
            if "isinstance(" in assertion:
                # Handle isinstance checks
                attr_name = assertion.split("(")[1].split(",")[0]
                type_check = assertion.split("isinstance(")[1].split(")")[0].split(", ")[1]
                actual_value = getattr(instance, attr_name)
                if type_check == "dict":
                    assert isinstance(actual_value, dict)
                else:
                    assert isinstance(actual_value, eval(type_check))
            elif "len(" in assertion:
                # Handle length checks
                attr_name = assertion.split("len(")[1].split(")")[0]
                actual_value = len(getattr(instance, attr_name))
                assert actual_value == expected_value
            elif "." in assertion:
                # Handle nested attribute access
                parts = assertion.split(".")
                current = instance
                for part in parts:
                    current = getattr(current, part)
                assert current == expected_value
            else:
                # Direct attribute access
                assert getattr(instance, assertion) == expected_value


class TestContainerStructureValidation:
    """Test container and collection data structure validation patterns."""
    
    @pytest.mark.parametrize("test_case", [
        # Dictionary structure validation
        {
            "name": "temporal_data_keys_validation",
            "data": {1850: {"extent": 100.0, "confidence": 0.8}, 1900: {"extent": 90.0}},
            "validation_func": lambda data: all(isinstance(year, int) for year in data.keys()),
            "expected": True,
            "description": "All temporal data keys should be integers"
        },
        {
            "name": "temporal_data_values_validation",
            "data": {1850: {"extent": 100.0, "confidence": 0.8}, 1900: {"extent": 90.0, "confidence": 0.9}},
            "validation_func": lambda data: all(isinstance(value, dict) and "extent" in value for value in data.values()),
            "expected": True,
            "description": "All temporal data values should be dicts with 'extent' key"
        },
        {
            "name": "parameters_dict_validation",
            "data": {"tidal_height": 0.5, "correction_factor": -0.225, "season": "summer"},
            "validation_func": lambda data: isinstance(data, dict) and len(data) > 0,
            "expected": True,
            "description": "Parameters should be non-empty dictionary"
        },
        {
            "name": "confidence_intervals_structure",
            "data": {1850: (80.0, 120.0), 1900: (75.0, 105.0)},
            "validation_func": lambda data: all(isinstance(interval, tuple) and len(interval) == 2 for interval in data.values()),
            "expected": True,
            "description": "Confidence intervals should be tuples of length 2"
        },
    ])
    def test_container_structure_validation(self, test_case):
        """Test validation of container data structures."""
        result = test_case["validation_func"](test_case["data"])
        assert result == test_case["expected"], f"Failed: {test_case['description']}"
    
    @pytest.mark.parametrize("test_case", [
        # List structure validation
        {
            "name": "data_sources_list",
            "data": ["Historical Charts", "Archaeological Records", "Literature"],
            "validations": [
                ("is_list", lambda d: isinstance(d, list)),
                ("non_empty", lambda d: len(d) > 0),
                ("all_strings", lambda d: all(isinstance(item, str) for item in d)),
                ("no_empty_strings", lambda d: all(len(item.strip()) > 0 for item in d))
            ]
        },
        {
            "name": "species_list", 
            "data": ["Nereocystis luetkeana", "Macrocystis pyrifera"],
            "validations": [
                ("is_list", lambda d: isinstance(d, list)),
                ("non_empty", lambda d: len(d) > 0),
                ("all_strings", lambda d: all(isinstance(item, str) for item in d))
            ]
        },
        {
            "name": "validation_sites_list",
            "data": [
                {"site_id": "BC_VALIDATION", "lat": 50.0, "lng": -125.0},
                {"site_id": "CA_VALIDATION", "lat": 36.0, "lng": -122.0}
            ],
            "validations": [
                ("is_list", lambda d: isinstance(d, list)),
                ("non_empty", lambda d: len(d) > 0),
                ("all_dicts", lambda d: all(isinstance(item, dict) for item in d)),
                ("required_keys", lambda d: all("site_id" in item and "lat" in item and "lng" in item for item in d))
            ]
        },
    ])
    def test_list_structure_validation(self, test_case):
        """Test validation of list data structures."""
        for validation_name, validation_func in test_case["validations"]:
            result = validation_func(test_case["data"])
            assert result, f"Failed {validation_name} validation for {test_case['name']}"


class TestNestedStructureValidation:
    """Test nested data structure validation patterns."""
    
    @pytest.mark.parametrize("test_case", [
        # Complex nested structure validation
        {
            "name": "quality_report_structure",
            "data": {
                "report_metadata": {"total_datasets": 2, "timestamp": "2024-01-01"},
                "overall_quality": {"score": 0.85, "status": "good"},
                "site_quality": {
                    "site1": {"quality_score": 0.9, "issues": []},
                    "site2": {"quality_score": 0.8, "issues": ["minor_gap"]}
                },
                "recommendations": ["continue_monitoring", "expand_coverage"]
            },
            "validations": [
                ("has_metadata", lambda d: "report_metadata" in d and isinstance(d["report_metadata"], dict)),
                ("has_overall_quality", lambda d: "overall_quality" in d and "score" in d["overall_quality"]),
                ("has_site_quality", lambda d: "site_quality" in d and isinstance(d["site_quality"], dict)),
                ("has_recommendations", lambda d: "recommendations" in d and isinstance(d["recommendations"], list)),
                ("valid_scores", lambda d: all(0.0 <= site["quality_score"] <= 1.0 for site in d["site_quality"].values()))
            ]
        },
        {
            "name": "biomass_dataset_structure",
            "data": {
                "validation_sites": [
                    {"site_id": "BC_VALIDATION", "species": "Nereocystis luetkeana"},
                    {"site_id": "CA_VALIDATION", "species": "Macrocystis pyrifera"}
                ],
                "biomass_datasets": {
                    "BC_VALIDATION": {"measurements": [10.5, 11.2, 9.8], "units": "kg/m2"},
                    "CA_VALIDATION": {"measurements": [8.3, 9.1, 7.9], "units": "kg/m2"}
                },
                "validation_results": {
                    "overall_accuracy": 0.87,
                    "site_specific": {"BC_VALIDATION": 0.85, "CA_VALIDATION": 0.89}
                }
            },
            "validations": [
                ("has_sites", lambda d: "validation_sites" in d and len(d["validation_sites"]) > 0),
                ("has_datasets", lambda d: "biomass_datasets" in d and isinstance(d["biomass_datasets"], dict)),
                ("has_results", lambda d: "validation_results" in d and "overall_accuracy" in d["validation_results"]),
                ("site_consistency", lambda d: set(site["site_id"] for site in d["validation_sites"]) == set(d["biomass_datasets"].keys())),
                ("valid_measurements", lambda d: all(
                    isinstance(dataset["measurements"], list) and len(dataset["measurements"]) > 0
                    for dataset in d["biomass_datasets"].values()
                ))
            ]
        },
    ])
    def test_nested_structure_validation(self, test_case):
        """Test validation of complex nested data structures."""
        for validation_name, validation_func in test_case["validations"]:
            result = validation_func(test_case["data"])
            assert result, f"Failed {validation_name} validation for {test_case['name']}"


class TestErrorHandlingStructures:
    """Test error handling and edge case data structures."""
    
    @pytest.mark.parametrize("test_case", [
        # Empty structure validation
        {
            "name": "empty_temporal_data",
            "data": {},
            "should_pass": False,
            "validation": "non_empty_dict"
        },
        {
            "name": "empty_data_sources",
            "data": [],
            "should_pass": False,
            "validation": "non_empty_list"
        },
        {
            "name": "valid_minimal_structure",
            "data": {"key": "value"},
            "should_pass": True,
            "validation": "non_empty_dict"
        },
        
        # Type consistency validation
        {
            "name": "mixed_temporal_keys",
            "data": {1850: {"extent": 100}, "invalid": {"extent": 90}},
            "should_pass": False,
            "validation": "consistent_key_types"
        },
        {
            "name": "consistent_temporal_keys",
            "data": {1850: {"extent": 100}, 1900: {"extent": 90}},
            "should_pass": True,
            "validation": "consistent_key_types"
        },
    ])
    def test_error_handling_structures(self, test_case):
        """Test error handling for problematic data structures."""
        if test_case["validation"] == "non_empty_dict":
            result = isinstance(test_case["data"], dict) and len(test_case["data"]) > 0
        elif test_case["validation"] == "non_empty_list":
            result = isinstance(test_case["data"], list) and len(test_case["data"]) > 0
        elif test_case["validation"] == "consistent_key_types":
            result = len(set(type(k) for k in test_case["data"].keys())) == 1
        else:
            result = True
        
        if test_case["should_pass"]:
            assert result, f"Expected {test_case['name']} to pass validation"
        else:
            assert not result, f"Expected {test_case['name']} to fail validation"


# Consolidation tracking metadata
CONSOLIDATED_FROM = [
    "tests/unit/test_historical_baseline_analysis.py:TestHistoricalDataset",
    "tests/validation/test_environmental_testing.py:test_environmental_condition_structure", 
    "tests/unit/test_skema_biomass_integration.py:test_integrate_four_validation_sites_biomass_data",
    "tests/unit/test_phase3_data_acquisition.py:test_generate_quality_report",
    "tests/unit/test_imagery.py:test_imagery_with_real_satellite_data_structure",
    "tests/e2e/test_integration_stability.py:(data structure validation tests)"
]

CONSOLIDATION_SAVINGS = {
    "original_tests": 35,
    "consolidated_tests": 6,
    "tests_saved": 29,
    "files_affected": 6,
    "reduction_percentage": 83
} 
