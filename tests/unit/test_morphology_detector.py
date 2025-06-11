"""Test morphological detection functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.kelpie_carbon_v1.processing.morphology_detector import (
    BladeFromdDetector,
    MorphologicalFeature,
    MorphologyDetectionResult,
    MorphologyDetector,
    MorphologyType,
    PneumatocystDetector,
    create_morphology_detector,
)


class TestMorphologyType:
    """Test MorphologyType enumeration."""

    def test_morphology_types(self):
        """Test that all expected morphology types are available."""
        expected_types = [
            "pneumatocyst",
            "blade",
            "frond",
            "stipe",
            "holdfast",
            "unknown",
        ]

        for type_name in expected_types:
            morph_type = MorphologyType(type_name)
            assert morph_type.value == type_name


class TestMorphologicalFeature:
    """Test MorphologicalFeature dataclass."""

    def test_morphological_feature_creation(self):
        """Test creating a morphological feature."""
        feature = MorphologicalFeature(
            feature_type=MorphologyType.PNEUMATOCYST,
            confidence=0.85,
            area=150.0,
            centroid=(100.0, 150.0),
            bounding_box=(90, 140, 110, 160),
            circularity=0.9,
            aspect_ratio=1.1,
            solidity=0.85,
            eccentricity=0.2,
            properties={"detection_method": "hough_circles"},
        )

        assert feature.feature_type == MorphologyType.PNEUMATOCYST
        assert feature.confidence == 0.85
        assert feature.area == 150.0
        assert feature.centroid == (100.0, 150.0)
        assert feature.properties["detection_method"] == "hough_circles"


class TestMorphologyDetectionResult:
    """Test MorphologyDetectionResult dataclass."""

    def test_detection_result_creation(self):
        """Test creating a morphology detection result."""
        features = [
            MorphologicalFeature(
                feature_type=MorphologyType.PNEUMATOCYST,
                confidence=0.8,
                area=100.0,
                centroid=(50.0, 60.0),
                bounding_box=(45, 55, 55, 65),
                circularity=0.9,
                aspect_ratio=1.0,
                solidity=0.9,
                eccentricity=0.1,
                properties={},
            )
        ]

        result = MorphologyDetectionResult(
            detected_features=features,
            pneumatocyst_count=1,
            blade_count=0,
            frond_count=0,
            total_pneumatocyst_area=100.0,
            total_blade_area=0.0,
            total_frond_area=0.0,
            morphology_confidence=0.8,
            species_indicators={"nereocystis_score": 0.7},
            processing_notes=["Detected 1 pneumatocyst"],
        )

        assert len(result.detected_features) == 1
        assert result.pneumatocyst_count == 1
        assert result.morphology_confidence == 0.8


class TestPneumatocystDetector:
    """Test PneumatocystDetector class."""

    def test_pneumatocyst_detector_init(self):
        """Test pneumatocyst detector initialization."""
        detector = PneumatocystDetector(min_size=30, max_size=400)

        assert detector.min_size == 30
        assert detector.max_size == 400
        assert detector.circularity_threshold == 0.6

    def test_detect_pneumatocysts_basic(self):
        """Test basic pneumatocyst detection."""
        detector = PneumatocystDetector()

        # Create test data
        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)
        kelp_mask[40:60, 40:60] = True  # Small kelp patch

        # Test detection
        pneumatocysts = detector.detect_pneumatocysts(rgb_image, kelp_mask)

        # Should return a list (may be empty with random data)
        assert isinstance(pneumatocysts, list)

        # All detections should be pneumatocysts
        for p in pneumatocysts:
            assert p.feature_type == MorphologyType.PNEUMATOCYST
            assert 0.0 <= p.confidence <= 1.0

    def test_detect_pneumatocysts_empty_mask(self):
        """Test pneumatocyst detection with empty mask."""
        detector = PneumatocystDetector()

        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)  # Empty mask

        pneumatocysts = detector.detect_pneumatocysts(rgb_image, kelp_mask)

        assert pneumatocysts == []

    def test_circularity_calculation(self):
        """Test circularity calculation method."""
        detector = PneumatocystDetector()

        # Mock regionprops object
        mock_prop = Mock()
        mock_prop.area = 100
        mock_prop.perimeter = 20

        circularity = detector._calculate_circularity(mock_prop)

        # Circularity = 4 * pi * area / perimeter^2
        expected = 4 * np.pi * 100 / (20 * 20)
        assert abs(circularity - expected) < 1e-6

    def test_confidence_calculation(self):
        """Test pneumatocyst confidence calculation."""
        detector = PneumatocystDetector()

        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.ones((100, 100), dtype=bool)

        detection = {
            "area": 100,
            "circularity": 0.8,
            "aspect_ratio": 1.0,
            "centroid": (50, 50),
        }

        confidence = detector._calculate_pneumatocyst_confidence(
            detection, rgb_image, kelp_mask
        )

        assert 0.0 <= confidence <= 1.0


class TestBladeFromdDetector:
    """Test BladeFromdDetector class."""

    def test_blade_frond_detector_init(self):
        """Test blade/frond detector initialization."""
        detector = BladeFromdDetector()

        assert detector.blade_aspect_ratio_min == 2.0
        assert detector.blade_solidity_min == 0.6
        assert detector.blade_area_range == (100, 2000)

    def test_detect_blades_and_fronds_basic(self):
        """Test basic blade/frond detection."""
        detector = BladeFromdDetector()

        # Create test data
        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)
        kelp_mask[30:70, 30:70] = True  # Medium kelp patch

        # Test detection
        features = detector.detect_blades_and_fronds(rgb_image, kelp_mask)

        # Should return a list
        assert isinstance(features, list)

        # All detections should be blades or fronds
        for f in features:
            assert f.feature_type in [MorphologyType.BLADE, MorphologyType.FROND]
            assert 0.0 <= f.confidence <= 1.0

    def test_classify_blade_or_frond(self):
        """Test blade vs frond classification logic."""
        detector = BladeFromdDetector()

        # Test blade characteristics (elongated, solid)
        blade_type, blade_conf = detector._classify_blade_or_frond(
            area=500,  # Within blade range
            aspect_ratio=3.0,  # High aspect ratio (elongated)
            solidity=0.8,  # High solidity
            eccentricity=0.6,  # Moderate eccentricity
            complexity=0.2,  # Low complexity (simple boundary)
        )

        assert blade_type == MorphologyType.BLADE
        assert blade_conf > 0.5

        # Test frond characteristics (complex, moderate ratio)
        frond_type, frond_conf = detector._classify_blade_or_frond(
            area=1000,  # Within frond range
            aspect_ratio=2.0,  # Moderate aspect ratio
            solidity=0.5,  # Lower solidity
            eccentricity=0.4,  # Low eccentricity
            complexity=0.6,  # High complexity (complex boundary)
        )

        assert frond_type == MorphologyType.FROND
        assert frond_conf > 0.5

    def test_boundary_complexity_calculation(self):
        """Test boundary complexity calculation."""
        detector = BladeFromdDetector()

        # Mock regionprops object
        mock_prop = Mock()
        mock_prop.area = 100
        mock_prop.perimeter = 40

        complexity = detector._calculate_boundary_complexity(mock_prop)

        # Should be normalized to 0-1 range
        assert 0.0 <= complexity <= 1.0


class TestMorphologyDetector:
    """Test main MorphologyDetector class."""

    def test_morphology_detector_init(self):
        """Test morphology detector initialization."""
        detector = MorphologyDetector()

        assert detector.pneumatocyst_detector is not None
        assert detector.blade_frond_detector is not None
        assert isinstance(detector.pneumatocyst_detector, PneumatocystDetector)
        assert isinstance(detector.blade_frond_detector, BladeFromdDetector)

    def test_analyze_morphology_basic(self):
        """Test basic morphological analysis."""
        detector = MorphologyDetector()

        # Create test data
        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)
        kelp_mask[25:75, 25:75] = True  # Large kelp patch

        # Test analysis
        result = detector.analyze_morphology(rgb_image, kelp_mask)

        # Verify result structure
        assert isinstance(result, MorphologyDetectionResult)
        assert isinstance(result.detected_features, list)
        assert result.pneumatocyst_count >= 0
        assert result.blade_count >= 0
        assert result.frond_count >= 0
        assert 0.0 <= result.morphology_confidence <= 1.0
        assert isinstance(result.species_indicators, dict)
        assert isinstance(result.processing_notes, list)

    def test_analyze_morphology_empty_mask(self):
        """Test morphological analysis with empty mask."""
        detector = MorphologyDetector()

        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)  # Empty mask

        result = detector.analyze_morphology(rgb_image, kelp_mask)

        # Should handle empty mask gracefully
        assert result.pneumatocyst_count == 0
        assert result.blade_count == 0
        assert result.frond_count == 0
        assert result.morphology_confidence == 0.0

    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        detector = MorphologyDetector()

        # Create test features
        features = [
            MorphologicalFeature(
                feature_type=MorphologyType.PNEUMATOCYST,
                confidence=0.8,
                area=100.0,
                centroid=(50.0, 50.0),
                bounding_box=(45, 45, 55, 55),
                circularity=0.9,
                aspect_ratio=1.0,
                solidity=0.9,
                eccentricity=0.1,
                properties={},
            ),
            MorphologicalFeature(
                feature_type=MorphologyType.BLADE,
                confidence=0.6,
                area=200.0,
                centroid=(75.0, 75.0),
                bounding_box=(70, 70, 80, 80),
                circularity=0.3,
                aspect_ratio=3.0,
                solidity=0.8,
                eccentricity=0.7,
                properties={},
            ),
        ]

        confidence = detector._calculate_overall_confidence(features)

        # Should be weighted average based on area
        assert 0.0 <= confidence <= 1.0

        # Empty features should return 0
        empty_confidence = detector._calculate_overall_confidence([])
        assert empty_confidence == 0.0

    def test_calculate_species_indicators(self):
        """Test species indicator calculation."""
        detector = MorphologyDetector()

        indicators = detector._calculate_species_indicators(
            pneumatocyst_count=2,
            blade_count=1,
            frond_count=3,
            pneumatocyst_area=200.0,
            blade_area=300.0,
            frond_area=500.0,
            total_kelp_area=1000.0,
        )

        assert "nereocystis_morphology_score" in indicators
        assert "macrocystis_morphology_score" in indicators
        assert "morphological_complexity" in indicators

        # All scores should be in 0-1 range
        for score in indicators.values():
            assert 0.0 <= score <= 1.0

    @patch("src.kelpie_carbon_v1.processing.morphology_detector.logger")
    def test_analyze_morphology_error_handling(self, mock_logger):
        """Test error handling in morphological analysis."""
        detector = MorphologyDetector()

        # Force an error by passing invalid data
        with patch.object(
            detector.pneumatocyst_detector,
            "detect_pneumatocysts",
            side_effect=Exception("Test error"),
        ):
            result = detector.analyze_morphology(
                np.array([[[1, 2, 3]]]), np.array([[True]])
            )

            # Should handle error gracefully
            assert result.pneumatocyst_count == 0
            assert result.blade_count == 0
            assert result.frond_count == 0
            assert "Morphological analysis error" in result.processing_notes[0]


class TestCreateMorphologyDetector:
    """Test factory function."""

    def test_create_morphology_detector(self):
        """Test creating morphology detector through factory function."""
        detector = create_morphology_detector()

        assert isinstance(detector, MorphologyDetector)
        assert detector.pneumatocyst_detector is not None
        assert detector.blade_frond_detector is not None


class TestIntegrationWithSpeciesClassifier:
    """Test integration with species classification system."""

    def test_morphology_integration(self):
        """Test that morphology detector integrates properly with species classifier."""
        from src.kelpie_carbon_v1.processing.species_classifier import SpeciesClassifier

        # Create classifier with morphology enabled
        classifier = SpeciesClassifier(enable_morphology=True)

        assert classifier.enable_morphology is True
        assert classifier.morphology_detector is not None
        assert isinstance(classifier.morphology_detector, MorphologyDetector)

    def test_morphology_features_in_classification(self):
        """Test that morphological features are properly extracted for classification."""
        from src.kelpie_carbon_v1.processing.species_classifier import SpeciesClassifier

        classifier = SpeciesClassifier(enable_morphology=True)

        # Create test data
        rgb_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        kelp_mask = np.zeros((100, 100), dtype=bool)
        kelp_mask[30:70, 30:70] = True

        # Extract morphological features
        features = classifier._extract_morphological_features(rgb_image, kelp_mask)

        # Should include advanced morphological features
        expected_advanced_features = [
            "pneumatocyst_count",
            "blade_count",
            "frond_count",
            "morphology_confidence",
        ]

        for feature in expected_advanced_features:
            assert feature in features
            assert isinstance(features[feature], float)
