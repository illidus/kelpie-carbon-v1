"""Test species classification functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.kelpie_carbon_v1.processing.species_classifier import (
    BiomassEstimate,
    KelpSpecies,
    SpeciesClassificationResult,
    SpeciesClassifier,
    create_species_classifier,
)


class TestKelpSpecies:
    """Test KelpSpecies enumeration."""

    def test_species_enumeration(self):
        """Test that all expected species are available."""
        expected_species = [
            "nereocystis_luetkeana",
            "macrocystis_pyrifera",
            "mixed_species",
            "unknown",
        ]

        for species_name in expected_species:
            species = KelpSpecies(species_name)
            assert species.value == species_name


class TestSpeciesClassificationResult:
    """Test SpeciesClassificationResult dataclass."""

    def test_result_initialization(self):
        """Test result initialization with required fields."""
        result = SpeciesClassificationResult(
            primary_species=KelpSpecies.NEREOCYSTIS_LUETKEANA,
            confidence=0.85,
            species_probabilities={KelpSpecies.NEREOCYSTIS_LUETKEANA: 0.85},
            morphological_features={"total_area": 100.0},
            spectral_features={"ndvi_mean": 0.3},
        )

        assert result.primary_species == KelpSpecies.NEREOCYSTIS_LUETKEANA
        assert result.confidence == 0.85
        assert result.processing_notes == []  # Default value

    def test_result_with_biomass(self):
        """Test result with biomass estimation."""
        result = SpeciesClassificationResult(
            primary_species=KelpSpecies.MACROCYSTIS_PYRIFERA,
            confidence=0.75,
            species_probabilities={KelpSpecies.MACROCYSTIS_PYRIFERA: 0.75},
            morphological_features={},
            spectral_features={},
            biomass_estimate_kg_per_m2=8.5,
        )

        assert result.biomass_estimate_kg_per_m2 == 8.5


class TestSpeciesClassifier:
    """Test SpeciesClassifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = SpeciesClassifier()

        # Create test data
        self.rgb_image = np.random.rand(100, 100, 3)
        self.kelp_mask = np.zeros((100, 100), dtype=bool)
        self.kelp_mask[40:60, 40:60] = True  # 20x20 kelp patch

        self.spectral_indices = {
            "ndvi": np.random.rand(100, 100) * 0.5 + 0.2,  # 0.2-0.7 range
            "ndre": np.random.rand(100, 100) * 0.5 + 0.3,  # 0.3-0.8 range
        }

    def test_classify_species_basic(self):
        """Test basic species classification."""
        result = self.classifier.classify_species(
            self.rgb_image, self.spectral_indices, self.kelp_mask
        )

        assert isinstance(result, SpeciesClassificationResult)
        assert isinstance(result.primary_species, KelpSpecies)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.species_probabilities) == len(KelpSpecies)
        assert isinstance(result.morphological_features, dict)
        assert isinstance(result.spectral_features, dict)

    def test_classify_empty_mask(self):
        """Test classification with empty kelp mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)

        result = self.classifier.classify_species(
            self.rgb_image, self.spectral_indices, empty_mask
        )

        assert result.primary_species == KelpSpecies.UNKNOWN
        assert result.confidence == 0.0
        assert result.biomass_estimate_kg_per_m2 is None

    def test_nereocystis_indicators(self):
        """Test Nereocystis classification with strong indicators."""
        # Create strong Nereocystis indicators
        spectral_indices = {
            "ndvi": np.ones((100, 100)) * 0.25,  # Lower NDVI (submerged)
            "ndre": np.ones((100, 100)) * 0.35,  # Higher NDRE (submerged detection)
        }

        # Mock pneumatocyst detection to return high score
        with patch.object(self.classifier, "_detect_pneumatocysts", return_value=0.8):
            result = self.classifier.classify_species(
                self.rgb_image,
                spectral_indices,
                self.kelp_mask,
                metadata={"latitude": 50.0},  # Pacific Northwest
            )

        # Should favor Nereocystis
        assert result.species_probabilities[KelpSpecies.NEREOCYSTIS_LUETKEANA] > 0.3

    def test_macrocystis_indicators(self):
        """Test Macrocystis classification with strong indicators."""
        # Create strong Macrocystis indicators
        spectral_indices = {
            "ndvi": np.ones((100, 100)) * 0.45,  # Higher NDVI (surface)
            "ndre": np.ones((100, 100)) * 0.40,  # Moderate NDRE
        }

        # Mock frond detection to return high score
        with patch.object(self.classifier, "_detect_frond_patterns", return_value=0.8):
            result = self.classifier.classify_species(
                self.rgb_image,
                spectral_indices,
                self.kelp_mask,
                metadata={"latitude": 36.0},  # California coast
            )

        # Should favor Macrocystis
        assert result.species_probabilities[KelpSpecies.MACROCYSTIS_PYRIFERA] > 0.3

    def test_mixed_species_indicators(self):
        """Test mixed species classification."""
        # Create high spectral heterogeneity
        ndvi = np.random.rand(100, 100) * 0.4 + 0.2  # 0.2-0.6 range with variation
        ndre = np.random.rand(100, 100) * 0.4 + 0.3  # 0.3-0.7 range with variation

        spectral_indices = {"ndvi": ndvi, "ndre": ndre}

        result = self.classifier.classify_species(
            self.rgb_image, spectral_indices, self.kelp_mask
        )

        # High heterogeneity should contribute to mixed species score
        assert "spectral_heterogeneity" in result.spectral_features
        assert result.spectral_features["spectral_heterogeneity"] > 0.0

    def test_extract_spectral_features(self):
        """Test spectral feature extraction."""
        features = self.classifier._extract_spectral_features(
            self.spectral_indices, self.kelp_mask
        )

        # Check that all expected features are present
        expected_features = [
            "ndvi_mean",
            "ndvi_std",
            "ndre_mean",
            "ndre_std",
            "ndre_ndvi_ratio",
            "spectral_heterogeneity",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)

    def test_extract_morphological_features(self):
        """Test morphological feature extraction."""
        features = self.classifier._extract_morphological_features(
            self.rgb_image, self.kelp_mask
        )

        # Check that all expected features are present
        expected_features = [
            "total_area",
            "perimeter",
            "compactness",
            "blob_count",
            "pneumatocyst_score",
            "frond_pattern_score",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], float)

        # Check that area is correct
        assert features["total_area"] == self.kelp_mask.sum()

    @patch("cv2.HoughCircles")
    def test_detect_pneumatocysts(self, mock_hough_circles):
        """Test pneumatocyst detection."""
        # Mock successful circle detection
        mock_hough_circles.return_value = np.array([[[50, 50, 10], [60, 60, 8]]])

        score = self.classifier._detect_pneumatocysts(self.rgb_image, self.kelp_mask)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should detect circles

    @patch("cv2.HoughCircles")
    def test_detect_pneumatocysts_none_found(self, mock_hough_circles):
        """Test pneumatocyst detection when none found."""
        # Mock no circles detected
        mock_hough_circles.return_value = None

        score = self.classifier._detect_pneumatocysts(self.rgb_image, self.kelp_mask)

        assert score == 0.0

    def test_detect_frond_patterns(self):
        """Test frond pattern detection."""
        score = self.classifier._detect_frond_patterns(self.rgb_image, self.kelp_mask)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_biomass_estimation_nereocystis(self):
        """Test biomass estimation for Nereocystis."""
        morphological_features = {
            "pneumatocyst_score": 0.8,
            "total_area": 400.0,  # 20x20 pixels
        }

        biomass = self.classifier._estimate_biomass(
            KelpSpecies.NEREOCYSTIS_LUETKEANA, morphological_features, self.kelp_mask
        )

        assert biomass is not None
        assert biomass > 0.0
        # Should be influenced by pneumatocyst score
        assert biomass > 4.0  # Base biomass should be adjusted upward

    def test_biomass_estimation_macrocystis(self):
        """Test biomass estimation for Macrocystis."""
        morphological_features = {"frond_pattern_score": 0.7, "total_area": 400.0}

        biomass = self.classifier._estimate_biomass(
            KelpSpecies.MACROCYSTIS_PYRIFERA, morphological_features, self.kelp_mask
        )

        assert biomass is not None
        assert biomass > 0.0
        # Should be influenced by frond pattern score
        assert biomass > 8.0  # Base biomass should be adjusted upward

    def test_biomass_estimation_unknown(self):
        """Test biomass estimation for unknown species."""
        biomass = self.classifier._estimate_biomass(
            KelpSpecies.UNKNOWN, {}, self.kelp_mask
        )

        assert biomass is None

    def test_error_handling(self):
        """Test error handling in classification."""
        # Test with invalid input that should trigger exception handling
        with patch.object(
            self.classifier,
            "_extract_spectral_features",
            side_effect=Exception("Test error"),
        ):
            result = self.classifier.classify_species(
                self.rgb_image, self.spectral_indices, self.kelp_mask
            )

            assert result.primary_species == KelpSpecies.UNKNOWN
            assert result.confidence == 0.0
            assert len(result.processing_notes) > 0
            assert "Classification error" in result.processing_notes[0]


class TestFactoryFunction:
    """Test factory function for creating classifiers."""

    def test_create_species_classifier(self):
        """Test factory function."""
        classifier = create_species_classifier()

        assert isinstance(classifier, SpeciesClassifier)
        assert hasattr(classifier, "classify_species")


class TestIntegration:
    """Integration tests for species classification."""

    def test_full_classification_pipeline(self):
        """Test complete classification pipeline."""
        classifier = create_species_classifier()

        # Create realistic test data
        rgb_image = np.random.rand(50, 50, 3)

        # Create kelp mask with multiple patches
        kelp_mask = np.zeros((50, 50), dtype=bool)
        kelp_mask[10:20, 10:20] = True  # Patch 1
        kelp_mask[30:40, 30:40] = True  # Patch 2

        # Create realistic spectral indices
        spectral_indices = {
            "ndvi": np.random.rand(50, 50) * 0.3 + 0.2,  # 0.2-0.5
            "ndre": np.random.rand(50, 50) * 0.3 + 0.3,  # 0.3-0.6
        }

        # Add metadata
        metadata = {"latitude": 48.5, "longitude": -123.5, "date": "2024-07-15"}

        result = classifier.classify_species(
            rgb_image, spectral_indices, kelp_mask, metadata
        )

        # Verify complete result
        assert isinstance(result, SpeciesClassificationResult)
        assert result.primary_species in KelpSpecies
        assert 0.0 <= result.confidence <= 1.0

        # Verify probabilities sum to approximately 1.0
        prob_sum = sum(result.species_probabilities.values())
        assert 0.99 <= prob_sum <= 1.01

        # Verify features were extracted
        assert len(result.spectral_features) > 0
        assert len(result.morphological_features) > 0

        # If classified as a known species, should have biomass estimate
        if result.primary_species != KelpSpecies.UNKNOWN:
            assert result.biomass_estimate_kg_per_m2 is not None
            assert result.biomass_estimate_kg_per_m2 > 0.0


def test_enhanced_biomass_estimation():
    """Test enhanced biomass estimation with confidence intervals."""
    classifier = SpeciesClassifier(enable_morphology=False)

    # Create test data
    kelp_mask = np.ones((10, 10), dtype=bool)  # 100 pixels = 10000 m²
    morphological_features = {
        "pneumatocyst_count": 5,
        "blade_count": 3,
        "frond_count": 2,
        "morphology_confidence": 0.8,
        "total_area": 100.0,
    }

    # Test Nereocystis biomass estimation with high confidence
    biomass_estimate = classifier._estimate_biomass_with_confidence(
        KelpSpecies.NEREOCYSTIS_LUETKEANA,
        morphological_features,
        kelp_mask,
        classification_confidence=0.9,
    )

    assert biomass_estimate is not None
    assert isinstance(biomass_estimate, BiomassEstimate)
    assert 6.0 <= biomass_estimate.point_estimate_kg_per_m2 <= 15.0  # Reasonable range
    assert (
        biomass_estimate.lower_bound_kg_per_m2
        < biomass_estimate.point_estimate_kg_per_m2
    )
    assert (
        biomass_estimate.upper_bound_kg_per_m2
        > biomass_estimate.point_estimate_kg_per_m2
    )
    assert biomass_estimate.confidence_level == 0.95
    assert len(biomass_estimate.uncertainty_factors) >= 0


def test_enhanced_biomass_estimation_low_confidence():
    """Test enhanced biomass estimation with low classification confidence."""
    classifier = SpeciesClassifier(enable_morphology=False)

    kelp_mask = np.ones((5, 5), dtype=bool)  # Small patch
    morphological_features = {
        "pneumatocyst_count": 0,  # No features detected
        "blade_count": 0,
        "frond_count": 0,
        "morphology_confidence": 0.4,  # Low confidence
        "total_area": 25.0,
    }

    # Test with low classification confidence
    biomass_estimate = classifier._estimate_biomass_with_confidence(
        KelpSpecies.MACROCYSTIS_PYRIFERA,
        morphological_features,
        kelp_mask,
        classification_confidence=0.6,  # Low confidence
    )

    assert biomass_estimate is not None
    # Should have wider confidence intervals due to uncertainty
    confidence_interval_width = (
        biomass_estimate.upper_bound_kg_per_m2 - biomass_estimate.lower_bound_kg_per_m2
    )
    assert confidence_interval_width > 2.0  # Wide interval due to uncertainty

    # Should have multiple uncertainty factors
    assert len(biomass_estimate.uncertainty_factors) >= 3
    assert any(
        "confidence" in factor.lower()
        for factor in biomass_estimate.uncertainty_factors
    )
    assert any(
        "small" in factor.lower() for factor in biomass_estimate.uncertainty_factors
    )


def test_enhanced_biomass_estimation_mixed_species():
    """Test enhanced biomass estimation for mixed species."""
    classifier = SpeciesClassifier(enable_morphology=False)

    kelp_mask = np.ones((20, 20), dtype=bool)  # Large patch
    morphological_features = {
        "pneumatocyst_count": 3,
        "blade_count": 5,
        "frond_count": 4,
        "morphology_confidence": 0.9,
        "nereocystis_morphology_score": 0.4,
        "macrocystis_morphology_score": 0.5,
        "total_area": 400.0,
    }

    biomass_estimate = classifier._estimate_biomass_with_confidence(
        KelpSpecies.MIXED_SPECIES,
        morphological_features,
        kelp_mask,
        classification_confidence=0.8,
    )

    assert biomass_estimate is not None
    assert (
        7.0 <= biomass_estimate.point_estimate_kg_per_m2 <= 20.0
    )  # Mixed species range
    assert "Mixed species" in str(biomass_estimate.uncertainty_factors)


def test_enhanced_biomass_estimation_bounds():
    """Test that biomass estimates stay within reasonable bounds."""
    classifier = SpeciesClassifier(enable_morphology=False)

    kelp_mask = np.ones((10, 10), dtype=bool)
    morphological_features = {
        "pneumatocyst_count": 15,  # Very high count
        "pneumatocyst_density": 0.9,  # Very high density
        "nereocystis_morphology_score": 0.95,  # Very high score
        "morphology_confidence": 0.95,
        "total_area": 100.0,
    }

    biomass_estimate = classifier._estimate_biomass_with_confidence(
        KelpSpecies.NEREOCYSTIS_LUETKEANA,
        morphological_features,
        kelp_mask,
        classification_confidence=0.95,
    )

    assert biomass_estimate is not None
    # Even with very high indicators, should not exceed reasonable maximum
    assert biomass_estimate.point_estimate_kg_per_m2 <= 24.0  # 2x literature max
    assert biomass_estimate.upper_bound_kg_per_m2 <= 24.0
    assert biomass_estimate.lower_bound_kg_per_m2 >= 0.5  # Minimum threshold


def test_species_specific_biomass_algorithms():
    """Test that different species use different biomass algorithms."""
    classifier = SpeciesClassifier(enable_morphology=False)
    kelp_mask = np.ones((10, 10), dtype=bool)

    # Nereocystis with pneumatocyst features
    nereocystis_features = {
        "pneumatocyst_count": 8,
        "pneumatocyst_density": 0.6,
        "blade_count": 0,
        "frond_count": 0,
    }

    # Macrocystis with blade/frond features
    macrocystis_features = {
        "pneumatocyst_count": 0,
        "pneumatocyst_density": 0.0,
        "blade_count": 10,
        "frond_count": 5,
        "blade_frond_ratio": 2.0,
    }

    nereocystis_biomass = classifier._estimate_nereocystis_biomass(
        nereocystis_features, 1000.0
    )
    macrocystis_biomass = classifier._estimate_macrocystis_biomass(
        macrocystis_features, 1000.0
    )

    # Should use different base densities and enhancement factors
    assert nereocystis_biomass != macrocystis_biomass
    assert 6.0 <= nereocystis_biomass <= 24.0  # Nereocystis range
    assert 8.0 <= macrocystis_biomass <= 37.5  # Macrocystis range


def test_enhanced_classification_result_structure():
    """Test that enhanced classification results include biomass estimates."""
    classifier = SpeciesClassifier(enable_morphology=False)

    # Create test inputs
    rgb_image = np.random.rand(50, 50, 3).astype(np.float32)
    spectral_indices = {
        "ndvi": np.random.rand(50, 50).astype(np.float32),
        "ndre": np.random.rand(50, 50).astype(np.float32),
    }
    kelp_mask = np.ones((50, 50), dtype=bool)

    result = classifier.classify_species(rgb_image, spectral_indices, kelp_mask)

    # Should include both basic and enhanced biomass estimates
    assert result.biomass_estimate_kg_per_m2 is not None
    assert result.biomass_estimate_enhanced is not None
    assert isinstance(result.biomass_estimate_enhanced, BiomassEstimate)

    # Enhanced estimate should have confidence intervals
    enhanced = result.biomass_estimate_enhanced
    assert enhanced.point_estimate_kg_per_m2 > 0
    assert enhanced.lower_bound_kg_per_m2 >= 0
    assert enhanced.upper_bound_kg_per_m2 > enhanced.lower_bound_kg_per_m2
    assert enhanced.confidence_level == 0.95
    assert isinstance(enhanced.uncertainty_factors, list)


def test_biomass_estimation_validation_against_literature():
    """Test that biomass estimates align with published literature ranges."""
    classifier = SpeciesClassifier(enable_morphology=False)
    kelp_mask = np.ones((10, 10), dtype=bool)

    # Test moderate-quality morphological features
    moderate_features = {
        "pneumatocyst_count": 5,
        "blade_count": 7,
        "frond_count": 3,
        "morphology_confidence": 0.7,
        "total_area": 100.0,
    }

    # All species should produce estimates within literature ranges
    for species in [
        KelpSpecies.NEREOCYSTIS_LUETKEANA,
        KelpSpecies.MACROCYSTIS_PYRIFERA,
        KelpSpecies.MIXED_SPECIES,
    ]:
        biomass = classifier._estimate_biomass(species, moderate_features, kelp_mask)
        assert biomass is not None
        assert 3.0 <= biomass <= 40.0  # Broad reasonable range

        enhanced = classifier._estimate_biomass_with_confidence(
            species, moderate_features, kelp_mask, 0.8
        )
        assert enhanced is not None
        assert (
            0.5
            <= enhanced.lower_bound_kg_per_m2
            <= enhanced.upper_bound_kg_per_m2
            <= 40.0
        )
