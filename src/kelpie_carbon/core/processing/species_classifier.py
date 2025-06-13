"""Species-level classification for kelp detection.

This module implements automated multi-species kelp classification based on:
- Spectral signature analysis
- Morphological feature detection
- Species-specific biomass estimation

Addresses SKEMA Phase 4: Species-Level Detection gaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from .morphology_detector import MorphologyDetector

logger = logging.getLogger(__name__)


class KelpSpecies(Enum):
    """Enumeration of kelp species that can be classified."""

    NEREOCYSTIS_LUETKEANA = "nereocystis_luetkeana"  # Bull kelp
    MACROCYSTIS_PYRIFERA = "macrocystis_pyrifera"  # Giant kelp
    MIXED_SPECIES = "mixed_species"  # Multiple species present
    UNKNOWN = "unknown"  # Cannot determine species


@dataclass
class BiomassEstimate:
    """Biomass estimation with confidence intervals."""

    point_estimate_kg_per_m2: float
    lower_bound_kg_per_m2: float
    upper_bound_kg_per_m2: float
    confidence_level: float = 0.95  # 95% confidence interval
    uncertainty_factors: list[str] = None

    def __post_init__(self):
        """Initialize uncertainty_factors list if None."""
        if self.uncertainty_factors is None:
            self.uncertainty_factors = []


@dataclass
class SpeciesClassificationResult:
    """Result of species classification analysis."""

    primary_species: KelpSpecies
    confidence: float
    species_probabilities: dict[KelpSpecies, float]
    morphological_features: dict[str, float]
    spectral_features: dict[str, float]
    biomass_estimate_kg_per_m2: float | None = None
    biomass_estimate_enhanced: BiomassEstimate | None = None
    processing_notes: list[str] = None

    def __post_init__(self):
        """Initialize processing_notes list if None."""
        if self.processing_notes is None:
            self.processing_notes = []


class SpeciesClassifier:
    """Multi-species kelp classifier using spectral and morphological analysis."""

    def __init__(self, enable_morphology: bool = True):
        """Initialize the species classifier.

        Args:
            enable_morphology: Whether to enable advanced morphological analysis

        """
        self.logger = logging.getLogger(__name__)
        self.enable_morphology = enable_morphology

        # Initialize morphology detector if enabled
        if self.enable_morphology:
            self.morphology_detector = MorphologyDetector()
        else:
            self.morphology_detector = None

    def classify_species(
        self,
        rgb_image: np.ndarray,
        spectral_indices: dict[str, np.ndarray],
        kelp_mask: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> SpeciesClassificationResult:
        """Classify kelp species in the given image.

        Args:
            rgb_image: RGB image array [H, W, 3]
            spectral_indices: Dictionary of calculated spectral indices
            kelp_mask: Boolean mask indicating kelp presence [H, W]
            metadata: Optional metadata (location, date, etc.)

        Returns:
            SpeciesClassificationResult with classification details

        """
        if metadata is None:
            metadata = {}

        try:
            # Extract spectral features
            spectral_features = self._extract_spectral_features(
                spectral_indices, kelp_mask
            )

            # Extract morphological features
            morphological_features = self._extract_morphological_features(
                rgb_image, kelp_mask
            )

            # Classify based on combined features
            species_probabilities = self._classify_from_features(
                spectral_features, morphological_features, metadata
            )

            # Determine primary species and confidence
            primary_species = max(species_probabilities, key=species_probabilities.get)
            confidence = species_probabilities[primary_species]

            # Special case: empty mask should have 0 confidence even if classified as UNKNOWN
            if kelp_mask.sum() == 0:
                confidence = 0.0

            # Estimate biomass if possible
            biomass_estimate = self._estimate_biomass(
                primary_species, morphological_features, kelp_mask
            )

            # Enhanced biomass estimation with confidence intervals
            biomass_enhanced = self._estimate_biomass_with_confidence(
                primary_species, morphological_features, kelp_mask, confidence
            )

            return SpeciesClassificationResult(
                primary_species=primary_species,
                confidence=confidence,
                species_probabilities=species_probabilities,
                morphological_features=morphological_features,
                spectral_features=spectral_features,
                biomass_estimate_kg_per_m2=biomass_estimate,
                biomass_estimate_enhanced=biomass_enhanced,
                processing_notes=[],
            )

        except Exception as e:
            self.logger.error(f"Error in species classification: {e}")
            return SpeciesClassificationResult(
                primary_species=KelpSpecies.UNKNOWN,
                confidence=0.0,
                species_probabilities=dict.fromkeys(KelpSpecies, 0.0),
                morphological_features={},
                spectral_features={},
                processing_notes=[f"Classification error: {str(e)}"],
            )

    def _extract_spectral_features(
        self, spectral_indices: dict[str, np.ndarray], kelp_mask: np.ndarray
    ) -> dict[str, float]:
        """Extract spectral features from kelp areas."""
        features = {}

        # Extract mean values for key indices within kelp areas
        for index_name, index_array in spectral_indices.items():
            if kelp_mask.sum() > 0:
                kelp_values = index_array[kelp_mask]
                features[f"{index_name}_mean"] = float(np.mean(kelp_values))
                features[f"{index_name}_std"] = float(np.std(kelp_values))
            else:
                features[f"{index_name}_mean"] = 0.0
                features[f"{index_name}_std"] = 0.0

        # Species-specific spectral ratios
        if "ndvi" in spectral_indices and "ndre" in spectral_indices:
            ndvi_values = (
                spectral_indices["ndvi"][kelp_mask]
                if kelp_mask.sum() > 0
                else np.array([0])
            )
            ndre_values = (
                spectral_indices["ndre"][kelp_mask]
                if kelp_mask.sum() > 0
                else np.array([0])
            )

            # NDRE/NDVI ratio (higher for submerged kelp like Nereocystis)
            features["ndre_ndvi_ratio"] = float(
                np.mean(ndre_values) / (np.mean(ndvi_values) + 1e-8)
            )

            # Spectral heterogeneity (indicator of mixed species)
            features["spectral_heterogeneity"] = float(
                np.std(ndvi_values) + np.std(ndre_values)
            )

        return features

    def _extract_morphological_features(
        self, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> dict[str, float]:
        """Extract morphological features from kelp areas."""
        features = {}

        if kelp_mask.sum() == 0:
            return {
                "total_area": 0.0,
                "perimeter": 0.0,
                "compactness": 0.0,
                "blob_count": 0.0,
                "pneumatocyst_score": 0.0,
                "frond_pattern_score": 0.0,
                "pneumatocyst_count": 0.0,
                "blade_count": 0.0,
                "frond_count": 0.0,
                "morphology_confidence": 0.0,
            }

        # Use advanced morphological analysis if available
        if self.enable_morphology and self.morphology_detector:
            try:
                morphology_result = self.morphology_detector.analyze_morphology(
                    rgb_image, kelp_mask
                )

                # Extract advanced features
                features["pneumatocyst_count"] = float(
                    morphology_result.pneumatocyst_count
                )
                features["blade_count"] = float(morphology_result.blade_count)
                features["frond_count"] = float(morphology_result.frond_count)
                features["total_pneumatocyst_area"] = float(
                    morphology_result.total_pneumatocyst_area
                )
                features["total_blade_area"] = float(morphology_result.total_blade_area)
                features["total_frond_area"] = float(morphology_result.total_frond_area)
                features["morphology_confidence"] = float(
                    morphology_result.morphology_confidence
                )

                # Extract species indicators
                for indicator, value in morphology_result.species_indicators.items():
                    features[indicator] = float(value)

                # Calculate derived features
                total_area = float(kelp_mask.sum())
                features["pneumatocyst_density"] = (
                    features["pneumatocyst_count"] / (total_area / 1000)
                    if total_area > 0
                    else 0.0
                )
                features["blade_frond_ratio"] = (
                    features["blade_count"] / (features["frond_count"] + 1)
                    if features["frond_count"] > 0
                    else features["blade_count"]
                )

            except Exception as e:
                self.logger.warning(f"Advanced morphology analysis failed: {e}")
                # Fallback to basic analysis
                self.enable_morphology = False

        # Basic morphological analysis (always computed as fallback)
        mask_uint8 = kelp_mask.astype(np.uint8) * 255

        # Basic area and perimeter
        features["total_area"] = float(kelp_mask.sum())
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            total_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
            features["perimeter"] = float(total_perimeter)
            features["compactness"] = float(
                features["total_area"] / (total_perimeter**2 + 1e-8)
            )
            features["blob_count"] = float(len(contours))
        else:
            features["perimeter"] = 0.0
            features["compactness"] = 0.0
            features["blob_count"] = 0.0

        # Legacy morphological indicators (always computed for compatibility)
        if not self.enable_morphology or "pneumatocyst_score" not in features:
            features["pneumatocyst_score"] = self._detect_pneumatocysts(
                rgb_image, kelp_mask
            )
            features["frond_pattern_score"] = self._detect_frond_patterns(
                rgb_image, kelp_mask
            )
            # Only set these if not already set by advanced analysis
            if "pneumatocyst_count" not in features:
                features["pneumatocyst_count"] = 0.0
            if "blade_count" not in features:
                features["blade_count"] = 0.0
            if "frond_count" not in features:
                features["frond_count"] = 0.0
            if "morphology_confidence" not in features:
                features["morphology_confidence"] = 0.0

        return features

    def _detect_pneumatocysts(
        self, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> float:
        """Detect pneumatocysts (gas-filled bladders) characteristic of Nereocystis."""
        if kelp_mask.sum() == 0:
            return 0.0

        # Ensure image is in correct format for OpenCV
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        # Convert to grayscale for blob detection
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        masked_gray = gray.copy()
        masked_gray[~kelp_mask] = 0

        # Detect circular/elliptical bright regions (pneumatocysts)
        circles = cv2.HoughCircles(
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50,
        )

        if circles is not None:
            num_circles = len(circles[0])
            kelp_area = kelp_mask.sum()
            pneumatocyst_score = min(1.0, num_circles / (kelp_area / 1000 + 1))
        else:
            pneumatocyst_score = 0.0

        return float(pneumatocyst_score)

    def _detect_frond_patterns(
        self, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> float:
        """Detect frond patterns characteristic of Macrocystis."""
        if kelp_mask.sum() == 0:
            return 0.0

        # Ensure image is in correct format for OpenCV
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        masked_gray = gray.copy()
        masked_gray[~kelp_mask] = 0

        # Detect linear/branching patterns using morphological operations
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

        opened_h = cv2.morphologyEx(masked_gray, cv2.MORPH_OPEN, kernel_horizontal)
        opened_v = cv2.morphologyEx(masked_gray, cv2.MORPH_OPEN, kernel_vertical)

        linear_features = np.maximum(opened_h, opened_v)
        linear_area = np.sum(linear_features > 0)
        kelp_area = kelp_mask.sum()
        frond_score = min(1.0, linear_area / (kelp_area + 1e-8))

        return float(frond_score)

    def _classify_from_features(
        self,
        spectral_features: dict[str, float],
        morphological_features: dict[str, float],
        metadata: dict[str, Any],
    ) -> dict[KelpSpecies, float]:
        """Classify species based on extracted features."""
        # Initialize probabilities
        probabilities = dict.fromkeys(KelpSpecies, 0.0)

        # Handle empty mask case - no kelp area to classify
        if morphological_features.get("total_area", 0) == 0:
            probabilities[KelpSpecies.UNKNOWN] = 1.0
            return probabilities

        # Nereocystis luetkeana indicators:
        nereocystis_score = 0.0

        # Spectral indicators for Nereocystis
        if "ndre_ndvi_ratio" in spectral_features:
            ndre_ratio = spectral_features["ndre_ndvi_ratio"]
            if ndre_ratio > 1.1:  # Higher NDRE/NDVI suggests submerged kelp
                nereocystis_score += 0.2

        # Advanced morphological indicators for Nereocystis
        if "pneumatocyst_count" in morphological_features:
            pneumatocyst_count = morphological_features["pneumatocyst_count"]
            if pneumatocyst_count > 0:
                nereocystis_score += min(
                    0.4, pneumatocyst_count * 0.1
                )  # Strong indicator

        if "nereocystis_morphology_score" in morphological_features:
            nereocystis_score += (
                morphological_features["nereocystis_morphology_score"] * 0.3
            )

        if "pneumatocyst_density" in morphological_features:
            density = morphological_features["pneumatocyst_density"]
            if density > 0.5:  # High pneumatocyst density
                nereocystis_score += 0.2

        # Legacy morphological indicators (fallback)
        if (
            "pneumatocyst_score" in morphological_features
            and "pneumatocyst_count" not in morphological_features
        ):
            nereocystis_score += morphological_features["pneumatocyst_score"] * 0.3

        # Macrocystis pyrifera indicators:
        macrocystis_score = 0.0

        # Spectral indicators for Macrocystis
        if "ndvi_mean" in spectral_features:
            ndvi_mean = spectral_features["ndvi_mean"]
            if ndvi_mean > 0.3:  # Strong surface signal
                macrocystis_score += 0.2

        # Advanced morphological indicators for Macrocystis
        if (
            "blade_count" in morphological_features
            and "frond_count" in morphological_features
        ):
            blade_count = morphological_features["blade_count"]
            frond_count = morphological_features["frond_count"]

            if blade_count > 0 or frond_count > 0:
                macrocystis_score += min(
                    0.4, (blade_count + frond_count) * 0.05
                )  # Strong indicator

        if "macrocystis_morphology_score" in morphological_features:
            macrocystis_score += (
                morphological_features["macrocystis_morphology_score"] * 0.3
            )

        if "blade_frond_ratio" in morphological_features:
            ratio = morphological_features["blade_frond_ratio"]
            if ratio > 1.0:  # More blades than fronds typical of Macrocystis
                macrocystis_score += 0.2

        # Legacy morphological indicators (fallback)
        if (
            "frond_pattern_score" in morphological_features
            and "blade_count" not in morphological_features
        ):
            macrocystis_score += morphological_features["frond_pattern_score"] * 0.3

        # Mixed species indicators:
        mixed_score = 0.0

        # Spectral heterogeneity suggests mixed species
        if "spectral_heterogeneity" in spectral_features:
            heterogeneity = spectral_features["spectral_heterogeneity"]
            if heterogeneity > 0.15:  # High heterogeneity suggests mixed species
                mixed_score += 0.3

        # Both pneumatocysts and blades/fronds present
        pneumatocyst_present = morphological_features.get("pneumatocyst_count", 0) > 0
        blade_frond_present = (
            morphological_features.get("blade_count", 0)
            + morphological_features.get("frond_count", 0)
        ) > 0

        if pneumatocyst_present and blade_frond_present:
            mixed_score += 0.4  # Strong indicator of mixed species

        # High morphological complexity may indicate mixed species
        if "morphological_complexity" in morphological_features:
            complexity = morphological_features["morphological_complexity"]
            if complexity > 0.7:
                mixed_score += 0.2

        # Location-based priors
        if "latitude" in metadata:
            lat = metadata["latitude"]
            if lat > 48.0:  # Far Pacific Northwest
                nereocystis_score += 0.1
            elif lat < 38.0:  # Southern California
                macrocystis_score += 0.1
            elif 40.0 <= lat <= 46.0:  # Transition zone
                mixed_score += 0.1

        # Normalize scores to probabilities
        total_score = nereocystis_score + macrocystis_score + mixed_score

        if total_score > 0:
            probabilities[KelpSpecies.NEREOCYSTIS_LUETKEANA] = (
                nereocystis_score / total_score
            )
            probabilities[KelpSpecies.MACROCYSTIS_PYRIFERA] = (
                macrocystis_score / total_score
            )
            probabilities[KelpSpecies.MIXED_SPECIES] = mixed_score / total_score
            probabilities[KelpSpecies.UNKNOWN] = 0.0
        else:
            # No indicators found, classify as unknown with low confidence
            probabilities[KelpSpecies.UNKNOWN] = 1.0

        return probabilities

    def _estimate_biomass(
        self,
        species: KelpSpecies,
        morphological_features: dict[str, float],
        kelp_mask: np.ndarray,
    ) -> float | None:
        """Estimate biomass based on species and morphological features."""
        if species == KelpSpecies.UNKNOWN or kelp_mask.sum() == 0:
            return None

        # Get basic area (assuming 10m pixel resolution)
        area_m2 = float(kelp_mask.sum()) * 100  # 10m x 10m pixels

        # Enhanced species-specific biomass estimation using morphological features
        if species == KelpSpecies.NEREOCYSTIS_LUETKEANA:
            return self._estimate_nereocystis_biomass(morphological_features, area_m2)
        elif species == KelpSpecies.MACROCYSTIS_PYRIFERA:
            return self._estimate_macrocystis_biomass(morphological_features, area_m2)
        elif species == KelpSpecies.MIXED_SPECIES:
            return self._estimate_mixed_species_biomass(morphological_features, area_m2)
        else:
            return None

    def _estimate_nereocystis_biomass(
        self, morphological_features: dict[str, float], area_m2: float
    ) -> float:
        """Enhanced biomass estimation for Nereocystis luetkeana (Bull kelp).

        Based on research showing typical biomass range: 600-1200 kg/ha (6-12 kg/m²).
        Morphological factors significantly affect biomass density.
        """
        # Base biomass density from literature (kg/m²)
        base_density_min = 6.0  # 600 kg/ha
        base_density_max = 12.0  # 1200 kg/ha
        base_density = (base_density_min + base_density_max) / 2  # 9.0 kg/m²

        # Morphological enhancement factors
        density_multiplier = 1.0

        # Pneumatocyst density factor (most important for Nereocystis)
        pneumatocyst_count = morphological_features.get("pneumatocyst_count", 0)
        pneumatocyst_density = morphological_features.get("pneumatocyst_density", 0.0)

        if pneumatocyst_count > 0:
            # High pneumatocyst count indicates mature, dense kelp forest
            if pneumatocyst_count >= 10:
                density_multiplier *= 1.3  # Mature forest
            elif pneumatocyst_count >= 5:
                density_multiplier *= 1.15  # Medium density
            else:
                density_multiplier *= 1.05  # Low density

        if pneumatocyst_density > 0.7:
            density_multiplier *= 1.2  # Very dense pneumatocyst field
        elif pneumatocyst_density > 0.4:
            density_multiplier *= 1.1  # Medium density

        # Advanced morphological factors
        nereocystis_morphology_score = morphological_features.get(
            "nereocystis_morphology_score", 0.5
        )
        if nereocystis_morphology_score > 0.8:
            density_multiplier *= 1.25  # Strong Nereocystis characteristics
        elif nereocystis_morphology_score > 0.6:
            density_multiplier *= 1.1

        # Size and coverage factors
        feature_area = morphological_features.get("feature_area", area_m2)
        coverage_ratio = min(1.0, feature_area / area_m2) if area_m2 > 0 else 1.0

        # Large continuous coverage indicates healthy forest
        if coverage_ratio > 0.8:
            density_multiplier *= 1.15
        elif coverage_ratio < 0.3:
            density_multiplier *= 0.85  # Fragmented coverage

        # Calculate final biomass density
        final_density = base_density * density_multiplier

        # Apply reasonable bounds (50% to 200% of literature range)
        final_density = max(
            base_density_min * 0.5, min(final_density, base_density_max * 2.0)
        )

        return float(final_density)

    def _estimate_macrocystis_biomass(
        self, morphological_features: dict[str, float], area_m2: float
    ) -> float:
        """Enhanced biomass estimation for Macrocystis pyrifera (Giant kelp).

        Based on research showing typical biomass range: 800-1500 kg/ha (8-15 kg/m²).
        Blade and frond characteristics are key biomass indicators.
        """
        # Base biomass density from literature (kg/m²)
        base_density_min = 8.0  # 800 kg/ha
        base_density_max = 15.0  # 1500 kg/ha
        base_density = (base_density_min + base_density_max) / 2  # 11.5 kg/m²

        # Morphological enhancement factors
        density_multiplier = 1.0

        # Blade and frond density factors (key for Macrocystis)
        blade_count = morphological_features.get("blade_count", 0)
        frond_count = morphological_features.get("frond_count", 0)
        total_structures = blade_count + frond_count

        if total_structures > 0:
            # High blade/frond count indicates dense, mature kelp
            if total_structures >= 15:
                density_multiplier *= 1.4  # Very dense forest
            elif total_structures >= 8:
                density_multiplier *= 1.2  # Medium-high density
            elif total_structures >= 3:
                density_multiplier *= 1.1  # Medium density
            else:
                density_multiplier *= 1.0  # Low density

        # Blade to frond ratio optimization
        blade_frond_ratio = morphological_features.get("blade_frond_ratio", 1.0)
        if blade_frond_ratio > 2.0:
            # High blade ratio indicates optimal growing conditions
            density_multiplier *= 1.15
        elif blade_frond_ratio > 1.5:
            density_multiplier *= 1.08

        # Advanced morphological factors
        macrocystis_morphology_score = morphological_features.get(
            "macrocystis_morphology_score", 0.5
        )
        if macrocystis_morphology_score > 0.8:
            density_multiplier *= 1.3  # Strong Macrocystis characteristics
        elif macrocystis_morphology_score > 0.6:
            density_multiplier *= 1.15

        # Structural complexity factor
        morphological_complexity = morphological_features.get(
            "morphological_complexity", 0.5
        )
        if morphological_complexity > 0.7:
            # High complexity indicates mature, multi-layered canopy
            density_multiplier *= 1.2
        elif morphological_complexity > 0.5:
            density_multiplier *= 1.1

        # Size and coverage factors
        feature_area = morphological_features.get("feature_area", area_m2)
        coverage_ratio = min(1.0, feature_area / area_m2) if area_m2 > 0 else 1.0

        # Continuous large patches are more productive
        if coverage_ratio > 0.9:
            density_multiplier *= 1.2  # Dense continuous forest
        elif coverage_ratio > 0.7:
            density_multiplier *= 1.1
        elif coverage_ratio < 0.4:
            density_multiplier *= 0.9  # Fragmented patches

        # Calculate final biomass density
        final_density = base_density * density_multiplier

        # Apply reasonable bounds (40% to 250% of literature range)
        final_density = max(
            base_density_min * 0.4, min(final_density, base_density_max * 2.5)
        )

        return float(final_density)

    def _estimate_mixed_species_biomass(
        self, morphological_features: dict[str, float], area_m2: float
    ) -> float:
        """Enhanced biomass estimation for mixed species kelp forests.

        Uses intermediate values and combined morphological indicators.
        """
        # Base biomass density for mixed species (kg/m²)
        # Average of Nereocystis (6-12) and Macrocystis (8-15) ranges
        base_density_min = 7.0  # 700 kg/ha
        base_density_max = 13.5  # 1350 kg/ha
        base_density = (base_density_min + base_density_max) / 2  # 10.25 kg/m²

        # Morphological enhancement factors
        density_multiplier = 1.0

        # Combined morphological indicators
        pneumatocyst_count = morphological_features.get("pneumatocyst_count", 0)
        blade_count = morphological_features.get("blade_count", 0)
        frond_count = morphological_features.get("frond_count", 0)

        total_features = pneumatocyst_count + blade_count + frond_count

        if total_features > 0:
            # Diversity of features indicates healthy mixed forest
            if total_features >= 12:
                density_multiplier *= 1.25  # Highly diverse forest
            elif total_features >= 6:
                density_multiplier *= 1.15  # Medium diversity
            else:
                density_multiplier *= 1.05  # Low diversity

        # Species balance factor
        nereocystis_score = morphological_features.get(
            "nereocystis_morphology_score", 0.0
        )
        macrocystis_score = morphological_features.get(
            "macrocystis_morphology_score", 0.0
        )

        if nereocystis_score > 0.3 and macrocystis_score > 0.3:
            # Good balance of both species
            density_multiplier *= 1.2
        elif abs(nereocystis_score - macrocystis_score) < 0.2:
            # Balanced mixed forest
            density_multiplier *= 1.1

        # Morphological complexity (mixed species often show high complexity)
        morphological_complexity = morphological_features.get(
            "morphological_complexity", 0.5
        )
        if morphological_complexity > 0.8:
            density_multiplier *= 1.18  # Very complex mixed system
        elif morphological_complexity > 0.6:
            density_multiplier *= 1.1

        # Coverage and size factors
        feature_area = morphological_features.get("feature_area", area_m2)
        coverage_ratio = min(1.0, feature_area / area_m2) if area_m2 > 0 else 1.0

        if coverage_ratio > 0.8:
            density_multiplier *= 1.15  # Dense mixed forest
        elif coverage_ratio < 0.5:
            density_multiplier *= 0.9  # Sparse mixed patches

        # Calculate final biomass density
        final_density = base_density * density_multiplier

        # Apply reasonable bounds (50% to 200% of base range)
        final_density = max(
            base_density_min * 0.5, min(final_density, base_density_max * 2.0)
        )

        return float(final_density)

    def _estimate_biomass_with_confidence(
        self,
        species: KelpSpecies,
        morphological_features: dict[str, float],
        kelp_mask: np.ndarray,
        classification_confidence: float,
    ) -> BiomassEstimate | None:
        """Estimate biomass with confidence intervals and uncertainty quantification.

        Args:
            species: Classified kelp species
            morphological_features: Extracted morphological features
            kelp_mask: Binary mask of kelp areas
            classification_confidence: Confidence in species classification

        Returns:
            BiomassEstimate with point estimate and confidence intervals

        """
        if species == KelpSpecies.UNKNOWN or kelp_mask.sum() == 0:
            return None

        # Get point estimate
        point_estimate = self._estimate_biomass(
            species, morphological_features, kelp_mask
        )
        if point_estimate is None:
            return None

        # Calculate uncertainty factors
        uncertainty_factors = []
        uncertainty_multiplier = 1.0

        # Species classification uncertainty
        if classification_confidence < 0.9:
            uncertainty_multiplier *= (
                1.3  # Higher uncertainty for low-confidence classifications
            )
            uncertainty_factors.append("Low species classification confidence")
        elif classification_confidence < 0.7:
            uncertainty_multiplier *= 1.5
            uncertainty_factors.append("Very low species classification confidence")

        # Morphological feature quality assessment
        morphology_confidence = morphological_features.get("morphology_confidence", 0.0)
        if morphology_confidence < 0.6:
            uncertainty_multiplier *= 1.2
            uncertainty_factors.append("Low morphological feature confidence")

        # Data availability uncertainty
        total_features = (
            morphological_features.get("pneumatocyst_count", 0)
            + morphological_features.get("blade_count", 0)
            + morphological_features.get("frond_count", 0)
        )
        if total_features == 0:
            uncertainty_multiplier *= 1.4
            uncertainty_factors.append("No advanced morphological features detected")
        elif total_features < 3:
            uncertainty_multiplier *= 1.2
            uncertainty_factors.append("Limited morphological features detected")

        # Coverage area uncertainty
        area_m2 = float(kelp_mask.sum()) * 100  # 10m x 10m pixels
        if area_m2 < 1000:  # Small patches have higher uncertainty
            uncertainty_multiplier *= 1.3
            uncertainty_factors.append("Small kelp patch size")
        elif area_m2 < 500:
            uncertainty_multiplier *= 1.5
            uncertainty_factors.append("Very small kelp patch size")

        # Species-specific uncertainty factors
        if species == KelpSpecies.MIXED_SPECIES:
            uncertainty_multiplier *= 1.25  # Mixed species inherently more uncertain
            uncertainty_factors.append("Mixed species biomass estimation")

        # Calculate confidence intervals (assuming log-normal distribution)
        # Standard error as percentage of point estimate
        base_std_error = 0.15  # 15% base uncertainty
        adjusted_std_error = base_std_error * uncertainty_multiplier

        # 95% confidence interval (±1.96 standard errors)
        confidence_factor = 1.96 * adjusted_std_error

        lower_bound = point_estimate * (1 - confidence_factor)
        upper_bound = point_estimate * (1 + confidence_factor)

        # Apply species-specific bounds based on literature ranges
        if species == KelpSpecies.NEREOCYSTIS_LUETKEANA:
            # Nereocystis: 6-12 kg/m² typical range
            lower_bound = max(lower_bound, 3.0)  # Absolute minimum
            upper_bound = min(upper_bound, 24.0)  # Absolute maximum (2x literature max)
        elif species == KelpSpecies.MACROCYSTIS_PYRIFERA:
            # Macrocystis: 8-15 kg/m² typical range
            lower_bound = max(lower_bound, 3.2)  # 40% of literature min
            upper_bound = min(upper_bound, 37.5)  # 250% of literature max
        elif species == KelpSpecies.MIXED_SPECIES:
            # Mixed: 7-13.5 kg/m² typical range
            lower_bound = max(lower_bound, 3.5)  # 50% of literature min
            upper_bound = min(upper_bound, 27.0)  # 200% of literature max

        # Ensure bounds are reasonable
        lower_bound = max(0.5, lower_bound)  # Never below 0.5 kg/m²
        upper_bound = max(lower_bound + 0.5, upper_bound)  # Ensure minimum interval

        return BiomassEstimate(
            point_estimate_kg_per_m2=point_estimate,
            lower_bound_kg_per_m2=lower_bound,
            upper_bound_kg_per_m2=upper_bound,
            confidence_level=0.95,
            uncertainty_factors=uncertainty_factors,
        )


def create_species_classifier() -> SpeciesClassifier:
    """Create and return a configured species classifier instance."""
    return SpeciesClassifier()


def run_species_classification(
    rgb_image: np.ndarray,
    spectral_indices: dict[str, np.ndarray],
    kelp_mask: np.ndarray,
    metadata: dict[str, Any] | None = None,
    enable_morphology: bool = True,
) -> SpeciesClassificationResult:
    """Run species classification on kelp areas.

    Args:
        rgb_image: RGB satellite image array
        spectral_indices: Dictionary of spectral indices
        kelp_mask: Binary mask of kelp areas
        metadata: Optional metadata dictionary
        enable_morphology: Whether to enable morphological analysis

    Returns:
        SpeciesClassificationResult with species identification and biomass estimates

    """
    classifier = SpeciesClassifier(enable_morphology=enable_morphology)
    return classifier.classify_species(rgb_image, spectral_indices, kelp_mask, metadata)
