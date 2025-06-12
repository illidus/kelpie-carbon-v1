"""Advanced morphological detection for kelp species identification.

This module implements specialized morphological detection algorithms for:
- Pneumatocyst detection in Nereocystis luetkeana (Bull kelp)
- Blade vs. frond differentiation in Macrocystis pyrifera (Giant kelp)
- Advanced shape analysis and feature extraction

Addresses SKEMA Phase 4: Species-Level Detection through morphological analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


class MorphologyType(Enum):
    """Types of morphological features that can be detected."""

    PNEUMATOCYST = "pneumatocyst"  # Gas-filled bladders (Nereocystis)
    BLADE = "blade"  # Flat leaf-like structures (Macrocystis)
    FROND = "frond"  # Branching structures (Macrocystis)
    STIPE = "stipe"  # Stem-like structures
    HOLDFAST = "holdfast"  # Anchor structures
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class MorphologicalFeature:
    """Represents a detected morphological feature."""

    feature_type: MorphologyType
    confidence: float
    area: float
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    circularity: float
    aspect_ratio: float
    solidity: float
    eccentricity: float
    properties: dict[str, Any]


@dataclass
class MorphologyDetectionResult:
    """Results from morphological detection analysis."""

    detected_features: list[MorphologicalFeature]
    pneumatocyst_count: int
    blade_count: int
    frond_count: int
    total_pneumatocyst_area: float
    total_blade_area: float
    total_frond_area: float
    morphology_confidence: float
    species_indicators: dict[str, float]
    processing_notes: list[str]


class PneumatocystDetector:
    """Specialized detector for pneumatocysts (gas-filled bladders) in Nereocystis."""

    def __init__(self, min_size: int = 20, max_size: int = 500):
        """Initialize pneumatocyst detector.

        Args:
            min_size: Minimum pneumatocyst size in pixels
            max_size: Maximum pneumatocyst size in pixels
        """
        self.min_size = min_size
        self.max_size = max_size

        # Pneumatocyst characteristics (from literature)
        self.circularity_threshold = 0.6  # Pneumatocysts are roughly circular
        self.intensity_threshold = 0.7  # Often bright due to gas content
        self.aspect_ratio_range = (0.7, 1.4)  # Nearly circular to slightly oval

    def detect_pneumatocysts(
        self, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> list[MorphologicalFeature]:
        """Detect pneumatocysts in kelp image.

        Args:
            rgb_image: RGB image array [H, W, 3]
            kelp_mask: Boolean mask indicating kelp areas

        Returns:
            List of detected pneumatocyst features
        """
        try:
            # Ensure proper image format
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

            # Apply kelp mask
            masked_gray = gray.copy()
            masked_gray[~kelp_mask] = 0

            # Enhance pneumatocyst detection through preprocessing
            enhanced = self._enhance_pneumatocysts(masked_gray)

            # Detect circular structures using HoughCircles
            circles = self._detect_circular_structures(enhanced, kelp_mask)

            # Detect blob-like structures using connected components
            blobs = self._detect_blob_structures(enhanced, kelp_mask)

            # Combine and filter detections
            pneumatocysts = self._combine_and_filter_detections(
                circles, blobs, rgb_image, kelp_mask
            )

            logger.info(f"Detected {len(pneumatocysts)} pneumatocyst candidates")
            return pneumatocysts

        except Exception as e:
            logger.error(f"Error in pneumatocyst detection: {e}")
            return []

    def _enhance_pneumatocysts(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhance pneumatocysts through preprocessing."""
        # Gaussian blur to reduce noise
        blurred = gaussian(gray_image, sigma=1.0)

        # Enhance circular structures using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        enhanced = cv2.morphologyEx(blurred.astype(np.uint8), cv2.MORPH_TOPHAT, kernel)

        # Additional enhancement for bright structures
        enhanced = cv2.add(blurred.astype(np.uint8), enhanced)

        return enhanced

    def _detect_circular_structures(
        self, enhanced_image: np.ndarray, kelp_mask: np.ndarray
    ) -> list[dict]:
        """Detect circular structures using HoughCircles."""
        circles = []

        try:
            # Apply HoughCircles with parameters optimized for pneumatocysts
            detected_circles = cv2.HoughCircles(
                enhanced_image,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(self.min_size * 0.8),
                param1=50,
                param2=30,
                minRadius=self.min_size // 2,
                maxRadius=self.max_size // 2,
            )

            if detected_circles is not None:
                detected_circles = np.round(detected_circles[0, :]).astype("int")

                for x, y, r in detected_circles:
                    # Check if circle center is within kelp mask
                    if (
                        0 <= y < kelp_mask.shape[0]
                        and 0 <= x < kelp_mask.shape[1]
                        and kelp_mask[y, x]
                    ):
                        circles.append(
                            {
                                "centroid": (y, x),
                                "radius": r,
                                "area": np.pi * r * r,
                                "detection_method": "hough_circles",
                            }
                        )

        except Exception as e:
            logger.warning(f"HoughCircles detection failed: {e}")

        return circles

    def _detect_blob_structures(
        self, enhanced_image: np.ndarray, kelp_mask: np.ndarray
    ) -> list[dict]:
        """Detect blob-like structures using connected components."""
        blobs = []

        try:
            # Threshold image to identify bright regions
            threshold_val = threshold_otsu(enhanced_image[kelp_mask])
            binary = enhanced_image > threshold_val
            binary = binary & kelp_mask

            # Clean up binary image
            binary = morphology.remove_small_objects(binary, min_size=self.min_size)
            binary = morphology.remove_small_holes(
                binary, area_threshold=self.min_size // 4
            )

            # Label connected components
            labeled_image = measure.label(binary)
            props = measure.regionprops(labeled_image)

            for prop in props:
                # Filter based on size
                if self.min_size <= prop.area <= self.max_size:
                    # Calculate morphological properties
                    circularity = self._calculate_circularity(prop)

                    # Filter based on circularity (pneumatocysts are roughly circular)
                    if circularity >= self.circularity_threshold:
                        blobs.append(
                            {
                                "centroid": prop.centroid,
                                "area": prop.area,
                                "circularity": circularity,
                                "aspect_ratio": prop.major_axis_length
                                / prop.minor_axis_length
                                if prop.minor_axis_length > 0
                                else 1.0,
                                "solidity": prop.solidity,
                                "eccentricity": prop.eccentricity,
                                "bounding_box": prop.bbox,
                                "detection_method": "blob_analysis",
                            }
                        )

        except Exception as e:
            logger.warning(f"Blob detection failed: {e}")

        return blobs

    def _calculate_circularity(self, prop) -> float:
        """Calculate circularity of a region."""
        if prop.perimeter == 0:
            return 0.0
        return 4 * np.pi * prop.area / (prop.perimeter * prop.perimeter)

    def _combine_and_filter_detections(
        self,
        circles: list[dict],
        blobs: list[dict],
        rgb_image: np.ndarray,
        kelp_mask: np.ndarray,
    ) -> list[MorphologicalFeature]:
        """Combine and filter pneumatocyst detections."""
        pneumatocysts = []

        # Process circle detections
        for circle in circles:
            confidence = self._calculate_pneumatocyst_confidence(
                circle, rgb_image, kelp_mask
            )

            if confidence > 0.5:  # Confidence threshold
                feature = MorphologicalFeature(
                    feature_type=MorphologyType.PNEUMATOCYST,
                    confidence=confidence,
                    area=circle["area"],
                    centroid=circle["centroid"],
                    bounding_box=self._get_circle_bbox(circle, rgb_image.shape),
                    circularity=0.9,  # Assumed high for circles
                    aspect_ratio=1.0,  # Perfect circle
                    solidity=0.9,  # Assumed high for circles
                    eccentricity=0.1,  # Low for circles
                    properties={"detection_method": circle["detection_method"]},
                )
                pneumatocysts.append(feature)

        # Process blob detections
        for blob in blobs:
            confidence = self._calculate_pneumatocyst_confidence(
                blob, rgb_image, kelp_mask
            )

            if confidence > 0.4:  # Slightly lower threshold for blobs
                feature = MorphologicalFeature(
                    feature_type=MorphologyType.PNEUMATOCYST,
                    confidence=confidence,
                    area=blob["area"],
                    centroid=blob["centroid"],
                    bounding_box=blob["bounding_box"],
                    circularity=blob["circularity"],
                    aspect_ratio=blob["aspect_ratio"],
                    solidity=blob["solidity"],
                    eccentricity=blob["eccentricity"],
                    properties={"detection_method": blob["detection_method"]},
                )
                pneumatocysts.append(feature)

        # Remove overlapping detections
        pneumatocysts = self._remove_overlapping_detections(pneumatocysts)

        return pneumatocysts

    def _calculate_pneumatocyst_confidence(
        self, detection: dict, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> float:
        """Calculate confidence score for pneumatocyst detection."""
        confidence_factors = []

        # Size factor (moderate size is typical)
        area = detection["area"]
        if 50 <= area <= 200:
            confidence_factors.append(1.0)
        elif 20 <= area <= 300:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)

        # Circularity factor (if available)
        if "circularity" in detection:
            circ = detection["circularity"]
            if circ >= 0.8:
                confidence_factors.append(1.0)
            elif circ >= 0.6:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.8)  # Default for circles

        # Aspect ratio factor (if available)
        if "aspect_ratio" in detection:
            ar = detection["aspect_ratio"]
            if 0.7 <= ar <= 1.4:
                confidence_factors.append(1.0)
            elif 0.5 <= ar <= 2.0:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.9)  # Default for circles

        # Location factor (should be within kelp area)
        centroid = detection["centroid"]
        y, x = int(centroid[0]), int(centroid[1])
        if (
            0 <= y < kelp_mask.shape[0]
            and 0 <= x < kelp_mask.shape[1]
            and kelp_mask[y, x]
        ):
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.2)

        # Return weighted average
        return np.mean(confidence_factors)

    def _get_circle_bbox(
        self, circle: dict, image_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Get bounding box for a circle detection."""
        y, x = circle["centroid"]
        r = circle["radius"]

        min_row = max(0, int(y - r))
        max_row = min(image_shape[0], int(y + r))
        min_col = max(0, int(x - r))
        max_col = min(image_shape[1], int(x + r))

        return (min_row, min_col, max_row, max_col)

    def _remove_overlapping_detections(
        self, detections: list[MorphologicalFeature]
    ) -> list[MorphologicalFeature]:
        """Remove overlapping pneumatocyst detections."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        filtered = []
        for detection in sorted_detections:
            is_overlapping = False

            for existing in filtered:
                # Calculate distance between centroids
                dist = np.sqrt(
                    (detection.centroid[0] - existing.centroid[0]) ** 2
                    + (detection.centroid[1] - existing.centroid[1]) ** 2
                )

                # If too close, consider overlapping
                min_separation = np.sqrt(detection.area + existing.area) / 2
                if dist < min_separation:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered.append(detection)

        return filtered


class BladeFromdDetector:
    """Specialized detector for blades and fronds in Macrocystis."""

    def __init__(self):
        """Initialize blade/frond detector."""
        # Blade characteristics (flat, elongated)
        self.blade_aspect_ratio_min = 2.0  # Blades are elongated
        self.blade_solidity_min = 0.6  # Blades are relatively solid
        self.blade_area_range = (100, 2000)  # Typical blade sizes

        # Frond characteristics (branching, complex)
        self.frond_complexity_min = 0.3  # Fronds have complex boundaries
        self.frond_area_range = (200, 5000)  # Fronds can be large
        self.frond_eccentricity_max = 0.8  # Fronds are less elongated than blades

    def detect_blades_and_fronds(
        self, rgb_image: np.ndarray, kelp_mask: np.ndarray
    ) -> list[MorphologicalFeature]:
        """Detect blades and fronds in Macrocystis kelp.

        Args:
            rgb_image: RGB image array [H, W, 3]
            kelp_mask: Boolean mask indicating kelp areas

        Returns:
            List of detected blade and frond features
        """
        try:
            # Ensure proper image format
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8)

            # Convert to grayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

            # Apply kelp mask
            masked_gray = gray.copy()
            masked_gray[~kelp_mask] = 0

            # Segment kelp regions
            segmented_regions = self._segment_kelp_regions(masked_gray, kelp_mask)

            # Analyze each region for blade/frond characteristics
            features = self._analyze_morphological_regions(segmented_regions, rgb_image)

            logger.info(f"Detected {len(features)} blade/frond features")
            return features

        except Exception as e:
            logger.error(f"Error in blade/frond detection: {e}")
            return []

    def _segment_kelp_regions(
        self, gray_image: np.ndarray, kelp_mask: np.ndarray
    ) -> np.ndarray:
        """Segment kelp areas into distinct regions."""
        try:
            # Apply threshold to create binary image
            threshold_val = threshold_otsu(gray_image[kelp_mask])
            binary = gray_image > threshold_val
            binary = binary & kelp_mask

            # Clean up binary image
            binary = morphology.remove_small_objects(binary, min_size=50)
            binary = morphology.remove_small_holes(binary, area_threshold=25)

            # Use watershed to separate touching objects
            distance = ndimage.distance_transform_edt(binary)
            local_maxima = peak_local_max(distance, min_distance=20, threshold_abs=5)

            markers = np.zeros_like(distance, dtype=int)
            for i, (y, x) in enumerate(zip(local_maxima[0], local_maxima[1], strict=False)):
                markers[y, x] = i + 1

            # Apply watershed
            labels = watershed(-distance, markers, mask=binary)

            return labels

        except Exception as e:
            logger.warning(f"Region segmentation failed: {e}")
            # Fallback to simple connected components
            return measure.label(kelp_mask)

    def _analyze_morphological_regions(
        self, labeled_image: np.ndarray, rgb_image: np.ndarray
    ) -> list[MorphologicalFeature]:
        """Analyze segmented regions for morphological characteristics."""
        features = []

        try:
            props = measure.regionprops(labeled_image)

            for prop in props:
                # Skip very small regions
                if prop.area < 50:
                    continue

                # Calculate morphological properties
                aspect_ratio = (
                    prop.major_axis_length / prop.minor_axis_length
                    if prop.minor_axis_length > 0
                    else 1.0
                )
                solidity = prop.solidity
                eccentricity = prop.eccentricity
                complexity = self._calculate_boundary_complexity(prop)

                # Classify as blade or frond based on characteristics
                feature_type, confidence = self._classify_blade_or_frond(
                    prop.area, aspect_ratio, solidity, eccentricity, complexity
                )

                if feature_type != MorphologyType.UNKNOWN and confidence > 0.3:
                    feature = MorphologicalFeature(
                        feature_type=feature_type,
                        confidence=confidence,
                        area=prop.area,
                        centroid=prop.centroid,
                        bounding_box=prop.bbox,
                        circularity=self._calculate_circularity(prop),
                        aspect_ratio=aspect_ratio,
                        solidity=solidity,
                        eccentricity=eccentricity,
                        properties={
                            "complexity": complexity,
                            "major_axis_length": prop.major_axis_length,
                            "minor_axis_length": prop.minor_axis_length,
                        },
                    )
                    features.append(feature)

        except Exception as e:
            logger.warning(f"Region analysis failed: {e}")

        return features

    def _calculate_boundary_complexity(self, prop) -> float:
        """Calculate boundary complexity as a measure of frond-like characteristics."""
        if prop.perimeter == 0 or prop.area == 0:
            return 0.0

        # Perimeter-to-area ratio normalized by a circle
        circle_perimeter = 2 * np.sqrt(np.pi * prop.area)
        complexity = prop.perimeter / circle_perimeter

        # Normalize to 0-1 range (values > 1 indicate complex boundaries)
        return min(complexity / 3.0, 1.0)  # Normalize assuming max complexity of 3

    def _calculate_circularity(self, prop) -> float:
        """Calculate circularity of a region."""
        if prop.perimeter == 0:
            return 0.0
        return 4 * np.pi * prop.area / (prop.perimeter * prop.perimeter)

    def _classify_blade_or_frond(
        self,
        area: float,
        aspect_ratio: float,
        solidity: float,
        eccentricity: float,
        complexity: float,
    ) -> tuple[MorphologyType, float]:
        """Classify region as blade or frond based on morphological characteristics."""
        blade_score = 0.0
        frond_score = 0.0

        # Blade scoring (elongated, solid, simple)
        if self.blade_area_range[0] <= area <= self.blade_area_range[1]:
            blade_score += 0.3

        if aspect_ratio >= self.blade_aspect_ratio_min:
            blade_score += 0.4

        if solidity >= self.blade_solidity_min:
            blade_score += 0.2

        if complexity < 0.4:  # Simple boundary
            blade_score += 0.1

        # Frond scoring (complex, branching, moderate aspect ratio)
        if self.frond_area_range[0] <= area <= self.frond_area_range[1]:
            frond_score += 0.3

        if complexity >= self.frond_complexity_min:
            frond_score += 0.4

        if eccentricity <= self.frond_eccentricity_max:
            frond_score += 0.2

        if 1.2 <= aspect_ratio <= 3.0:  # Moderate elongation
            frond_score += 0.1

        # Determine classification
        max_score = max(blade_score, frond_score)

        if max_score < 0.3:
            return MorphologyType.UNKNOWN, 0.0
        elif blade_score > frond_score:
            return MorphologyType.BLADE, blade_score
        else:
            return MorphologyType.FROND, frond_score


class MorphologyDetector:
    """Main morphological detection system for kelp species identification."""

    def __init__(self):
        """Initialize morphology detector."""
        self.pneumatocyst_detector = PneumatocystDetector()
        self.blade_frond_detector = BladeFromdDetector()

    def analyze_morphology(
        self,
        rgb_image: np.ndarray,
        kelp_mask: np.ndarray,
        metadata: dict | None = None,
    ) -> MorphologyDetectionResult:
        """Perform comprehensive morphological analysis of kelp image.

        Args:
            rgb_image: RGB image array [H, W, 3]
            kelp_mask: Boolean mask indicating kelp areas
            metadata: Optional metadata (location, species hints, etc.)

        Returns:
            Comprehensive morphological detection results
        """
        processing_notes = []

        try:
            # Detect pneumatocysts
            pneumatocysts = self.pneumatocyst_detector.detect_pneumatocysts(
                rgb_image, kelp_mask
            )
            processing_notes.append(f"Detected {len(pneumatocysts)} pneumatocysts")

            # Detect blades and fronds
            blade_frond_features = self.blade_frond_detector.detect_blades_and_fronds(
                rgb_image, kelp_mask
            )
            processing_notes.append(
                f"Detected {len(blade_frond_features)} blade/frond features"
            )

            # Combine all detected features
            all_features = pneumatocysts + blade_frond_features

            # Calculate summary statistics
            pneumatocyst_count = len(pneumatocysts)
            blade_count = sum(
                1
                for f in blade_frond_features
                if f.feature_type == MorphologyType.BLADE
            )
            frond_count = sum(
                1
                for f in blade_frond_features
                if f.feature_type == MorphologyType.FROND
            )

            total_pneumatocyst_area = sum(f.area for f in pneumatocysts)
            total_blade_area = sum(
                f.area
                for f in blade_frond_features
                if f.feature_type == MorphologyType.BLADE
            )
            total_frond_area = sum(
                f.area
                for f in blade_frond_features
                if f.feature_type == MorphologyType.FROND
            )

            # Calculate overall morphology confidence
            morphology_confidence = self._calculate_overall_confidence(all_features)

            # Calculate species indicators
            species_indicators = self._calculate_species_indicators(
                pneumatocyst_count,
                blade_count,
                frond_count,
                total_pneumatocyst_area,
                total_blade_area,
                total_frond_area,
                kelp_mask.sum(),
            )

            return MorphologyDetectionResult(
                detected_features=all_features,
                pneumatocyst_count=pneumatocyst_count,
                blade_count=blade_count,
                frond_count=frond_count,
                total_pneumatocyst_area=total_pneumatocyst_area,
                total_blade_area=total_blade_area,
                total_frond_area=total_frond_area,
                morphology_confidence=morphology_confidence,
                species_indicators=species_indicators,
                processing_notes=processing_notes,
            )

        except Exception as e:
            logger.error(f"Error in morphological analysis: {e}")
            processing_notes.append(f"Morphological analysis error: {str(e)}")

            return MorphologyDetectionResult(
                detected_features=[],
                pneumatocyst_count=0,
                blade_count=0,
                frond_count=0,
                total_pneumatocyst_area=0.0,
                total_blade_area=0.0,
                total_frond_area=0.0,
                morphology_confidence=0.0,
                species_indicators={},
                processing_notes=processing_notes,
            )

    def _calculate_overall_confidence(
        self, features: list[MorphologicalFeature]
    ) -> float:
        """Calculate overall confidence in morphological detections."""
        if not features:
            return 0.0

        # Weight by area and confidence
        weighted_confidences = []
        total_area = sum(f.area for f in features)

        for feature in features:
            area_weight = (
                feature.area / total_area if total_area > 0 else 1.0 / len(features)
            )
            weighted_confidences.append(feature.confidence * area_weight)

        return sum(weighted_confidences)

    def _calculate_species_indicators(
        self,
        pneumatocyst_count: int,
        blade_count: int,
        frond_count: int,
        pneumatocyst_area: float,
        blade_area: float,
        frond_area: float,
        total_kelp_area: float,
    ) -> dict[str, float]:
        """Calculate species indicator scores based on morphological features."""
        indicators = {}

        # Nereocystis indicators (pneumatocysts)
        if total_kelp_area > 0:
            pneumatocyst_density = pneumatocyst_count / (
                total_kelp_area / 1000
            )  # Per 1000 pixels
            pneumatocyst_coverage = pneumatocyst_area / total_kelp_area

            indicators["nereocystis_morphology_score"] = min(
                1.0, (pneumatocyst_density * 0.3) + (pneumatocyst_coverage * 0.7)
            )
        else:
            indicators["nereocystis_morphology_score"] = 0.0

        # Macrocystis indicators (blades and fronds)
        if total_kelp_area > 0:
            blade_frond_coverage = (blade_area + frond_area) / total_kelp_area
            blade_frond_ratio = (
                blade_area / (blade_area + frond_area)
                if (blade_area + frond_area) > 0
                else 0
            )

            indicators["macrocystis_morphology_score"] = min(
                1.0, (blade_frond_coverage * 0.6) + (blade_frond_ratio * 0.4)
            )
        else:
            indicators["macrocystis_morphology_score"] = 0.0

        # General morphological complexity
        total_features = pneumatocyst_count + blade_count + frond_count
        indicators["morphological_complexity"] = min(
            1.0, total_features / 10.0
        )  # Normalize to 10 features

        return indicators


def create_morphology_detector() -> MorphologyDetector:
    """Factory function to create a morphology detector."""
    return MorphologyDetector()
