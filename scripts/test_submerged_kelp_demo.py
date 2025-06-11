#!/usr/bin/env python3
"""
Submerged Kelp Detection Demonstration Script.

This script demonstrates the submerged kelp detection capabilities including:
- Depth-sensitive red-edge detection
- Water column modeling
- Species-specific depth detection
- Comprehensive depth analysis and reporting

Usage:
    python scripts/test_submerged_kelp_demo.py [--mode MODE] [--species SPECIES]
    
    Modes:
    - basic: Basic submerged kelp detection demonstration
    - advanced: Advanced depth analysis with water column modeling
    - comparative: Compare surface vs submerged detection capabilities
    - species: Species-specific detection comparison
    - comprehensive: All demonstrations
"""

import sys
import os
import argparse
import logging
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import xarray as xr
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.kelpie_carbon_v1.detection.submerged_kelp_detection import (
    SubmergedKelpDetector,
    SubmergedKelpConfig,
    WaterColumnModel,
    detect_submerged_kelp,
    analyze_depth_distribution,
    create_submerged_kelp_detector
)
from src.kelpie_carbon_v1.logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


class SubmergedKelpDemo:
    """Demonstration class for submerged kelp detection capabilities."""
    
    def __init__(self):
        """Initialize the demonstration with sample data."""
        self.demo_sites = self._create_demo_sites()
        self.species_configs = self._create_species_configs()
        
    def _create_demo_sites(self) -> Dict[str, Dict[str, Any]]:
        """Create demonstration sites with different kelp characteristics."""
        sites = {
            "broughton_archipelago": {
                "name": "Broughton Archipelago, BC",
                "coordinates": (50.0833, -126.1667),
                "species": "Nereocystis",
                "description": "Bull kelp with surface canopies and moderate submerged fronds",
                "water_clarity": "clear",  # Secchi depth > 7m
                "typical_depth_range": (0.0, 0.8),  # meters
                "kelp_density": "high"
            },
            "monterey_bay": {
                "name": "Monterey Bay, CA", 
                "coordinates": (36.8000, -121.9000),
                "species": "Macrocystis",
                "description": "Giant kelp with extensive submerged frond systems",
                "water_clarity": "moderate",  # Secchi depth 4-7m
                "typical_depth_range": (0.2, 1.2),  # meters
                "kelp_density": "very_high"
            },
            "saanich_inlet": {
                "name": "Saanich Inlet, BC",
                "coordinates": (48.5830, -123.5000),
                "species": "Mixed",
                "description": "Mixed species with varying depth preferences",
                "water_clarity": "variable",  # Secchi depth 3-8m
                "typical_depth_range": (0.0, 1.0),  # meters
                "kelp_density": "moderate"
            },
            "puget_sound": {
                "name": "Puget Sound, WA",
                "coordinates": (47.6062, -122.3321),
                "species": "Laminaria",
                "description": "Sugar kelp in shallow protected waters",
                "water_clarity": "turbid",  # Secchi depth < 4m
                "typical_depth_range": (0.1, 0.6),  # meters
                "kelp_density": "moderate"
            }
        }
        return sites
    
    def _create_species_configs(self) -> Dict[str, SubmergedKelpConfig]:
        """Create species-specific detection configurations."""
        configs = {}
        
        # Nereocystis luetkeana (Bull kelp) - surface oriented
        configs["Nereocystis"] = SubmergedKelpConfig(
            ndre_surface_threshold=0.06,
            ndre_shallow_threshold=0.03,
            ndre_deep_threshold=0.00,
            species_depth_factors={"Nereocystis": 1.0},
            water_column_model=WaterColumnModel(
                attenuation_coefficient=0.12,  # Clear coastal water
                depth_max_detectable=1.0
            )
        )
        
        # Macrocystis pyrifera (Giant kelp) - deep fronds
        configs["Macrocystis"] = SubmergedKelpConfig(
            ndre_surface_threshold=0.05,
            ndre_shallow_threshold=0.02,
            ndre_deep_threshold=-0.01,
            species_depth_factors={"Macrocystis": 1.3},
            water_column_model=WaterColumnModel(
                attenuation_coefficient=0.15,  # Moderate attenuation
                depth_max_detectable=1.5
            )
        )
        
        # Mixed species
        configs["Mixed"] = SubmergedKelpConfig(
            ndre_surface_threshold=0.055,
            ndre_shallow_threshold=0.025,
            ndre_deep_threshold=-0.005,
            species_depth_factors={"Mixed": 1.1},
            water_column_model=WaterColumnModel(
                attenuation_coefficient=0.15,
                depth_max_detectable=1.2
            )
        )
        
        # Laminaria (Sugar kelp) - shallow preference
        configs["Laminaria"] = SubmergedKelpConfig(
            ndre_surface_threshold=0.07,
            ndre_shallow_threshold=0.04,
            ndre_deep_threshold=0.01,
            species_depth_factors={"Laminaria": 0.8},
            water_column_model=WaterColumnModel(
                attenuation_coefficient=0.20,  # Turbid water
                depth_max_detectable=0.8
            )
        )
        
        return configs
    
    def create_synthetic_kelp_dataset(
        self, 
        site_name: str, 
        size: Tuple[int, int] = (50, 50)
    ) -> xr.Dataset:
        """
        Create synthetic satellite dataset with realistic kelp signatures.
        
        Args:
            site_name: Name of the demo site
            size: Dataset dimensions (height, width)
            
        Returns:
            Synthetic satellite dataset with kelp spectral signatures
        """
        site_info = self.demo_sites[site_name]
        height, width = size
        
        logger.info(f"Creating synthetic dataset for {site_info['name']}")
        
        # Base water reflectance values (typical coastal water)
        base_reflectance = {
            'blue': (0.08, 0.15),     # 443nm
            'green': (0.10, 0.18),    # 560nm  
            'red': (0.06, 0.12),      # 665nm
            'red_edge': (0.08, 0.16), # 705nm
            'red_edge_2': (0.10, 0.18), # 740nm
            'nir': (0.15, 0.25),      # 842nm
        }
        
        # Adjust for water clarity
        clarity_factor = {
            "clear": 0.8,     # Lower background reflectance
            "moderate": 1.0,  # Normal reflectance
            "variable": 1.1,  # Slightly higher
            "turbid": 1.3     # Higher background reflectance
        }[site_info.get("water_clarity", "moderate")]
        
        # Create base dataset
        dataset = xr.Dataset()
        for band, (min_val, max_val) in base_reflectance.items():
            min_adj = min_val * clarity_factor
            max_adj = max_val * clarity_factor
            dataset[band] = (('y', 'x'), np.random.uniform(min_adj, max_adj, size))
        
        # Add realistic kelp spectral signatures based on species and depth
        kelp_areas = self._generate_kelp_areas(site_info, size)
        
        for area_info in kelp_areas:
            self._add_kelp_signature(dataset, area_info, site_info)
        
        # Add some non-kelp vegetation for contrast
        self._add_terrestrial_vegetation(dataset, size)
        
        # Add noise and atmospheric effects
        self._add_atmospheric_effects(dataset)
        
        logger.debug(f"Generated dataset with {len(kelp_areas)} kelp areas")
        return dataset
    
    def _generate_kelp_areas(self, site_info: Dict, size: Tuple[int, int]) -> List[Dict]:
        """Generate kelp areas with realistic size and depth distributions."""
        height, width = size
        areas = []
        
        # Density-based number of kelp patches
        density_patches = {
            "low": 2, "moderate": 4, "high": 6, "very_high": 8
        }
        num_patches = density_patches.get(site_info.get("kelp_density", "moderate"), 4)
        
        depth_min, depth_max = site_info["typical_depth_range"]
        
        for i in range(num_patches):
            # Random patch location and size
            patch_size = np.random.randint(4, 12)  # 4x4 to 12x12 pixels
            start_y = np.random.randint(5, height - patch_size - 5)
            start_x = np.random.randint(5, width - patch_size - 5)
            
            # Random depth within species range
            patch_depth = np.random.uniform(depth_min, depth_max)
            
            # Kelp density within patch (0.3 to 1.0)
            kelp_density = np.random.uniform(0.3, 1.0)
            
            area_info = {
                "bounds": (start_y, start_y + patch_size, start_x, start_x + patch_size),
                "depth": patch_depth,
                "density": kelp_density,
                "species": site_info["species"]
            }
            areas.append(area_info)
        
        return areas
    
    def _add_kelp_signature(
        self, 
        dataset: xr.Dataset, 
        area_info: Dict, 
        site_info: Dict
    ) -> None:
        """Add realistic kelp spectral signature to dataset area."""
        y1, y2, x1, x2 = area_info["bounds"]
        depth = area_info["depth"]
        density = area_info["density"]
        species = area_info["species"]
        
        # Species-specific spectral characteristics
        species_params = {
            "Nereocystis": {
                "red_edge_enhancement": 1.8,   # Strong red-edge response
                "red_absorption": 0.7,          # Moderate red absorption
                "nir_reflectance": 1.2          # Moderate NIR
            },
            "Macrocystis": {
                "red_edge_enhancement": 2.2,   # Very strong red-edge
                "red_absorption": 0.6,          # Strong red absorption
                "nir_reflectance": 1.4          # Higher NIR
            },
            "Laminaria": {
                "red_edge_enhancement": 1.5,   # Moderate red-edge
                "red_absorption": 0.8,          # Less red absorption
                "nir_reflectance": 1.1          # Lower NIR
            },
            "Mixed": {
                "red_edge_enhancement": 1.8,   # Average response
                "red_absorption": 0.7,          # Average absorption
                "nir_reflectance": 1.25         # Average NIR
            }
        }
        
        params = species_params.get(species, species_params["Mixed"])
        
        # Depth attenuation factor (exponential decay)
        attenuation = np.exp(-depth * 0.8)  # Approximate water attenuation
        
        # Kelp spectral modification
        area_slice = np.s_[y1:y2, x1:x2]
        area_shape = (y2 - y1, x2 - x1)
        
        # Add spatial variability within the kelp patch
        spatial_var = np.random.uniform(0.8, 1.2, area_shape)
        density_map = np.ones(area_shape) * density * spatial_var
        
        # Apply kelp spectral characteristics with depth and density effects
        enhancement_factor = attenuation * density_map
        
        # Red-edge enhancement (primary kelp signature)
        current_red_edge = dataset['red_edge_2'][area_slice].values
        kelp_red_edge = current_red_edge * (1 + params["red_edge_enhancement"] * enhancement_factor)
        dataset['red_edge_2'][area_slice] = kelp_red_edge
        
        current_red_edge_1 = dataset['red_edge'][area_slice].values
        kelp_red_edge_1 = current_red_edge_1 * (1 + params["red_edge_enhancement"] * 0.8 * enhancement_factor)
        dataset['red_edge'][area_slice] = kelp_red_edge_1
        
        # Red absorption (kelp absorbs red light)
        current_red = dataset['red'][area_slice].values
        kelp_red = current_red * (params["red_absorption"] * (1 - 0.5 * enhancement_factor))
        dataset['red'][area_slice] = kelp_red
        
        # NIR reflectance (moderate enhancement)
        current_nir = dataset['nir'][area_slice].values
        kelp_nir = current_nir * (1 + (params["nir_reflectance"] - 1) * enhancement_factor)
        dataset['nir'][area_slice] = kelp_nir
        
        # Green light (slight modification)
        current_green = dataset['green'][area_slice].values
        kelp_green = current_green * (1 + 0.2 * enhancement_factor)
        dataset['green'][area_slice] = kelp_green
    
    def _add_terrestrial_vegetation(self, dataset: xr.Dataset, size: Tuple[int, int]) -> None:
        """Add terrestrial vegetation signatures for contrast."""
        height, width = size
        
        # Add 1-2 small terrestrial vegetation patches
        for _ in range(np.random.randint(1, 3)):
            patch_size = np.random.randint(3, 8)
            start_y = np.random.randint(0, height - patch_size)
            start_x = np.random.randint(0, width - patch_size)
            
            area_slice = np.s_[start_y:start_y + patch_size, start_x:start_x + patch_size]
            
            # Terrestrial vegetation: high NIR, low red, moderate green
            dataset['nir'][area_slice] *= 2.5      # Strong NIR response
            dataset['red'][area_slice] *= 0.4      # Low red reflectance
            dataset['green'][area_slice] *= 1.3    # Moderate green
            dataset['red_edge'][area_slice] *= 1.1 # Slight red-edge
    
    def _add_atmospheric_effects(self, dataset: xr.Dataset) -> None:
        """Add realistic atmospheric effects and noise."""
        # Add small amount of noise
        noise_level = 0.02
        for band in dataset.data_vars:
            noise = np.random.normal(0, noise_level, dataset[band].shape)
            dataset[band] = dataset[band] + noise
            
            # Ensure values stay in valid range [0, 1]
            dataset[band] = np.clip(dataset[band], 0.0, 1.0)
        
        # Add slight atmospheric haze (increases blue reflectance)
        haze_factor = np.random.uniform(1.0, 1.1)
        dataset['blue'] *= haze_factor
        dataset['blue'] = np.clip(dataset['blue'], 0.0, 1.0)
    
    def run_basic_demo(self) -> None:
        """Run basic submerged kelp detection demonstration."""
        print("\n" + "="*70)
        print("🌊 BASIC SUBMERGED KELP DETECTION DEMONSTRATION")
        print("="*70)
        
        # Use Broughton Archipelago as example
        site_name = "broughton_archipelago"
        site_info = self.demo_sites[site_name]
        
        print(f"\n📍 Test Site: {site_info['name']}")
        print(f"   Species: {site_info['species']} (Bull kelp)")
        print(f"   Water Clarity: {site_info['water_clarity'].title()}")
        print(f"   Typical Depth Range: {site_info['typical_depth_range'][0]:.1f}-{site_info['typical_depth_range'][1]:.1f}m")
        
        # Create synthetic dataset
        print("\n🛰️ Creating synthetic satellite imagery...")
        dataset = self.create_synthetic_kelp_dataset(site_name, size=(40, 40))
        
        # Configure detector for this species
        config = self.species_configs[site_info['species']]
        detector = SubmergedKelpDetector(config)
        
        # Run detection
        print("🔍 Running submerged kelp detection...")
        start_time = time.time()
        
        result = detector.detect_submerged_kelp(
            dataset,
            species=site_info['species'],
            include_depth_analysis=True
        )
        
        detection_time = time.time() - start_time
        
        # Analyze results
        print(f"⏱️ Detection completed in {detection_time:.2f} seconds")
        
        metadata = result.detection_metadata
        print(f"\n📊 Detection Results:")
        print(f"   Total pixels analyzed: {metadata['total_pixels']:,}")
        print(f"   Surface kelp pixels: {metadata['surface_kelp_pixels']:,}")
        print(f"   Submerged kelp pixels: {metadata['submerged_kelp_pixels']:,}")
        print(f"   Total kelp coverage: {metadata['total_kelp_coverage_percent']:.2f}%")
        print(f"   Surface area: {metadata['surface_area_m2']:,.0f} m²")
        print(f"   Submerged area: {metadata['submerged_area_m2']:,.0f} m²")
        print(f"   Surface/Submerged ratio: {metadata['surface_to_submerged_ratio']:.2f}")
        
        # Depth analysis
        if np.any(result.combined_kelp_mask):
            depth_analysis = analyze_depth_distribution(result)
            print(f"\n🌊 Depth Analysis:")
            print(f"   Mean depth: {depth_analysis['mean_depth_m']:.2f}m")
            print(f"   Depth range: {depth_analysis['min_depth_m']:.2f}m - {depth_analysis['max_depth_m']:.2f}m")
            print(f"   Surface kelp (0-30cm): {depth_analysis['surface_fraction']:.1%}")
            print(f"   Shallow submerged (30-70cm): {depth_analysis['shallow_fraction']:.1%}")
            print(f"   Deep submerged (>70cm): {depth_analysis['deep_fraction']:.1%}")
            print(f"   Mean confidence: {depth_analysis['mean_confidence']:.2f}")
        else:
            print("\n⚠️ No kelp detected in this synthetic scene")
        
        print("\n✅ Basic demonstration completed successfully!")
    
    def run_advanced_demo(self) -> None:
        """Run advanced depth analysis demonstration."""
        print("\n" + "="*70)
        print("🔬 ADVANCED DEPTH ANALYSIS DEMONSTRATION")
        print("="*70)
        
        # Use Monterey Bay for giant kelp with complex depth structure
        site_name = "monterey_bay"
        site_info = self.demo_sites[site_name]
        
        print(f"\n📍 Test Site: {site_info['name']}")
        print(f"   Species: {site_info['species']} (Giant kelp)")
        print(f"   Description: {site_info['description']}")
        
        # Create larger dataset for better depth analysis
        print("\n🛰️ Creating high-resolution synthetic imagery...")
        dataset = self.create_synthetic_kelp_dataset(site_name, size=(60, 60))
        
        # Use advanced configuration
        config = self.species_configs[site_info['species']]
        print(f"\n⚙️ Advanced Configuration:")
        print(f"   Surface threshold: {config.ndre_surface_threshold:.3f}")
        print(f"   Shallow threshold: {config.ndre_shallow_threshold:.3f}")
        print(f"   Deep threshold: {config.ndre_deep_threshold:.3f}")
        print(f"   Max detectable depth: {config.water_column_model.depth_max_detectable:.1f}m")
        print(f"   Attenuation coefficient: {config.water_column_model.attenuation_coefficient:.3f} m⁻¹")
        
        detector = SubmergedKelpDetector(config)
        
        # Run comprehensive detection
        print("\n🔍 Running advanced submerged kelp detection...")
        start_time = time.time()
        
        result = detector.detect_submerged_kelp(
            dataset,
            species=site_info['species'],
            include_depth_analysis=True
        )
        
        detection_time = time.time() - start_time
        print(f"⏱️ Advanced analysis completed in {detection_time:.2f} seconds")
        
        # Detailed water column analysis
        if result.water_column_properties:
            print(f"\n🌊 Water Column Properties:")
            for prop_name, prop_array in result.water_column_properties.items():
                mean_val = np.mean(prop_array)
                std_val = np.std(prop_array)
                print(f"   {prop_name.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Comprehensive depth analysis
        if np.any(result.combined_kelp_mask):
            depth_analysis = analyze_depth_distribution(result)
            
            print(f"\n📊 Comprehensive Depth Distribution:")
            print(f"   Total kelp pixels: {depth_analysis['total_kelp_pixels']:,}")
            print(f"   Mean depth: {depth_analysis['mean_depth_m']:.3f}m")
            print(f"   Median depth: {depth_analysis['median_depth_m']:.3f}m")
            print(f"   Depth standard deviation: {depth_analysis['depth_std_m']:.3f}m")
            print(f"   Depth range: {depth_analysis['min_depth_m']:.3f}m to {depth_analysis['max_depth_m']:.3f}m")
            
            print(f"\n🏊 Depth Zone Distribution:")
            print(f"   Surface zone (0-30cm): {depth_analysis['surface_fraction']:.1%}")
            print(f"   Shallow zone (30-70cm): {depth_analysis['shallow_fraction']:.1%}")
            print(f"   Deep zone (>70cm): {depth_analysis['deep_fraction']:.1%}")
            
            print(f"\n🎯 Detection Quality:")
            print(f"   Mean confidence: {depth_analysis['mean_confidence']:.3f}")
            
            # Depth histogram analysis
            depths = result.depth_estimate[result.combined_kelp_mask]
            print(f"\n📈 Depth Distribution Analysis:")
            print(f"   0-25cm: {np.sum(depths <= 0.25)}/{len(depths)} pixels ({np.sum(depths <= 0.25)/len(depths):.1%})")
            print(f"   25-50cm: {np.sum((depths > 0.25) & (depths <= 0.5))}/{len(depths)} pixels ({np.sum((depths > 0.25) & (depths <= 0.5))/len(depths):.1%})")
            print(f"   50-75cm: {np.sum((depths > 0.5) & (depths <= 0.75))}/{len(depths)} pixels ({np.sum((depths > 0.5) & (depths <= 0.75))/len(depths):.1%})")
            print(f"   75-100cm: {np.sum((depths > 0.75) & (depths <= 1.0))}/{len(depths)} pixels ({np.sum((depths > 0.75) & (depths <= 1.0))/len(depths):.1%})")
            print(f"   >100cm: {np.sum(depths > 1.0)}/{len(depths)} pixels ({np.sum(depths > 1.0)/len(depths):.1%})")
        
        print("\n✅ Advanced demonstration completed successfully!")
    
    def run_comparative_demo(self) -> None:
        """Run comparative analysis of surface vs submerged detection."""
        print("\n" + "="*70)
        print("⚖️ SURFACE vs SUBMERGED DETECTION COMPARISON")
        print("="*70)
        
        print("\n🧪 Testing detection capabilities across multiple sites...")
        
        comparison_results = []
        
        for site_name, site_info in self.demo_sites.items():
            print(f"\n📍 Testing: {site_info['name']}")
            
            # Create dataset for this site
            dataset = self.create_synthetic_kelp_dataset(site_name, size=(45, 45))
            
            # Configure detector
            config = self.species_configs[site_info['species']]
            detector = SubmergedKelpDetector(config)
            
            # Run detection
            result = detector.detect_submerged_kelp(
                dataset,
                species=site_info['species'],
                include_depth_analysis=True
            )
            
            # Collect results
            metadata = result.detection_metadata
            site_result = {
                "site": site_info['name'],
                "species": site_info['species'],
                "water_clarity": site_info['water_clarity'],
                "surface_pixels": metadata['surface_kelp_pixels'],
                "submerged_pixels": metadata['submerged_kelp_pixels'],
                "total_pixels": metadata['total_kelp_pixels'],
                "surface_percentage": metadata['surface_coverage_percent'],
                "submerged_percentage": metadata['submerged_coverage_percent'],
                "surface_to_submerged_ratio": metadata['surface_to_submerged_ratio']
            }
            
            if np.any(result.combined_kelp_mask):
                depth_analysis = analyze_depth_distribution(result)
                site_result.update({
                    "mean_depth": depth_analysis['mean_depth_m'],
                    "surface_fraction": depth_analysis['surface_fraction'],
                    "deep_fraction": depth_analysis['deep_fraction'],
                    "mean_confidence": depth_analysis['mean_confidence']
                })
            else:
                site_result.update({
                    "mean_depth": 0.0,
                    "surface_fraction": 0.0,
                    "deep_fraction": 0.0,
                    "mean_confidence": 0.0
                })
            
            comparison_results.append(site_result)
            
            print(f"   Surface: {site_result['surface_pixels']:,} pixels ({site_result['surface_percentage']:.1f}%)")
            print(f"   Submerged: {site_result['submerged_pixels']:,} pixels ({site_result['submerged_percentage']:.1f}%)")
            print(f"   Ratio: {site_result['surface_to_submerged_ratio']:.2f}")
        
        # Summary comparison
        print(f"\n📊 COMPARATIVE ANALYSIS SUMMARY")
        print("-" * 70)
        print(f"{'Site':<25} {'Species':<12} {'Surface':<8} {'Submerged':<10} {'Ratio':<8} {'Avg Depth':<10}")
        print("-" * 70)
        
        for result in comparison_results:
            site_short = result['site'].split(',')[0][:24]
            print(f"{site_short:<25} {result['species']:<12} {result['surface_pixels']:<8} "
                  f"{result['submerged_pixels']:<10} {result['surface_to_submerged_ratio']:<8.1f} "
                  f"{result['mean_depth']:<10.2f}")
        
        # Analysis insights
        print(f"\n🔍 Key Insights:")
        
        # Species comparison
        species_summary = {}
        for result in comparison_results:
            species = result['species']
            if species not in species_summary:
                species_summary[species] = []
            species_summary[species].append(result)
        
        for species, results in species_summary.items():
            if len(results) > 0:
                avg_ratio = np.mean([r['surface_to_submerged_ratio'] for r in results])
                avg_depth = np.mean([r['mean_depth'] for r in results])
                print(f"   {species}: Avg surface/submerged ratio = {avg_ratio:.2f}, Avg depth = {avg_depth:.2f}m")
        
        # Water clarity impact
        clarity_groups = {}
        for result in comparison_results:
            clarity = result['water_clarity']
            if clarity not in clarity_groups:
                clarity_groups[clarity] = []
            clarity_groups[clarity].append(result)
        
        print(f"\n🌊 Water Clarity Impact:")
        for clarity, results in clarity_groups.items():
            if len(results) > 0:
                avg_confidence = np.mean([r['mean_confidence'] for r in results if r['mean_confidence'] > 0])
                total_detection = np.mean([r['surface_percentage'] + r['submerged_percentage'] for r in results])
                print(f"   {clarity.title()}: Avg confidence = {avg_confidence:.2f}, Total detection = {total_detection:.1f}%")
        
        print("\n✅ Comparative analysis completed successfully!")
    
    def run_species_demo(self) -> None:
        """Run species-specific detection comparison."""
        print("\n" + "="*70)
        print("🐙 SPECIES-SPECIFIC DETECTION COMPARISON")
        print("="*70)
        
        print("\n🧬 Testing species-specific detection parameters...")
        
        # Test all species on the same synthetic location
        base_dataset = self.create_synthetic_kelp_dataset("saanich_inlet", size=(50, 50))
        
        species_results = {}
        
        for species, config in self.species_configs.items():
            print(f"\n🔬 Testing {species} detection parameters...")
            
            detector = SubmergedKelpDetector(config)
            
            start_time = time.time()
            result = detector.detect_submerged_kelp(
                base_dataset,
                species=species,
                include_depth_analysis=True
            )
            detection_time = time.time() - start_time
            
            metadata = result.detection_metadata
            
            # Store results
            species_results[species] = {
                "detection_time": detection_time,
                "surface_pixels": metadata['surface_kelp_pixels'],
                "submerged_pixels": metadata['submerged_kelp_pixels'],
                "total_pixels": metadata['total_kelp_pixels'],
                "coverage_percent": metadata['total_kelp_coverage_percent'],
                "surface_to_submerged": metadata['surface_to_submerged_ratio'],
                "config": {
                    "surface_threshold": config.ndre_surface_threshold,
                    "shallow_threshold": config.ndre_shallow_threshold,
                    "deep_threshold": config.ndre_deep_threshold,
                    "depth_factor": config.species_depth_factors.get(species, 1.0),
                    "max_depth": config.water_column_model.depth_max_detectable
                }
            }
            
            if np.any(result.combined_kelp_mask):
                depth_analysis = analyze_depth_distribution(result)
                species_results[species]["depth_analysis"] = depth_analysis
            
            print(f"   Detection time: {detection_time:.2f}s")
            print(f"   Total kelp: {metadata['total_kelp_pixels']:,} pixels ({metadata['total_kelp_coverage_percent']:.1f}%)")
            print(f"   Surface/Submerged: {metadata['surface_kelp_pixels']:,}/{metadata['submerged_kelp_pixels']:,}")
        
        # Comparative analysis
        print(f"\n📊 SPECIES DETECTION COMPARISON")
        print("-" * 80)
        print(f"{'Species':<12} {'Total':<8} {'Surface':<8} {'Submerged':<10} {'Coverage':<10} {'Avg Depth':<10}")
        print("-" * 80)
        
        for species, result in species_results.items():
            avg_depth = result.get('depth_analysis', {}).get('mean_depth_m', 0.0)
            print(f"{species:<12} {result['total_pixels']:<8} {result['surface_pixels']:<8} "
                  f"{result['submerged_pixels']:<10} {result['coverage_percent']:<10.1f}% "
                  f"{avg_depth:<10.2f}m")
        
        # Configuration comparison
        print(f"\n⚙️ SPECIES-SPECIFIC CONFIGURATIONS")
        print("-" * 90)
        print(f"{'Species':<12} {'Surface T':<10} {'Shallow T':<10} {'Deep T':<10} {'Depth F':<10} {'Max Depth':<10}")
        print("-" * 90)
        
        for species, result in species_results.items():
            config = result['config']
            print(f"{species:<12} {config['surface_threshold']:<10.3f} {config['shallow_threshold']:<10.3f} "
                  f"{config['deep_threshold']:<10.3f} {config['depth_factor']:<10.1f} "
                  f"{config['max_depth']:<10.1f}m")
        
        # Performance insights
        print(f"\n🎯 Species Detection Insights:")
        
        # Find best performing species
        best_total = max(species_results.items(), key=lambda x: x[1]['total_pixels'])
        best_submerged = max(species_results.items(), key=lambda x: x[1]['submerged_pixels'])
        deepest = max(species_results.items(), 
                     key=lambda x: x[1].get('depth_analysis', {}).get('mean_depth_m', 0))
        
        print(f"   Highest total detection: {best_total[0]} ({best_total[1]['total_pixels']:,} pixels)")
        print(f"   Best submerged detection: {best_submerged[0]} ({best_submerged[1]['submerged_pixels']:,} pixels)")
        print(f"   Deepest average detection: {deepest[0]} ({deepest[1].get('depth_analysis', {}).get('mean_depth_m', 0):.2f}m)")
        
        # Detection efficiency
        fastest = min(species_results.items(), key=lambda x: x[1]['detection_time'])
        print(f"   Fastest detection: {fastest[0]} ({fastest[1]['detection_time']:.2f}s)")
        
        print("\n✅ Species comparison completed successfully!")
    
    def run_comprehensive_demo(self) -> None:
        """Run all demonstrations in sequence."""
        print("\n" + "="*70)
        print("🌊 COMPREHENSIVE SUBMERGED KELP DETECTION DEMONSTRATION")
        print("="*70)
        print("\nRunning complete demonstration suite...")
        
        start_time = time.time()
        
        # Run all demonstrations
        self.run_basic_demo()
        self.run_advanced_demo()
        self.run_comparative_demo()
        self.run_species_demo()
        
        total_time = time.time() - start_time
        
        print(f"\n" + "="*70)
        print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETED")
        print("="*70)
        print(f"Total demonstration time: {total_time:.1f} seconds")
        print(f"\n📋 Summary of capabilities demonstrated:")
        print("   ✅ Basic submerged kelp detection")
        print("   ✅ Advanced depth analysis with water column modeling")
        print("   ✅ Surface vs submerged detection comparison")
        print("   ✅ Species-specific detection parameters")
        print("   ✅ Multi-site performance analysis")
        print("   ✅ Water clarity impact assessment")
        print("   ✅ Depth distribution analysis")
        print("   ✅ Detection quality metrics")
        
        print(f"\n🚀 Submerged kelp detection system ready for production use!")


def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(
        description="Submerged Kelp Detection Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_submerged_kelp_demo.py
    python scripts/test_submerged_kelp_demo.py --mode advanced
    python scripts/test_submerged_kelp_demo.py --mode species --species Macrocystis
    python scripts/test_submerged_kelp_demo.py --mode comprehensive
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "comparative", "species", "comprehensive"],
        default="basic",
        help="Demonstration mode to run (default: basic)"
    )
    
    parser.add_argument(
        "--species",
        choices=["Nereocystis", "Macrocystis", "Laminaria", "Mixed"],
        default="Mixed",
        help="Species to focus on for species-specific demos (default: Mixed)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Initialize demonstration
    demo = SubmergedKelpDemo()
    
    # Run selected demonstration
    try:
        if args.mode == "basic":
            demo.run_basic_demo()
        elif args.mode == "advanced":
            demo.run_advanced_demo()
        elif args.mode == "comparative":
            demo.run_comparative_demo()
        elif args.mode == "species":
            demo.run_species_demo()
        elif args.mode == "comprehensive":
            demo.run_comprehensive_demo()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 