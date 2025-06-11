"""
Enhanced Validation Metrics - Task ML1
Implements RMSE, MAE, R² accuracy metrics for biomass and carbon quantification.
User-requested metrics for 4 validation coordinates (BC, California, Tasmania, Broughton).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationCoordinate:
    """Validation site with coordinate and species information."""
    name: str
    latitude: float
    longitude: float
    species: str
    region: str


@dataclass
class BiomassValidationData:
    """Biomass validation data for a specific site."""
    coordinate: ValidationCoordinate
    observed_biomass_kg_m2: np.ndarray
    predicted_biomass_kg_m2: np.ndarray
    carbon_content_ratio: float
    measurement_dates: List[str]
    measurement_metadata: Dict[str, Any]


@dataclass
class ValidationMetricsResult:
    """Comprehensive validation metrics result."""
    coordinate: ValidationCoordinate
    biomass_metrics: Dict[str, float]
    carbon_metrics: Dict[str, float]
    uncertainty_analysis: Dict[str, Any]
    species_specific_metrics: Dict[str, float]
    temporal_metrics: Dict[str, Any]


class EnhancedValidationMetrics:
    """
    Enhanced validation metrics implementing RMSE, MAE, R² for biomass and carbon validation.
    Addresses critical validation gaps identified in model validation analysis.
    """
    
    # Four primary validation coordinates identified in task analysis
    VALIDATION_COORDINATES = [
        ValidationCoordinate(
            name="British Columbia",
            latitude=50.1163,
            longitude=-125.2735,
            species="Nereocystis luetkeana",
            region="Pacific Northwest"
        ),
        ValidationCoordinate(
            name="California",
            latitude=36.6002,
            longitude=-121.9015,
            species="Macrocystis pyrifera",
            region="California Current"
        ),
        ValidationCoordinate(
            name="Tasmania",
            latitude=-43.1,
            longitude=147.3,
            species="Macrocystis pyrifera",
            region="Southern Ocean"
        ),
        ValidationCoordinate(
            name="Broughton Archipelago",
            latitude=50.0833,
            longitude=-126.1667,
            species="Nereocystis luetkeana",
            region="British Columbia"
        )
    ]
    
    # Species-specific carbon content ratios (kg C / kg dry weight)
    SPECIES_CARBON_RATIOS = {
        "Nereocystis luetkeana": 0.30,  # Bull kelp
        "Macrocystis pyrifera": 0.28,   # Giant kelp
        "Mixed": 0.29                   # Mixed species average
    }
    
    def __init__(self):
        """Initialize enhanced validation metrics calculator."""
        logger.info("Initializing Enhanced Validation Metrics (RMSE, MAE, R²)")
        
    def calculate_biomass_accuracy_metrics(
        self, 
        predicted: np.ndarray, 
        observed: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive biomass prediction accuracy with RMSE, MAE, R².
        
        Args:
            predicted: Predicted biomass values (kg/m²)
            observed: Observed biomass values (kg/m²)
            
        Returns:
            Dictionary of biomass accuracy metrics
        """
        logger.debug(f"Calculating biomass accuracy metrics for {len(predicted)} data points")
        
        # Handle edge cases
        if len(predicted) == 0 or len(observed) == 0:
            logger.warning("Empty prediction or observation arrays")
            return self._empty_metrics_dict()
            
        if len(predicted) != len(observed):
            logger.warning(f"Mismatched array lengths: pred={len(predicted)}, obs={len(observed)}")
            min_len = min(len(predicted), len(observed))
            predicted = predicted[:min_len]
            observed = observed[:min_len]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(predicted) | np.isnan(observed))
        predicted_clean = predicted[valid_mask]
        observed_clean = observed[valid_mask]
        
        if len(predicted_clean) == 0:
            logger.warning("No valid data points after NaN removal")
            return self._empty_metrics_dict()
        
        if len(predicted_clean) == 1:
            logger.warning("Only one valid data point - limited metrics available")
            # Handle single point case with basic metrics only
            residual = predicted_clean[0] - observed_clean[0]
            return {
                'rmse_biomass_kg_m2': float(abs(residual)),
                'mae_biomass_kg_m2': float(abs(residual)),
                'r2_biomass_correlation': 0.0,  # Undefined for single point
                'mape_percentage': float(abs(residual / observed_clean[0] * 100)) if observed_clean[0] != 0 else 0.0,
                'bias_percentage': float(residual / observed_clean[0] * 100) if observed_clean[0] != 0 else 0.0,
                'uncertainty_bounds_95': {'lower_bound': 0.0, 'upper_bound': 0.0, 'confidence_level': 0.95, 'residual_std': 0.0},
                'pearson_correlation': 0.0,
                'pearson_p_value': 1.0,
                'spearman_correlation': 0.0,
                'spearman_p_value': 1.0,
                'n_valid_points': 1,
                'mean_observed': float(observed_clean[0]),
                'mean_predicted': float(predicted_clean[0]),
                'std_observed': 0.0,
                'std_predicted': 0.0
            }
        
        try:
            # PRIMARY METRICS (USER REQUESTED)
            rmse = np.sqrt(mean_squared_error(observed_clean, predicted_clean))
            mae = mean_absolute_error(observed_clean, predicted_clean)
            r2 = r2_score(observed_clean, predicted_clean)
            
            # ADDITIONAL BIOMASS METRICS
            mape = np.mean(np.abs((observed_clean - predicted_clean) / np.where(observed_clean != 0, observed_clean, 1))) * 100
            bias_percentage = np.mean((predicted_clean - observed_clean) / np.where(observed_clean != 0, observed_clean, 1)) * 100
            
            # Statistical metrics
            pearson_corr, pearson_p = stats.pearsonr(predicted_clean, observed_clean)
            spearman_corr, spearman_p = stats.spearmanr(predicted_clean, observed_clean)
            
            # Uncertainty bounds (95% confidence interval)
            uncertainty_bounds_95 = self._calculate_prediction_intervals(predicted_clean, observed_clean, 0.95)
            
            return {
                # PRIMARY METRICS (USER REQUESTED)
                'rmse_biomass_kg_m2': float(rmse),
                'mae_biomass_kg_m2': float(mae),
                'r2_biomass_correlation': float(r2),
                
                # ADDITIONAL BIOMASS METRICS
                'mape_percentage': float(mape),
                'bias_percentage': float(bias_percentage),
                'uncertainty_bounds_95': uncertainty_bounds_95,
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'n_valid_points': int(len(predicted_clean)),
                'mean_observed': float(np.mean(observed_clean)),
                'mean_predicted': float(np.mean(predicted_clean)),
                'std_observed': float(np.std(observed_clean)),
                'std_predicted': float(np.std(predicted_clean))
            }
            
        except Exception as e:
            logger.error(f"Error calculating biomass metrics: {e}")
            return self._empty_metrics_dict()
    
    def calculate_carbon_accuracy_metrics(
        self, 
        biomass_pred: np.ndarray, 
        biomass_obs: np.ndarray, 
        carbon_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate carbon sequestration accuracy metrics (RMSE, MAE, R²).
        
        Args:
            biomass_pred: Predicted biomass values (kg/m²)
            biomass_obs: Observed biomass values (kg/m²)
            carbon_factors: Dictionary with carbon conversion factors
            
        Returns:
            Dictionary of carbon accuracy metrics
        """
        logger.debug(f"Calculating carbon accuracy metrics for {len(biomass_pred)} data points")
        
        carbon_content_ratio = carbon_factors.get('carbon_content_ratio', 0.29)
        
        # Convert biomass to carbon
        carbon_pred = biomass_pred * carbon_content_ratio
        carbon_obs = biomass_obs * carbon_content_ratio
        
        # Handle edge cases
        if len(carbon_pred) == 0 or len(carbon_obs) == 0:
            logger.warning("Empty carbon prediction or observation arrays")
            return self._empty_carbon_metrics_dict()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(carbon_pred) | np.isnan(carbon_obs))
        carbon_pred_clean = carbon_pred[valid_mask]
        carbon_obs_clean = carbon_obs[valid_mask]
        
        if len(carbon_pred_clean) == 0:
            logger.warning("No valid carbon data points after NaN removal")
            return self._empty_carbon_metrics_dict()
        
        try:
            # PRIMARY CARBON METRICS (USER REQUESTED)
            rmse_carbon = np.sqrt(mean_squared_error(carbon_obs_clean, carbon_pred_clean)) * 10  # Convert to tC/hectare
            mae_carbon = mean_absolute_error(carbon_obs_clean, carbon_pred_clean) * 10
            r2_carbon = r2_score(carbon_obs_clean, carbon_pred_clean)
            
            # SEQUESTRATION METRICS
            sequestration_rate_rmse = self._calculate_annual_sequestration_rmse(carbon_pred_clean, carbon_obs_clean)
            carbon_bias_percentage = np.mean((carbon_pred_clean - carbon_obs_clean) / np.where(carbon_obs_clean != 0, carbon_obs_clean, 1)) * 100
            
            # Statistical metrics for carbon
            carbon_pearson_corr, carbon_pearson_p = stats.pearsonr(carbon_pred_clean, carbon_obs_clean)
            
            return {
                # PRIMARY CARBON METRICS (USER REQUESTED)
                'rmse_carbon_tc_hectare': float(rmse_carbon),
                'mae_carbon_tc_hectare': float(mae_carbon),
                'r2_carbon_correlation': float(r2_carbon),
                
                # SEQUESTRATION METRICS
                'sequestration_rate_rmse_tc_year': float(sequestration_rate_rmse),
                'carbon_bias_percentage': float(carbon_bias_percentage),
                'carbon_pearson_correlation': float(carbon_pearson_corr),
                'carbon_pearson_p_value': float(carbon_pearson_p),
                'n_valid_carbon_points': int(len(carbon_pred_clean)),
                'mean_carbon_observed_kg_m2': float(np.mean(carbon_obs_clean)),
                'mean_carbon_predicted_kg_m2': float(np.mean(carbon_pred_clean)),
                'carbon_content_ratio_used': float(carbon_content_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating carbon metrics: {e}")
            return self._empty_carbon_metrics_dict()
    
    def validate_model_predictions_against_real_data(
        self, 
        validation_data: List[BiomassValidationData]
    ) -> Dict[str, ValidationMetricsResult]:
        """
        Comprehensive validation against all four validation sample points.
        
        Args:
            validation_data: List of biomass validation data for each coordinate
            
        Returns:
            Dictionary mapping coordinate names to validation results
        """
        logger.info("Starting comprehensive model validation against 4 coordinate sites")
        
        results = {}
        
        for data in validation_data:
            logger.debug(f"Validating coordinate: {data.coordinate.name}")
            
            try:
                # Calculate biomass metrics
                biomass_metrics = self.calculate_biomass_accuracy_metrics(
                    data.predicted_biomass_kg_m2,
                    data.observed_biomass_kg_m2
                )
                
                # Calculate carbon metrics
                carbon_factors = {
                    'carbon_content_ratio': self.SPECIES_CARBON_RATIOS.get(
                        data.coordinate.species, 
                        self.SPECIES_CARBON_RATIOS['Mixed']
                    )
                }
                carbon_metrics = self.calculate_carbon_accuracy_metrics(
                    data.predicted_biomass_kg_m2,
                    data.observed_biomass_kg_m2,
                    carbon_factors
                )
                
                # Calculate uncertainty analysis
                uncertainty_analysis = self._calculate_uncertainty_analysis(
                    data.predicted_biomass_kg_m2,
                    data.observed_biomass_kg_m2
                )
                
                # Calculate species-specific metrics
                species_metrics = self._calculate_species_specific_metrics(
                    data.coordinate.species,
                    biomass_metrics,
                    carbon_metrics
                )
                
                # Calculate temporal metrics if date information available
                temporal_metrics = self._calculate_temporal_metrics(
                    data.measurement_dates,
                    data.predicted_biomass_kg_m2,
                    data.observed_biomass_kg_m2
                )
                
                # Create result
                result = ValidationMetricsResult(
                    coordinate=data.coordinate,
                    biomass_metrics=biomass_metrics,
                    carbon_metrics=carbon_metrics,
                    uncertainty_analysis=uncertainty_analysis,
                    species_specific_metrics=species_metrics,
                    temporal_metrics=temporal_metrics
                )
                
                results[data.coordinate.name] = result
                logger.info(f"Validation complete for {data.coordinate.name}: "
                          f"R²={biomass_metrics.get('r2_biomass_correlation', 0):.3f}, "
                          f"RMSE={biomass_metrics.get('rmse_biomass_kg_m2', 0):.3f} kg/m²")
                
            except Exception as e:
                logger.error(f"Error validating coordinate {data.coordinate.name}: {e}")
                continue
        
        logger.info(f"Comprehensive validation complete for {len(results)} coordinates")
        return results
    
    def generate_validation_summary(
        self, 
        validation_results: Dict[str, ValidationMetricsResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation summary across all coordinates.
        
        Args:
            validation_results: Results from validate_model_predictions_against_real_data
            
        Returns:
            Summary of validation performance across all sites
        """
        logger.info("Generating validation summary across all coordinates")
        
        if not validation_results:
            logger.warning("No validation results to summarize")
            return {}
        
        # Aggregate metrics across sites
        all_r2_biomass = []
        all_rmse_biomass = []
        all_mae_biomass = []
        all_r2_carbon = []
        all_rmse_carbon = []
        all_mae_carbon = []
        
        site_summaries = {}
        species_performance = {}
        
        for site_name, result in validation_results.items():
            # Collect aggregate metrics
            biomass_r2 = result.biomass_metrics.get('r2_biomass_correlation', np.nan)
            biomass_rmse = result.biomass_metrics.get('rmse_biomass_kg_m2', np.nan)
            biomass_mae = result.biomass_metrics.get('mae_biomass_kg_m2', np.nan)
            carbon_r2 = result.carbon_metrics.get('r2_carbon_correlation', np.nan)
            carbon_rmse = result.carbon_metrics.get('rmse_carbon_tc_hectare', np.nan)
            carbon_mae = result.carbon_metrics.get('mae_carbon_tc_hectare', np.nan)
            
            if not np.isnan(biomass_r2):
                all_r2_biomass.append(biomass_r2)
            if not np.isnan(biomass_rmse):
                all_rmse_biomass.append(biomass_rmse)
            if not np.isnan(biomass_mae):
                all_mae_biomass.append(biomass_mae)
            if not np.isnan(carbon_r2):
                all_r2_carbon.append(carbon_r2)
            if not np.isnan(carbon_rmse):
                all_rmse_carbon.append(carbon_rmse)
            if not np.isnan(carbon_mae):
                all_mae_carbon.append(carbon_mae)
            
            # Site-specific summary
            site_summaries[site_name] = {
                'coordinate': {
                    'lat': result.coordinate.latitude,
                    'lon': result.coordinate.longitude,
                    'species': result.coordinate.species,
                    'region': result.coordinate.region
                },
                'performance': {
                    'biomass_r2': biomass_r2,
                    'biomass_rmse_kg_m2': biomass_rmse,
                    'carbon_r2': carbon_r2,
                    'carbon_rmse_tc_hectare': carbon_rmse
                },
                'data_points': result.biomass_metrics.get('n_valid_points', 0)
            }
            
            # Species performance tracking
            species = result.coordinate.species
            if species not in species_performance:
                species_performance[species] = {
                    'sites': [],
                    'biomass_r2_values': [],
                    'carbon_r2_values': []
                }
            species_performance[species]['sites'].append(site_name)
            if not np.isnan(biomass_r2):
                species_performance[species]['biomass_r2_values'].append(biomass_r2)
            if not np.isnan(carbon_r2):
                species_performance[species]['carbon_r2_values'].append(carbon_r2)
        
        # Calculate overall performance
        overall_performance = {
            'biomass_metrics': {
                'mean_r2': float(np.mean(all_r2_biomass)) if all_r2_biomass else 0.0,
                'mean_rmse_kg_m2': float(np.mean(all_rmse_biomass)) if all_rmse_biomass else 0.0,
                'mean_mae_kg_m2': float(np.mean(all_mae_biomass)) if all_mae_biomass else 0.0,
                'std_r2': float(np.std(all_r2_biomass)) if all_r2_biomass else 0.0,
                'min_r2': float(np.min(all_r2_biomass)) if all_r2_biomass else 0.0,
                'max_r2': float(np.max(all_r2_biomass)) if all_r2_biomass else 0.0
            },
            'carbon_metrics': {
                'mean_r2': float(np.mean(all_r2_carbon)) if all_r2_carbon else 0.0,
                'mean_rmse_tc_hectare': float(np.mean(all_rmse_carbon)) if all_rmse_carbon else 0.0,
                'mean_mae_tc_hectare': float(np.mean(all_mae_carbon)) if all_mae_carbon else 0.0,
                'std_r2': float(np.std(all_r2_carbon)) if all_r2_carbon else 0.0,
                'min_r2': float(np.min(all_r2_carbon)) if all_r2_carbon else 0.0,
                'max_r2': float(np.max(all_r2_carbon)) if all_r2_carbon else 0.0
            }
        }
        
        # Species performance summary
        species_summary = {}
        for species, data in species_performance.items():
            species_summary[species] = {
                'n_sites': len(data['sites']),
                'sites': data['sites'],
                'mean_biomass_r2': float(np.mean(data['biomass_r2_values'])) if data['biomass_r2_values'] else 0.0,
                'mean_carbon_r2': float(np.mean(data['carbon_r2_values'])) if data['carbon_r2_values'] else 0.0
            }
        
        return {
            'validation_summary': {
                'total_sites_validated': len(validation_results),
                'overall_performance': overall_performance,
                'site_summaries': site_summaries,
                'species_performance': species_summary,
                'validation_coordinates': [
                    {'name': coord.name, 'lat': coord.latitude, 'lon': coord.longitude, 'species': coord.species}
                    for coord in self.VALIDATION_COORDINATES
                ]
            }
        }
    
    def _calculate_prediction_intervals(
        self, 
        predicted: np.ndarray, 
        observed: np.ndarray, 
        confidence_level: float
    ) -> Dict[str, float]:
        """Calculate prediction intervals for uncertainty quantification."""
        try:
            residuals = predicted - observed
            residual_std = np.std(residuals)
            
            # Calculate confidence interval bounds
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, len(residuals) - 1)
            margin_error = t_critical * residual_std
            
            return {
                'lower_bound': float(-margin_error),
                'upper_bound': float(margin_error),
                'confidence_level': confidence_level,
                'residual_std': float(residual_std)
            }
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {e}")
            return {'lower_bound': 0.0, 'upper_bound': 0.0, 'confidence_level': confidence_level, 'residual_std': 0.0}
    
    def _calculate_annual_sequestration_rmse(
        self, 
        carbon_pred: np.ndarray, 
        carbon_obs: np.ndarray
    ) -> float:
        """Calculate RMSE for annual carbon sequestration rates."""
        try:
            # Assume measurements represent annual sequestration
            # Convert to annual rates (tC/hectare/year)
            annual_pred = carbon_pred * 10  # Convert kg/m² to tC/hectare
            annual_obs = carbon_obs * 10
            
            return float(np.sqrt(mean_squared_error(annual_obs, annual_pred)))
        except Exception:
            return 0.0
    
    def _calculate_uncertainty_analysis(
        self, 
        predicted: np.ndarray, 
        observed: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive uncertainty analysis."""
        try:
            residuals = predicted - observed
            relative_errors = residuals / np.where(observed != 0, observed, 1)
            
            return {
                'residual_statistics': {
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'max_absolute_residual': float(np.max(np.abs(residuals)))
                },
                'relative_error_statistics': {
                    'mean_relative_error': float(np.mean(relative_errors)),
                    'std_relative_error': float(np.std(relative_errors)),
                    'max_relative_error': float(np.max(np.abs(relative_errors)))
                },
                'error_distribution': {
                    'q25': float(np.percentile(residuals, 25)),
                    'median': float(np.median(residuals)),
                    'q75': float(np.percentile(residuals, 75)),
                    'iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
                }
            }
        except Exception as e:
            logger.error(f"Error calculating uncertainty analysis: {e}")
            return {}
    
    def _calculate_species_specific_metrics(
        self, 
        species: str, 
        biomass_metrics: Dict[str, float], 
        carbon_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate species-specific performance metrics."""
        try:
            carbon_ratio = self.SPECIES_CARBON_RATIOS.get(species, self.SPECIES_CARBON_RATIOS['Mixed'])
            
            return {
                'species': species,
                'carbon_content_ratio': carbon_ratio,
                'biomass_performance_score': biomass_metrics.get('r2_biomass_correlation', 0) * biomass_metrics.get('pearson_correlation', 0),
                'carbon_performance_score': carbon_metrics.get('r2_carbon_correlation', 0) * carbon_metrics.get('carbon_pearson_correlation', 0),
                'overall_species_score': (biomass_metrics.get('r2_biomass_correlation', 0) + carbon_metrics.get('r2_carbon_correlation', 0)) / 2
            }
        except Exception as e:
            logger.error(f"Error calculating species-specific metrics: {e}")
            return {}
    
    def _calculate_temporal_metrics(
        self, 
        measurement_dates: List[str], 
        predicted: np.ndarray, 
        observed: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate temporal validation metrics if date information is available."""
        try:
            if not measurement_dates or len(measurement_dates) != len(predicted):
                return {'temporal_analysis_available': False}
            
            # Convert dates to datetime
            dates = pd.to_datetime(measurement_dates)
            df = pd.DataFrame({
                'date': dates,
                'predicted': predicted,
                'observed': observed,
                'residual': predicted - observed
            })
            
            # Seasonal analysis
            df['month'] = df['date'].dt.month
            monthly_performance = df.groupby('month')['residual'].agg(['mean', 'std']).to_dict('index')
            
            return {
                'temporal_analysis_available': True,
                'date_range': {
                    'start_date': str(dates.min().date()),
                    'end_date': str(dates.max().date()),
                    'n_measurements': len(dates)
                },
                'monthly_performance': monthly_performance,
                'temporal_correlation': float(stats.spearmanr(range(len(dates)), predicted - observed)[0])
            }
        except Exception as e:
            logger.error(f"Error calculating temporal metrics: {e}")
            return {'temporal_analysis_available': False, 'error': str(e)}
    
    def _empty_metrics_dict(self) -> Dict[str, float]:
        """Return empty metrics dictionary for error cases."""
        return {
            'rmse_biomass_kg_m2': 0.0,
            'mae_biomass_kg_m2': 0.0,
            'r2_biomass_correlation': 0.0,
            'mape_percentage': 0.0,
            'bias_percentage': 0.0,
            'uncertainty_bounds_95': {'lower_bound': 0.0, 'upper_bound': 0.0},
            'pearson_correlation': 0.0,
            'pearson_p_value': 1.0,
            'spearman_correlation': 0.0,
            'spearman_p_value': 1.0,
            'n_valid_points': 0
        }
    
    def _empty_carbon_metrics_dict(self) -> Dict[str, float]:
        """Return empty carbon metrics dictionary for error cases."""
        return {
            'rmse_carbon_tc_hectare': 0.0,
            'mae_carbon_tc_hectare': 0.0,
            'r2_carbon_correlation': 0.0,
            'sequestration_rate_rmse_tc_year': 0.0,
            'carbon_bias_percentage': 0.0,
            'carbon_pearson_correlation': 0.0,
            'carbon_pearson_p_value': 1.0,
            'n_valid_carbon_points': 0
        }


# Factory functions for easy usage
def create_enhanced_validation_metrics() -> EnhancedValidationMetrics:
    """Create enhanced validation metrics calculator."""
    return EnhancedValidationMetrics()


def validate_four_coordinate_sites(
    validation_data: List[BiomassValidationData]
) -> Dict[str, ValidationMetricsResult]:
    """
    Validate model against the four primary validation coordinates.
    
    Args:
        validation_data: Biomass validation data for BC, California, Tasmania, Broughton
        
    Returns:
        Validation results for all four coordinates
    """
    metrics_calculator = create_enhanced_validation_metrics()
    return metrics_calculator.validate_model_predictions_against_real_data(validation_data)


def calculate_validation_summary(
    validation_results: Dict[str, ValidationMetricsResult]
) -> Dict[str, Any]:
    """
    Calculate comprehensive validation summary across all coordinates.
    
    Args:
        validation_results: Results from validate_four_coordinate_sites
        
    Returns:
        Summary of validation performance
    """
    metrics_calculator = create_enhanced_validation_metrics()
    return metrics_calculator.generate_validation_summary(validation_results) 