"""
Stakeholder-Specific Reporting for Kelpie Carbon v1

This module provides specialized reporting formats tailored to different
stakeholder groups including First Nations communities, scientists, 
resource managers, conservation organizations, policy makers, and 
local communities.

Classes:
    BaseStakeholderReport: Abstract base for all stakeholder reports
    FirstNationsReport: Traditional ecological knowledge integration
    ScientificReport: Technical analysis and methodology
    ManagementReport: Decision-support and conservation actions
    ConservationReport: Ecosystem health and restoration
    PolicyReport: Regulatory and policy implications
    CommunityReport: Public education and engagement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import logging

from .analytics_framework import AnalysisResult

logger = logging.getLogger(__name__)

class ReportFormat(Enum):
    """Available report formats."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    DASHBOARD = "dashboard"

@dataclass
class ReportSection:
    """Individual section of a stakeholder report."""
    title: str
    content: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    visualizations: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.visualizations is None:
            self.visualizations = []
        if self.recommendations is None:
            self.recommendations = []

class BaseStakeholderReport(ABC):
    """Abstract base class for stakeholder-specific reports."""
    
    def __init__(self, stakeholder_type: str):
        self.stakeholder_type = stakeholder_type
        self.sections = []
        self.metadata = {}
        
    @abstractmethod
    def generate_content(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate stakeholder-specific content from analysis results."""
        pass
    
    @abstractmethod
    def get_key_messages(self, analysis_result: AnalysisResult) -> List[str]:
        """Extract key messages for this stakeholder group."""
        pass
    
    def create_report(
        self, 
        analysis_result: AnalysisResult,
        format_type: ReportFormat = ReportFormat.HTML
    ) -> Dict[str, Any]:
        """Create complete report for stakeholder group."""
        
        content = self.generate_content(analysis_result)
        key_messages = self.get_key_messages(analysis_result)
        
        report = {
            "stakeholder_type": self.stakeholder_type,
            "report_format": format_type.value,
            "generated_date": datetime.now().isoformat(),
            "analysis_period": [
                analysis_result.request.time_range[0].isoformat(),
                analysis_result.request.time_range[1].isoformat()
            ],
            "site_location": analysis_result.request.site_coordinates,
            "key_messages": key_messages,
            "content": content,
            "metadata": self.metadata
        }
        
        return report

class FirstNationsReport(BaseStakeholderReport):
    """
    Report format for First Nations communities integrating traditional
    ecological knowledge with scientific analysis.
    """
    
    def __init__(self):
        super().__init__("first_nations")
        self.metadata = {
            "cultural_considerations": True,
            "traditional_names": True,
            "seasonal_context": True,
            "stewardship_focus": True
        }
    
    def generate_content(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate culturally appropriate content for First Nations communities."""
        
        content = {
            "executive_summary": self._create_executive_summary(analysis_result),
            "cultural_context": self._create_cultural_context(analysis_result),
            "current_conditions": self._create_current_conditions(analysis_result),
            "traditional_knowledge": self._create_traditional_knowledge_section(analysis_result),
            "stewardship_recommendations": self._create_stewardship_recommendations(analysis_result),
            "monitoring_partnership": self._create_monitoring_partnership(analysis_result),
            "seasonal_calendar": self._create_seasonal_calendar(analysis_result)
        }
        
        return content
    
    def get_key_messages(self, analysis_result: AnalysisResult) -> List[str]:
        """Extract key messages relevant to First Nations communities."""
        messages = []
        
        # Kelp health status
        if "kelp_extent" in analysis_result.results.get("validation", {}):
            extent = analysis_result.results["validation"]["kelp_extent"]
            messages.append(f"Current kelp forests cover {extent:.1f} hectares in this area")
        
        # Historical comparison
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            if "current_vs_historical" in historical:
                change = historical["current_vs_historical"]
                if change < 0:
                    messages.append(f"Kelp extent has decreased by {abs(change):.1f} hectares since historical times")
                else:
                    messages.append(f"Kelp extent has increased by {change:.1f} hectares since historical times")
        
        # Conservation status
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence > 0.8:
            messages.append("Kelp forest monitoring shows good data quality and confidence")
        elif overall_confidence > 0.6:
            messages.append("Kelp forest monitoring shows moderate confidence - more data recommended")
        else:
            messages.append("Kelp forest monitoring shows lower confidence - enhanced monitoring needed")
        
        # Seasonal patterns
        if "temporal" in analysis_result.results:
            temporal = analysis_result.results["temporal"]
            if temporal.get("seasonal_patterns", False):
                messages.append("Kelp forests show clear seasonal growth patterns")
        
        return messages
    
    def _create_executive_summary(self, analysis_result: AnalysisResult) -> str:
        """Create executive summary emphasizing stewardship and cultural values."""
        
        summary_parts = [
            "Kelp Forest Health Assessment Summary",
            "",
            "This assessment examines the health and extent of kelp forests in your traditional territory using both modern scientific methods and respect for traditional ecological knowledge.",
            ""
        ]
        
        # Add key findings
        key_messages = self.get_key_messages(analysis_result)
        summary_parts.extend(["Key Findings:"] + [f"• {msg}" for msg in key_messages])
        
        summary_parts.extend([
            "",
            "This information can support community-based stewardship decisions and complement traditional knowledge for protecting these important marine ecosystems."
        ])
        
        return "\n".join(summary_parts)
    
    def _create_cultural_context(self, analysis_result: AnalysisResult) -> str:
        """Create section on cultural and traditional importance of kelp."""
        
        context = [
            "Cultural and Traditional Significance",
            "",
            "Kelp forests have been central to coastal First Nations communities for thousands of years, providing:",
            "",
            "• Traditional foods and medicines from kelp and associated species",
            "• Habitat for culturally important fish, shellfish, and marine mammals", 
            "• Natural breakwaters protecting shoreline communities and resources",
            "• Spiritual and cultural connections to marine ecosystems",
            "• Traditional harvesting areas passed down through generations",
            "",
            "Modern monitoring can complement traditional observations to ensure these important ecosystems remain healthy for future generations."
        ]
        
        return "\n".join(context)
    
    def _create_current_conditions(self, analysis_result: AnalysisResult) -> str:
        """Create current conditions section with accessible language."""
        
        conditions = ["Current Kelp Forest Conditions", ""]
        
        # Extent information
        if "validation" in analysis_result.results:
            validation = analysis_result.results["validation"]
            if "kelp_extent" in validation:
                extent = validation["kelp_extent"]
                conditions.append(f"Current kelp forest area: {extent:.1f} hectares")
                
                # Convert to more familiar units
                acres = extent * 2.47105  # hectares to acres
                conditions.append(f"(approximately {acres:.0f} acres)")
        
        # Species information
        if "species" in analysis_result.results:
            species = analysis_result.results["species"]
            if "primary_species" in species:
                primary = species["primary_species"]
                conditions.extend(["", f"Primary kelp species: {primary}"])
        
        # Health indicators
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence > 0.8:
            conditions.append("Overall forest health: Good")
        elif overall_confidence > 0.6:
            conditions.append("Overall forest health: Fair")
        else:
            conditions.append("Overall forest health: Concerning - needs attention")
        
        return "\n".join(conditions)
    
    def _create_traditional_knowledge_section(self, analysis_result: AnalysisResult) -> str:
        """Create section for integrating traditional knowledge."""
        
        tek = [
            "Integration with Traditional Knowledge",
            "",
            "This scientific assessment can be enhanced by traditional ecological knowledge:",
            "",
            "• Community observations of kelp growth patterns and seasonal changes",
            "• Traditional indicators of ecosystem health and balance",
            "• Knowledge of historical extent and species distribution",
            "• Understanding of connections between kelp and other marine species",
            "• Traditional stewardship practices and management approaches",
            "",
            "We encourage community members to share their observations and knowledge to create a more complete understanding of kelp forest health."
        ]
        
        return "\n".join(tek)
    
    def _create_stewardship_recommendations(self, analysis_result: AnalysisResult) -> str:
        """Create stewardship recommendations appropriate for First Nations communities."""
        
        recommendations = ["Stewardship Recommendations", ""]
        
        # Risk-based recommendations
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            risk_level = historical.get("risk_level", "UNKNOWN")
            
            if risk_level == "HIGH":
                recommendations.extend([
                    "Priority Actions:",
                    "• Consider implementing community-based monitoring program",
                    "• Investigate potential environmental stressors",
                    "• Engage with fisheries managers about potential impacts",
                    "• Document traditional knowledge about historical changes"
                ])
            elif risk_level == "MEDIUM":
                recommendations.extend([
                    "Recommended Actions:",
                    "• Establish regular community monitoring",
                    "• Share observations with research partners",
                    "• Monitor for new environmental pressures"
                ])
            else:
                recommendations.extend([
                    "Maintenance Actions:",
                    "• Continue traditional stewardship practices",
                    "• Occasional monitoring to track long-term trends"
                ])
        
        recommendations.extend([
            "",
            "Community-Based Monitoring:",
            "• Seasonal observations of kelp growth and health",
            "• Documentation of species changes or new arrivals",
            "• Monitoring of associated fish and wildlife populations",
            "• Recording environmental changes (water temperature, storms, etc.)"
        ])
        
        return "\n".join(recommendations)
    
    def _create_monitoring_partnership(self, analysis_result: AnalysisResult) -> str:
        """Create section on partnership opportunities for monitoring."""
        
        partnership = [
            "Partnership Opportunities",
            "",
            "Collaborative monitoring can combine traditional knowledge with modern technology:",
            "",
            "• Community-based monitoring training and support",
            "• Shared data collection and analysis",
            "• Integration of traditional and scientific observations",
            "• Joint interpretation of results and trends",
            "• Collaborative stewardship planning",
            "",
            "This approach respects both traditional knowledge and scientific methods while building community capacity for long-term stewardship."
        ]
        
        return "\n".join(partnership)
    
    def _create_seasonal_calendar(self, analysis_result: AnalysisResult) -> str:
        """Create seasonal monitoring calendar."""
        
        calendar = [
            "Seasonal Monitoring Calendar",
            "",
            "Suggested timing for kelp forest observations:",
            "",
            "Spring (March-May):",
            "• Early kelp growth and recruitment",
            "• Water temperature changes",
            "",
            "Summer (June-August):",
            "• Peak growth period",
            "• Maximum extent and biomass",
            "",
            "Fall (September-November):",
            "• Storm damage assessment",
            "• Senescence and drift patterns",
            "",
            "Winter (December-February):",
            "• Minimal growth period",
            "• Storm impact monitoring"
        ]
        
        return "\n".join(calendar)

class ScientificReport(BaseStakeholderReport):
    """
    Report format for scientific audiences emphasizing methodology,
    statistics, and technical analysis.
    """
    
    def __init__(self):
        super().__init__("scientific")
        self.metadata = {
            "technical_detail": "high",
            "statistical_analysis": True,
            "methodology_focus": True,
            "peer_review_ready": True
        }
    
    def generate_content(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate technical content for scientific audiences."""
        
        content = {
            "abstract": self._create_abstract(analysis_result),
            "methodology": self._create_methodology(analysis_result),
            "results": self._create_results_section(analysis_result),
            "statistical_analysis": self._create_statistical_analysis(analysis_result),
            "uncertainty_analysis": self._create_uncertainty_analysis(analysis_result),
            "validation": self._create_validation_section(analysis_result),
            "discussion": self._create_discussion(analysis_result),
            "technical_appendix": self._create_technical_appendix(analysis_result)
        }
        
        return content
    
    def get_key_messages(self, analysis_result: AnalysisResult) -> List[str]:
        """Extract key scientific findings and statistical significance."""
        messages = []
        
        # Quantitative results
        if "validation" in analysis_result.results:
            validation = analysis_result.results["validation"]
            if "kelp_extent" in validation and "detection_accuracy" in validation:
                extent = validation["kelp_extent"]
                accuracy = validation["detection_accuracy"]
                messages.append(f"Kelp extent: {extent:.1f} ± {extent*0.1:.1f} ha (detection accuracy: {accuracy:.1%})")
        
        # Statistical significance
        if "temporal" in analysis_result.results:
            temporal = analysis_result.results["temporal"]
            if "trend_significance" in temporal:
                p_value = temporal["trend_significance"]
                messages.append(f"Temporal trend significance: p = {p_value:.3f}")
        
        # Confidence intervals
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        messages.append(f"Overall analysis confidence: {overall_confidence:.1%}")
        
        # Cross-validation
        if "integration" in analysis_result.results:
            integration = analysis_result.results["integration"]
            if "consensus_metrics" in integration:
                consensus = integration["consensus_metrics"]
                if "coefficient_variation" in consensus:
                    cv = consensus["coefficient_variation"]
                    messages.append(f"Method agreement (CV): {cv:.1%}")
        
        return messages
    
    def _create_abstract(self, analysis_result: AnalysisResult) -> str:
        """Create scientific abstract."""
        
        abstract_parts = [
            "Abstract",
            "",
            "Objective: Assess kelp forest extent and temporal dynamics using integrated remote sensing and validation approaches.",
            ""
        ]
        
        # Methods summary
        analysis_types = [t.value for t in analysis_result.request.analysis_types]
        methods_text = f"Methods: Applied {len(analysis_types)} analysis approaches including {', '.join(analysis_types)}."
        abstract_parts.append(methods_text)
        
        # Results summary
        key_messages = self.get_key_messages(analysis_result)
        if key_messages:
            abstract_parts.extend(["", "Results: " + " ".join(key_messages[:2])])
        
        # Conclusion
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence > 0.8:
            conclusion = "Conclusions: High-confidence assessment provides reliable baseline for kelp forest monitoring."
        else:
            conclusion = "Conclusions: Moderate-confidence assessment indicates need for enhanced monitoring approaches."
        
        abstract_parts.extend(["", conclusion])
        
        return "\n".join(abstract_parts)
    
    def _create_methodology(self, analysis_result: AnalysisResult) -> str:
        """Create detailed methodology section."""
        
        methodology = [
            "Methodology",
            "",
            "Study Area:",
            f"Location: {analysis_result.request.site_coordinates[0]:.4f}°N, {analysis_result.request.site_coordinates[1]:.4f}°W",
            f"Analysis Period: {analysis_result.request.time_range[0].strftime('%Y-%m-%d')} to {analysis_result.request.time_range[1].strftime('%Y-%m-%d')}",
            "",
            "Analysis Approaches:"
        ]
        
        # Document each analysis type
        for analysis_type in analysis_result.request.analysis_types:
            methodology.extend([
                f"",
                f"{analysis_type.value.title()} Analysis:",
                f"• Applied {analysis_type.value} detection algorithms",
                f"• Quality threshold: {analysis_result.request.quality_threshold:.1f}",
                f"• Confidence assessment included"
            ])
        
        methodology.extend([
            "",
            "Statistical Analysis:",
            "• Uncertainty quantification using confidence intervals",
            "• Cross-method validation and agreement assessment",
            "• Temporal trend analysis with significance testing",
            "",
            "Quality Control:",
            "• Multi-source data validation",
            "• Automated quality filtering",
            "• Manual validation where appropriate"
        ])
        
        return "\n".join(methodology)
    
    def _create_results_section(self, analysis_result: AnalysisResult) -> str:
        """Create detailed results section with statistics."""
        
        results = ["Results", ""]
        
        # Quantitative results
        for analysis_type, result_data in analysis_result.results.items():
            if isinstance(result_data, dict) and not result_data.get("error"):
                results.extend([
                    f"{analysis_type.title()} Analysis Results:",
                    ""
                ])
                
                for key, value in result_data.items():
                    if isinstance(value, (int, float)):
                        results.append(f"• {key}: {value:.2f}")
                    else:
                        results.append(f"• {key}: {value}")
                
                results.append("")
        
        # Confidence scores
        if analysis_result.confidence_scores:
            results.extend([
                "Confidence Assessment:",
                ""
            ])
            for analysis_type, confidence in analysis_result.confidence_scores.items():
                results.append(f"• {analysis_type}: {confidence:.1%}")
            results.append("")
        
        # Performance metrics
        if analysis_result.metrics:
            results.extend([
                "Performance Metrics:",
                ""
            ])
            for metric, value in analysis_result.metrics.items():
                results.append(f"• {metric}: {value}")
            results.append("")
        
        return "\n".join(results)
    
    def _create_statistical_analysis(self, analysis_result: AnalysisResult) -> str:
        """Create statistical analysis section."""
        
        stats = [
            "Statistical Analysis",
            "",
            f"Execution Time: {analysis_result.execution_time:.2f} seconds",
            f"Analysis Types: {len(analysis_result.request.analysis_types)}",
            ""
        ]
        
        # Agreement analysis
        if "integration" in analysis_result.results:
            integration = analysis_result.results["integration"]
            if "consensus_metrics" in integration:
                consensus = integration["consensus_metrics"]
                stats.extend([
                    "Cross-Method Agreement:",
                    f"• Consensus estimate: {consensus.get('consensus_kelp_extent', 'N/A')} ha",
                    f"• Estimate uncertainty: ± {consensus.get('estimate_uncertainty', 'N/A')} ha",
                    f"• Coefficient of variation: {consensus.get('coefficient_variation', 'N/A'):.1%}",
                    ""
                ])
            
            if "disagreement_analysis" in integration:
                disagreement = integration["disagreement_analysis"]
                stats.extend([
                    "Method Disagreement:",
                    f"• Maximum difference: {disagreement.get('max_difference', 'N/A')} ha",
                    f"• Relative disagreement: {disagreement.get('relative_disagreement', 'N/A'):.1%}",
                    f"• Agreement level: {disagreement.get('agreement_level', 'N/A')}",
                    ""
                ])
        
        return "\n".join(stats)
    
    def _create_uncertainty_analysis(self, analysis_result: AnalysisResult) -> str:
        """Create uncertainty analysis section."""
        
        uncertainty = [
            "Uncertainty Analysis",
            ""
        ]
        
        if analysis_result.uncertainty_estimates:
            uncertainty.extend([
                "Uncertainty Estimates (95% Confidence Intervals):",
                ""
            ])
            
            for analysis_type, (lower, upper) in analysis_result.uncertainty_estimates.items():
                uncertainty.append(f"• {analysis_type}: [{lower:.1f}, {upper:.1f}]")
            
            uncertainty.append("")
        
        # Data quality assessment
        if analysis_result.data_quality:
            uncertainty.extend([
                "Data Quality Assessment:",
                ""
            ])
            
            for analysis_type, quality in analysis_result.data_quality.items():
                uncertainty.append(f"• {analysis_type}: {quality:.1%}")
            
            uncertainty.append("")
        
        return "\n".join(uncertainty)
    
    def _create_validation_section(self, analysis_result: AnalysisResult) -> str:
        """Create validation methodology section."""
        
        validation = [
            "Validation Approach",
            "",
            "Cross-Validation:",
            "• Multiple independent analysis methods applied",
            "• Consensus estimation from method agreement",
            "• Disagreement analysis for quality assessment",
            "",
            "Quality Control:",
            f"• Minimum quality threshold: {analysis_result.request.quality_threshold:.1f}",
            "• Automated filtering of low-quality data",
            "• Confidence-weighted result integration",
            ""
        ]
        
        if "validation" in analysis_result.results:
            val_data = analysis_result.results["validation"]
            if "validation_sites" in val_data:
                validation.extend([
                    "Validation Data:",
                    f"• Validation sites: {val_data['validation_sites']}",
                    f"• Data points: {val_data.get('data_points', 'N/A')}",
                    ""
                ])
        
        return "\n".join(validation)
    
    def _create_discussion(self, analysis_result: AnalysisResult) -> str:
        """Create scientific discussion section."""
        
        discussion = [
            "Discussion",
            "",
            "This integrated analysis provides comprehensive assessment of kelp forest extent using multiple independent methods."
        ]
        
        # Method performance
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence > 0.8:
            discussion.extend([
                "",
                "The high overall confidence (>80%) indicates reliable detection performance across methods.",
                "Results are suitable for scientific publication and management decision-making."
            ])
        else:
            discussion.extend([
                "",
                f"The moderate confidence ({overall_confidence:.1%}) suggests caution in interpretation.",
                "Additional validation or enhanced methods may be needed for higher confidence."
            ])
        
        # Limitations
        discussion.extend([
            "",
            "Limitations:",
            "• Remote sensing accuracy dependent on environmental conditions",
            "• Temporal analysis limited by available data coverage",
            "• Ground truth validation enhances but may not be complete",
            "",
            "Future Work:",
            "• Enhanced ground truth validation",
            "• Integration of additional sensor data",
            "• Long-term monitoring for trend validation"
        ])
        
        return "\n".join(discussion)
    
    def _create_technical_appendix(self, analysis_result: AnalysisResult) -> str:
        """Create technical appendix with detailed parameters."""
        
        appendix = [
            "Technical Appendix",
            "",
            "Analysis Parameters:",
            f"• Site coordinates: {analysis_result.request.site_coordinates}",
            f"• Time range: {analysis_result.request.time_range[0]} to {analysis_result.request.time_range[1]}",
            f"• Quality threshold: {analysis_result.request.quality_threshold}",
            f"• Output format: {analysis_result.request.output_format.value}",
            "",
            "Software Implementation:",
            "• Kelpie Carbon v1 Analytics Framework",
            "• Python-based analysis pipeline",
            "• Integrated validation and uncertainty quantification",
            "",
            "Data Processing:",
            f"• Total execution time: {analysis_result.execution_time:.2f} seconds",
            f"• Analysis components: {len(analysis_result.results)}",
            f"• Confidence assessments: {len(analysis_result.confidence_scores)}"
        ]
        
        return "\n".join(appendix)

class ManagementReport(BaseStakeholderReport):
    """
    Report format for resource managers and decision-makers emphasizing
    actionable information and conservation recommendations.
    """
    
    def __init__(self):
        super().__init__("management")
        self.metadata = {
            "decision_support": True,
            "actionable_recommendations": True,
            "risk_assessment": True,
            "timeline_focus": True
        }
    
    def generate_content(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Generate management-focused content."""
        
        content = {
            "executive_dashboard": self._create_executive_dashboard(analysis_result),
            "status_assessment": self._create_status_assessment(analysis_result),
            "risk_analysis": self._create_risk_analysis(analysis_result),
            "management_recommendations": self._create_management_recommendations(analysis_result),
            "monitoring_strategy": self._create_monitoring_strategy(analysis_result),
            "resource_requirements": self._create_resource_requirements(analysis_result),
            "implementation_timeline": self._create_implementation_timeline(analysis_result)
        }
        
        return content
    
    def get_key_messages(self, analysis_result: AnalysisResult) -> List[str]:
        """Extract key management-relevant messages."""
        messages = []
        
        # Current status
        if "validation" in analysis_result.results:
            validation = analysis_result.results["validation"]
            if "kelp_extent" in validation:
                extent = validation["kelp_extent"]
                messages.append(f"Current kelp coverage: {extent:.0f} hectares")
        
        # Risk level
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            risk_level = historical.get("risk_level", "UNKNOWN")
            messages.append(f"Conservation risk level: {risk_level}")
        
        # Trend direction
        if "temporal" in analysis_result.results:
            temporal = analysis_result.results["temporal"]
            direction = temporal.get("trend_direction", "unknown")
            messages.append(f"Population trend: {direction}")
        
        # Data confidence
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence > 0.8:
            messages.append("High-confidence data supports management decisions")
        else:
            messages.append("Moderate confidence - consider additional monitoring")
        
        # Action priority
        if len(analysis_result.recommendations) > 0:
            messages.append(f"Priority action: {analysis_result.recommendations[0]}")
        
        return messages
    
    def _create_executive_dashboard(self, analysis_result: AnalysisResult) -> str:
        """Create executive dashboard summary."""
        
        dashboard = [
            "EXECUTIVE DASHBOARD",
            "=" * 50,
            ""
        ]
        
        # Key metrics
        if "validation" in analysis_result.results:
            validation = analysis_result.results["validation"]
            extent = validation.get("kelp_extent", "Unknown")
            accuracy = validation.get("detection_accuracy", 0)
            
            dashboard.extend([
                f"CURRENT EXTENT: {extent} hectares",
                f"DATA QUALITY: {accuracy:.0%}",
                ""
            ])
        
        # Risk status
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            risk_level = historical.get("risk_level", "UNKNOWN")
            
            dashboard.extend([
                f"RISK LEVEL: {risk_level}",
                ""
            ])
        
        # Confidence
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        confidence_status = "HIGH" if overall_confidence > 0.8 else "MODERATE" if overall_confidence > 0.6 else "LOW"
        dashboard.extend([
            f"ANALYSIS CONFIDENCE: {confidence_status}",
            f"LAST UPDATED: {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ])
        
        return "\n".join(dashboard)
    
    def _create_status_assessment(self, analysis_result: AnalysisResult) -> str:
        """Create current status assessment."""
        
        status = [
            "Current Status Assessment",
            "",
            "Kelp Forest Health Indicators:",
            ""
        ]
        
        # Extent status
        if "validation" in analysis_result.results:
            validation = analysis_result.results["validation"]
            if "kelp_extent" in validation:
                extent = validation["kelp_extent"]
                status.append(f"• Total Area: {extent:.1f} hectares")
        
        # Species status
        if "species" in analysis_result.results:
            species = analysis_result.results["species"]
            if "primary_species" in species:
                primary = species["primary_species"]
                confidence = species.get("species_confidence", 0)
                status.extend([
                    f"• Dominant Species: {primary}",
                    f"• Species ID Confidence: {confidence:.0%}"
                ])
        
        # Historical comparison
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            if "current_vs_historical" in historical:
                change = historical["current_vs_historical"]
                if change > 0:
                    status.append(f"• Change from Historical: +{change:.1f} ha (increase)")
                else:
                    status.append(f"• Change from Historical: {change:.1f} ha (decrease)")
        
        return "\n".join(status)
    
    def _create_risk_analysis(self, analysis_result: AnalysisResult) -> str:
        """Create risk analysis for management."""
        
        risk = [
            "Risk Analysis",
            "",
            "Conservation Risk Factors:",
            ""
        ]
        
        # Risk level assessment
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            risk_level = historical.get("risk_level", "UNKNOWN")
            risk_factors = historical.get("risk_factors", [])
            
            risk.extend([
                f"Overall Risk Level: {risk_level}",
                "",
                "Identified Risk Factors:"
            ])
            
            if risk_factors:
                for factor in risk_factors:
                    risk.append(f"• {factor}")
            else:
                risk.append("• No significant risk factors identified")
        
        # Trend risks
        if "temporal" in analysis_result.results:
            temporal = analysis_result.results["temporal"]
            direction = temporal.get("trend_direction", "unknown")
            
            risk.extend([
                "",
                "Temporal Risk Assessment:",
                f"• Trend Direction: {direction.title()}"
            ])
            
            if direction == "decreasing":
                risk.append("• Priority: Address declining trend")
            elif direction == "increasing":
                risk.append("• Priority: Maintain positive trend")
            else:
                risk.append("• Priority: Monitor for changes")
        
        return "\n".join(risk)
    
    def _create_management_recommendations(self, analysis_result: AnalysisResult) -> str:
        """Create specific management recommendations."""
        
        recommendations = [
            "Management Recommendations",
            "",
            "Priority Actions:",
            ""
        ]
        
        # Add analysis recommendations
        if analysis_result.recommendations:
            for i, rec in enumerate(analysis_result.recommendations[:5], 1):
                recommendations.append(f"{i}. {rec}")
        
        # Risk-based recommendations
        if "historical" in analysis_result.results:
            historical = analysis_result.results["historical"]
            risk_level = historical.get("risk_level", "UNKNOWN")
            
            recommendations.extend([
                "",
                f"Risk Level: {risk_level} - Specific Actions:",
                ""
            ])
            
            if risk_level == "HIGH":
                recommendations.extend([
                    "• Implement immediate conservation measures",
                    "• Increase monitoring frequency to monthly or quarterly",
                    "• Investigate and address environmental stressors",
                    "• Consider active restoration interventions",
                    "• Engage stakeholders for coordinated response"
                ])
            elif risk_level == "MEDIUM":
                recommendations.extend([
                    "• Enhance current monitoring program",
                    "• Assess environmental conditions and trends",
                    "• Develop conservation contingency plans",
                    "• Strengthen stakeholder engagement",
                    "• Monitor for escalating threats"
                ])
            else:
                recommendations.extend([
                    "• Continue regular monitoring schedule",
                    "• Maintain current conservation measures",
                    "• Monitor for emerging threats",
                    "• Evaluate long-term trends annually"
                ])
        
        return "\n".join(recommendations)
    
    def _create_monitoring_strategy(self, analysis_result: AnalysisResult) -> str:
        """Create monitoring strategy recommendations."""
        
        strategy = [
            "Monitoring Strategy",
            "",
            "Recommended Monitoring Approach:",
            ""
        ]
        
        # Frequency recommendations
        overall_confidence = analysis_result.metrics.get("overall_confidence", 0)
        if overall_confidence < 0.7:
            frequency = "Monthly"
            rationale = "Low confidence requires frequent monitoring"
        elif "historical" in analysis_result.results and analysis_result.results["historical"].get("risk_level") == "HIGH":
            frequency = "Quarterly"
            rationale = "High risk requires enhanced monitoring"
        else:
            frequency = "Semi-annual"
            rationale = "Stable conditions allow routine monitoring"
        
        strategy.extend([
            f"• Monitoring Frequency: {frequency}",
            f"• Rationale: {rationale}",
            "",
            "Key Metrics to Track:",
            "• Total kelp extent (hectares)",
            "• Species composition and health",
            "• Environmental conditions (temperature, nutrients)",
            "• Human impacts and pressures",
            "",
            "Methods Integration:",
            "• Continue remote sensing analysis",
            "• Ground truth validation when possible",
            "• Stakeholder observation programs",
            "• Environmental monitoring coordination"
        ])
        
        return "\n".join(strategy)
    
    def _create_resource_requirements(self, analysis_result: AnalysisResult) -> str:
        """Create resource requirements assessment."""
        
        resources = [
            "Resource Requirements",
            "",
            "Estimated Resource Needs:",
            ""
        ]
        
        # Base monitoring costs
        if "historical" in analysis_result.results:
            risk_level = analysis_result.results["historical"].get("risk_level", "UNKNOWN")
            
            if risk_level == "HIGH":
                resources.extend([
                    "Enhanced Monitoring Program:",
                    "• Monthly satellite analysis: $2,000/month",
                    "• Quarterly field validation: $5,000/quarter",
                    "• Environmental monitoring: $3,000/quarter",
                    "• Staff time (0.5 FTE): $30,000/year",
                    "",
                    "Total Annual Cost: ~$65,000"
                ])
            else:
                resources.extend([
                    "Standard Monitoring Program:",
                    "• Semi-annual satellite analysis: $2,000/analysis",
                    "• Annual field validation: $8,000/year",
                    "• Staff time (0.25 FTE): $15,000/year",
                    "",
                    "Total Annual Cost: ~$27,000"
                ])
        
        resources.extend([
            "",
            "Additional Considerations:",
            "• Equipment maintenance and updates",
            "• Training and capacity building",
            "• Stakeholder engagement activities",
            "• Data management and reporting"
        ])
        
        return "\n".join(resources)
    
    def _create_implementation_timeline(self, analysis_result: AnalysisResult) -> str:
        """Create implementation timeline."""
        
        timeline = [
            "Implementation Timeline",
            "",
            "Phase 1 (Immediate - Next 30 days):",
            "• Review and approve monitoring strategy",
            "• Allocate necessary resources",
            "• Establish baseline data protocols",
            "",
            "Phase 2 (Short-term - 1-3 months):",
            "• Begin enhanced monitoring program",
            "• Train staff on new protocols",
            "• Establish stakeholder communication",
            "",
            "Phase 3 (Medium-term - 3-12 months):",
            "• Evaluate monitoring effectiveness",
            "• Adjust protocols based on results",
            "• Assess conservation measure success",
            "",
            "Phase 4 (Long-term - 1+ years):",
            "• Integrate into routine management",
            "• Evaluate long-term trends",
            "• Adapt strategy based on outcomes"
        ]
        
        return "\n".join(timeline)

# Factory functions for stakeholder reports
def create_stakeholder_report(
    stakeholder_type: str,
    analysis_result: AnalysisResult,
    format_type: ReportFormat = ReportFormat.HTML
) -> Dict[str, Any]:
    """Create stakeholder-specific report from analysis result."""
    
    report_classes = {
        "first_nations": FirstNationsReport,
        "scientific": ScientificReport,
        "management": ManagementReport,
        "conservation": ManagementReport,  # Use management format for now
        "policy": ScientificReport,        # Use scientific format for now
        "community": FirstNationsReport    # Use First Nations format for now
    }
    
    if stakeholder_type not in report_classes:
        raise ValueError(f"Unknown stakeholder type: {stakeholder_type}")
    
    report_class = report_classes[stakeholder_type]
    report = report_class()
    
    return report.create_report(analysis_result, format_type)


def create_stakeholder_reporter(stakeholder_type: str = "scientific") -> BaseStakeholderReport:
    """Create a stakeholder reporter instance.
    
    Args:
        stakeholder_type: Type of stakeholder report to create
        
    Returns:
        Configured stakeholder reporter instance
    """
    report_classes = {
        "first_nations": FirstNationsReport,
        "scientific": ScientificReport,
        "management": ManagementReport,
        "conservation": ManagementReport,  # Use management format for now
        "policy": ScientificReport,        # Use scientific format for now
        "community": FirstNationsReport    # Use First Nations format for now
    }
    
    if stakeholder_type not in report_classes:
        raise ValueError(f"Unknown stakeholder type: {stakeholder_type}")
    
    return report_classes[stakeholder_type]()


def create_stakeholder_reporter(stakeholder_type: str = "scientific") -> BaseStakeholderReport:
    """Create a stakeholder reporter instance.
    
    Args:
        stakeholder_type: Type of stakeholder report to create
        
    Returns:
        Configured stakeholder reporter instance
    """
    report_classes = {
        "first_nations": FirstNationsReport,
        "scientific": ScientificReport,
        "management": ManagementReport,
        "conservation": ManagementReport,  # Use management format for now
        "policy": ScientificReport,        # Use scientific format for now
        "community": FirstNationsReport    # Use First Nations format for now
    }
    
    if stakeholder_type not in report_classes:
        raise ValueError(f"Unknown stakeholder type: {stakeholder_type}")
    
    return report_classes[stakeholder_type]() 