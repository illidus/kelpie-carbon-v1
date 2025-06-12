"""
Advanced Analytics & Reporting Framework for Kelpie Carbon v1

This module provides comprehensive analytics and reporting capabilities that integrate
all validation, temporal, species, historical, and deep learning analysis into
stakeholder-ready formats.

Components:
    analytics_framework: Core analytics engine integrating all analysis types
    stakeholder_reports: Specialized reports for different audiences
"""

from .analytics_framework import (
    AnalysisRequest,
    AnalysisResult,
    AnalyticsFramework,
    MetricCalculator,
    PerformanceMetrics,
    TrendAnalyzer,
    create_analytics_framework,
)

# Professional Reporting System Components
from .enhanced_satellite_integration import (
    EnhancedSatelliteAnalyzer,
    SatelliteAnalysisResult,
    create_enhanced_satellite_analyzer,
)
from .jupyter_templates import (
    JupyterTemplateManager,
    ScientificAnalysisTemplate,
    TemporalAnalysisTemplate,
    create_jupyter_template_manager,
)
from .mathematical_transparency import (
    CalculationStep,
    CarbonCalculationBreakdown,
    FormulaDocumentation,
    MathematicalTransparencyEngine,
    create_mathematical_transparency_engine,
)
from .professional_report_templates import (
    ProfessionalReportGenerator,
    ProfessionalReportTemplate,
    RegulatoryComplianceReport,
    ReportConfiguration,
    create_professional_report_generator,
)
from .stakeholder_reports import (
    BaseStakeholderReport,
    FirstNationsReport,
    ManagementReport,
    ReportFormat,
    ReportSection,
    ScientificReport,
    create_stakeholder_report,
    create_stakeholder_reporter,
)

__all__ = [
    # Core Framework
    "AnalyticsFramework",
    "AnalysisRequest", 
    "AnalysisResult",
    "MetricCalculator",
    "TrendAnalyzer",
    "PerformanceMetrics",
    "create_analytics_framework",
    
    # Stakeholder Reports
    "FirstNationsReport",
    "ScientificReport",
    "ManagementReport",
    "BaseStakeholderReport",
    "ReportFormat",
    "ReportSection",
    "create_stakeholder_reporter",
    "create_stakeholder_report",

    # Professional Reporting System
    "EnhancedSatelliteAnalyzer",
    "SatelliteAnalysisResult", 
    "create_enhanced_satellite_analyzer",
    "MathematicalTransparencyEngine",
    "FormulaDocumentation",
    "CalculationStep",
    "CarbonCalculationBreakdown",
    "create_mathematical_transparency_engine",
    "JupyterTemplateManager",
    "ScientificAnalysisTemplate",
    "TemporalAnalysisTemplate", 
    "create_jupyter_template_manager",
    "ProfessionalReportGenerator",
    "ReportConfiguration",
    "ProfessionalReportTemplate",
    "RegulatoryComplianceReport",
    "create_professional_report_generator"
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Kelpie Carbon v1 Development Team"
__description__ = "Advanced Analytics & Reporting Framework for Kelp Detection" 
