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
    AnalyticsFramework,
    AnalysisRequest,
    AnalysisResult,
    MetricCalculator,
    TrendAnalyzer,
    PerformanceMetrics,
    create_analytics_framework,
)

from .stakeholder_reports import (
    FirstNationsReport,
    ScientificReport,
    ManagementReport,
    BaseStakeholderReport,
    ReportFormat,
    ReportSection,
    create_stakeholder_reporter,
    create_stakeholder_report,
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
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Kelpie Carbon v1 Development Team"
__description__ = "Advanced Analytics & Reporting Framework for Kelp Detection" 