#!/usr/bin/env python3
"""
Analytics Framework Demonstration Script

This script demonstrates the comprehensive analytics and reporting capabilities
of the Kelpie Carbon v1 Advanced Analytics Framework.

Features:
- Multi-analysis integration (validation, temporal, species, historical, etc.)
- Stakeholder-specific reporting (First Nations, Scientific, Management)
- Performance monitoring and system health assessment
- Interactive demonstration modes
- Realistic test scenarios and data

Usage:
    python scripts/test_analytics_framework_demo.py [mode]
    
Available modes:
    - basic: Basic analytics framework demonstration
    - stakeholder: Stakeholder reporting demonstration  
    - performance: Performance monitoring demonstration
    - integration: Cross-analysis integration demonstration
    - interactive: Interactive exploration mode
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

try:
    from src.kelpie_carbon_v1.analytics.analytics_framework import (
        AnalyticsFramework,
        AnalysisRequest,
        AnalysisType,
        OutputFormat,
        MetricCalculator,
        TrendAnalyzer,
        PerformanceMetrics,
        create_analysis_request,
        quick_analysis
    )
    
    from src.kelpie_carbon_v1.analytics.stakeholder_reports import (
        FirstNationsReport,
        ScientificReport,
        ManagementReport,
        ReportFormat,
        create_stakeholder_report
    )
    
    print("✅ Successfully imported analytics framework components")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Note: This is expected in development environment without full dependencies")
    sys.exit(1)

class AnalyticsFrameworkDemo:
    """Comprehensive demonstration of analytics framework capabilities."""
    
    def __init__(self):
        """Initialize demo with framework and test data."""
        self.framework = AnalyticsFramework()
        self.metric_calculator = MetricCalculator()
        self.trend_analyzer = TrendAnalyzer()
        
        # Test sites with realistic characteristics
        self.test_sites = {
            "broughton_archipelago": {
                "name": "Broughton Archipelago (UVic Research Site)",
                "coordinates": (50.0833, -126.1667),
                "description": "Primary UVic research site with historical data from 1858",
                "expected_extent": 150.0,
                "primary_species": "Nereocystis luetkeana",
                "risk_level": "MEDIUM"
            },
            "saanich_inlet": {
                "name": "Saanich Inlet (Multi-species Site)",
                "coordinates": (48.5830, -123.5000),
                "description": "Multi-species kelp forest with seasonal monitoring",
                "expected_extent": 85.0,
                "primary_species": "Macrocystis pyrifera",
                "risk_level": "LOW"
            },
            "monterey_bay": {
                "name": "Monterey Bay (Comparative Site)",
                "coordinates": (36.8000, -121.9000),
                "description": "California comparison site with extensive historical records",
                "expected_extent": 220.0,
                "primary_species": "Macrocystis pyrifera",
                "risk_level": "HIGH"
            },
            "juan_de_fuca": {
                "name": "Juan de Fuca Strait",
                "coordinates": (48.3000, -124.0000),
                "description": "Coastal kelp forests with submerged populations",
                "expected_extent": 95.0,
                "primary_species": "Nereocystis luetkeana",
                "risk_level": "MEDIUM"
            }
        }
        
        print(f"🚀 Analytics Framework Demo initialized with {len(self.test_sites)} test sites")
    
    def run_basic_demo(self):
        """Demonstrate basic analytics framework functionality."""
        print("\n" + "="*60)
        print("🔬 BASIC ANALYTICS FRAMEWORK DEMONSTRATION")
        print("="*60)
        
        # Select test site
        site_info = self.test_sites["broughton_archipelago"]
        print(f"\n📍 Test Site: {site_info['name']}")
        print(f"   Location: {site_info['coordinates']}")
        print(f"   Description: {site_info['description']}")
        
        # Create comprehensive analysis request
        print("\n📋 Creating comprehensive analysis request...")
        request = create_analysis_request(
            analysis_types=["validation", "temporal", "species", "historical"],
            latitude=site_info["coordinates"][0],
            longitude=site_info["coordinates"][1],
            start_date=(datetime.now() - timedelta(days=365)).isoformat(),
            end_date=datetime.now().isoformat(),
            include_confidence=True,
            include_uncertainty=True,
            stakeholder_type="scientific"
        )
        
        print(f"   ✅ Analysis types: {[t.value for t in request.analysis_types]}")
        print(f"   ✅ Time range: {request.time_range[0].strftime('%Y-%m-%d')} to {request.time_range[1].strftime('%Y-%m-%d')}")
        print(f"   ✅ Quality threshold: {request.quality_threshold}")
        
        # Execute analysis
        print("\n⚡ Executing comprehensive analysis...")
        start_time = time.time()
        
        result = self.framework.execute_analysis(request)
        
        execution_time = time.time() - start_time
        print(f"   ✅ Analysis completed in {execution_time:.2f} seconds")
        print(f"   ✅ Framework execution time: {result.execution_time:.2f} seconds")
        
        # Display results summary
        print("\n📊 ANALYSIS RESULTS SUMMARY")
        print("-" * 40)
        
        summary = result.get_summary()
        print(f"Analysis Types: {', '.join(summary['analysis_types'])}")
        print(f"Site Location: {summary['site_location']}")
        print(f"Overall Confidence: {summary['overall_confidence']:.1%}")
        print(f"Data Quality Score: {summary['data_quality_score']:.1%}")
        
        print("\n🔍 Key Findings:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"   {i}. {finding}")
        
        print("\n💡 Top Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Display detailed results by analysis type
        print("\n📈 DETAILED RESULTS BY ANALYSIS TYPE")
        print("-" * 50)
        
        for analysis_type, result_data in result.results.items():
            if isinstance(result_data, dict) and not result_data.get("error"):
                print(f"\n{analysis_type.upper()} Analysis:")
                
                for key, value in result_data.items():
                    if isinstance(value, (int, float)):
                        if key.endswith("_confidence") or key.endswith("_accuracy"):
                            print(f"   • {key}: {value:.1%}")
                        elif "extent" in key or "estimate" in key:
                            print(f"   • {key}: {value:.1f} hectares")
                        elif "rate" in key:
                            print(f"   • {key}: {value:.2f} ha/year")
                        else:
                            print(f"   • {key}: {value}")
                    else:
                        print(f"   • {key}: {value}")
        
        # Display confidence and uncertainty
        print("\n🎯 CONFIDENCE & UNCERTAINTY ASSESSMENT")
        print("-" * 45)
        
        print("Confidence Scores:")
        for analysis_type, confidence in result.confidence_scores.items():
            print(f"   • {analysis_type}: {confidence:.1%}")
        
        print("\nUncertainty Estimates (95% CI):")
        for analysis_type, (lower, upper) in result.uncertainty_estimates.items():
            print(f"   • {analysis_type}: [{lower:.1f}, {upper:.1f}]")
        
        # Display performance metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 25)
        
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                print(f"   • {metric}: {value:.3f}")
            else:
                print(f"   • {metric}: {value}")
        
        return result
    
    def run_stakeholder_demo(self):
        """Demonstrate stakeholder-specific reporting."""
        print("\n" + "="*60)
        print("👥 STAKEHOLDER REPORTING DEMONSTRATION")
        print("="*60)
        
        # Get analysis result from basic demo
        print("\n📊 Generating analysis result for stakeholder reports...")
        result = self.run_basic_analysis_for_stakeholder_demo()
        
        # Generate reports for different stakeholder types
        stakeholder_types = ["first_nations", "scientific", "management"]
        
        for stakeholder_type in stakeholder_types:
            print(f"\n{'='*20} {stakeholder_type.upper().replace('_', ' ')} REPORT {'='*20}")
            
            try:
                # Generate stakeholder-specific report
                report = create_stakeholder_report(
                    stakeholder_type=stakeholder_type,
                    analysis_result=result,
                    format_type=ReportFormat.MARKDOWN
                )
                
                print(f"✅ Report generated for {stakeholder_type} stakeholders")
                print(f"📅 Generated: {report['generated_date']}")
                print(f"📍 Site: {report['site_location']}")
                print(f"📊 Format: {report['report_format']}")
                
                # Display key messages
                print(f"\n🔑 Key Messages for {stakeholder_type.title()} Stakeholders:")
                for i, message in enumerate(report['key_messages'], 1):
                    print(f"   {i}. {message}")
                
                # Display content sections
                print(f"\n📋 Report Sections:")
                content_sections = list(report['content'].keys())
                for section in content_sections:
                    print(f"   • {section.replace('_', ' ').title()}")
                
                # Show sample content from first section
                if content_sections:
                    first_section = content_sections[0]
                    sample_content = report['content'][first_section]
                    
                    print(f"\n📄 Sample Content - {first_section.replace('_', ' ').title()}:")
                    # Show first 200 characters
                    if isinstance(sample_content, str):
                        preview = sample_content[:200] + "..." if len(sample_content) > 200 else sample_content
                        print(f"   {preview}")
                    
            except Exception as e:
                print(f"❌ Error generating {stakeholder_type} report: {e}")
        
        print(f"\n✅ Stakeholder reporting demonstration completed")
    
    def run_performance_demo(self):
        """Demonstrate performance monitoring capabilities."""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE MONITORING DEMONSTRATION")
        print("="*60)
        
        # Initialize performance metrics
        performance_metrics = PerformanceMetrics()
        
        print("\n📊 Simulating performance data collection...")
        
        # Simulate performance data over time
        analysis_types = ["validation", "temporal", "species", "historical", "deep_learning"]
        
        for day in range(30):
            for analysis_type in analysis_types:
                # Simulate realistic performance variations
                base_time = 20.0
                time_variation = np.random.normal(0, 5)
                processing_time = max(10.0, base_time + time_variation)
                
                base_accuracy = 0.85
                accuracy_variation = np.random.normal(0, 0.05)
                accuracy = max(0.6, min(1.0, base_accuracy + accuracy_variation))
                
                base_quality = 0.80
                quality_variation = np.random.normal(0, 0.03)
                data_quality = max(0.5, min(1.0, base_quality + quality_variation))
                
                # Record performance
                timestamp = datetime.now() - timedelta(days=30-day)
                performance_metrics.record_performance(
                    analysis_type=analysis_type,
                    processing_time=processing_time,
                    accuracy=accuracy,
                    data_quality=data_quality,
                    timestamp=timestamp
                )
        
        print(f"   ✅ Recorded {len(performance_metrics.performance_history)} performance measurements")
        
        # Get performance summary
        print("\n📈 Performance Summary (Last 30 Days):")
        summary = performance_metrics.get_performance_summary(days=30)
        
        if "error" not in summary:
            print(f"   • Total Analyses: {summary['total_analyses']}")
            print(f"   • Period: {summary['period_days']} days")
            
            metrics = summary['performance_metrics']
            print(f"\n⏱️  Processing Performance:")
            print(f"   • Mean processing time: {metrics['mean_processing_time']:.1f} seconds")
            print(f"   • 95th percentile time: {metrics['processing_time_95th']:.1f} seconds")
            
            print(f"\n🎯 Quality Metrics:")
            print(f"   • Mean accuracy: {metrics['mean_accuracy']:.1%}")
            print(f"   • Mean data quality: {metrics['mean_data_quality']:.1%}")
            print(f"   • 5th percentile accuracy: {metrics['accuracy_5th']:.1%}")
            
            print(f"\n✅ Target Compliance:")
            compliance = summary['target_compliance']
            for target, rate in compliance.items():
                status = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.6 else "❌"
                print(f"   • {target.replace('_', ' ').title()}: {rate:.1%} {status}")
            
            print(f"\n🏥 Overall System Health: {summary['overall_system_health']:.1%}")
        
        # Demonstrate metric calculator
        print("\n🧮 Metric Calculator Demonstration:")
        
        # Calculate composite score
        composite_score = self.metric_calculator.calculate_composite_score(
            accuracy=0.89,
            precision=0.85,
            recall=0.87,
            confidence=0.84,
            data_quality=0.82
        )
        print(f"   • Composite Performance Score: {composite_score:.3f}")
        
        # Calculate detection metrics
        detection_metrics = self.metric_calculator.calculate_detection_metrics(
            true_positives=45,
            false_positives=8,
            false_negatives=5,
            true_negatives=42
        )
        print(f"   • Detection Accuracy: {detection_metrics['accuracy']:.1%}")
        print(f"   • Precision: {detection_metrics['precision']:.1%}")
        print(f"   • Recall: {detection_metrics['recall']:.1%}")
        print(f"   • F1 Score: {detection_metrics['f1_score']:.1%}")
        
        # Get system health
        print("\n🏥 System Health Assessment:")
        health = self.framework.get_system_health()
        print(f"   • Status: {health['status'].upper()}")
        print(f"   • Health Score: {health['health_score']:.1%}")
        print(f"   • Last Updated: {health['last_updated']}")
        
        if health.get('recommendations'):
            print(f"\n💡 System Recommendations:")
            for i, rec in enumerate(health['recommendations'], 1):
                print(f"   {i}. {rec}")
    
    def run_integration_demo(self):
        """Demonstrate cross-analysis integration capabilities."""
        print("\n" + "="*60)
        print("🔗 CROSS-ANALYSIS INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Demonstrate trend analysis
        print("\n📈 Temporal Trend Analysis:")
        
        # Create sample temporal data with trends
        base_date = datetime(2023, 1, 1)
        
        # Decreasing trend with seasonal variation
        temporal_data = {
            base_date + timedelta(days=i): 120 - i * 0.5 + 10 * np.sin(i / 30 * np.pi) + np.random.normal(0, 2)
            for i in range(120)
        }
        
        trend_results = self.trend_analyzer.analyze_kelp_trends(temporal_data)
        
        if "linear_trend" in trend_results:
            linear = trend_results["linear_trend"]
            print(f"   • Trend Direction: {linear['direction']}")
            print(f"   • Slope: {linear['slope']:.3f} ha/day")
            print(f"   • R-squared: {linear['r_squared']:.3f}")
        
        if "polynomial_trend" in trend_results:
            poly = trend_results["polynomial_trend"]
            print(f"   • Polynomial R-squared: {poly['r_squared']:.3f}")
            print(f"   • Curvature: {poly['curvature']}")
        
        if "change_points" in trend_results:
            change_points = trend_results["change_points"]
            print(f"   • Change Points Detected: {len(change_points)}")
            if change_points:
                print(f"   • First Change Point: Day {change_points[0]}")
        
        if "risk_assessment" in trend_results:
            risk = trend_results["risk_assessment"]
            print(f"\n⚠️  Risk Assessment:")
            print(f"   • Risk Level: {risk['risk_level']}")
            print(f"   • Risk Score: {risk['risk_score']:.2f}")
            print(f"   • Risk Factors: {', '.join(risk['risk_factors'])}")
            
            print(f"\n💡 Risk-based Recommendations:")
            for i, rec in enumerate(risk['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
        
        # Demonstrate multi-site comparison
        print(f"\n🌐 Multi-Site Comparative Analysis:")
        
        comparison_results = {}
        for site_id, site_info in list(self.test_sites.items())[:3]:
            print(f"\n   📍 {site_info['name']}:")
            
            # Quick analysis for each site
            result_summary = quick_analysis(
                latitude=site_info["coordinates"][0],
                longitude=site_info["coordinates"][1],
                analysis_type="validation"
            )
            
            comparison_results[site_id] = result_summary
            print(f"      • Overall Confidence: {result_summary['overall_confidence']:.1%}")
            print(f"      • Data Quality: {result_summary['data_quality_score']:.1%}")
            print(f"      • Analysis Types: {len(result_summary['analysis_types'])}")
        
        # Compare results across sites
        print(f"\n📊 Site Comparison Summary:")
        confidences = [r['overall_confidence'] for r in comparison_results.values()]
        qualities = [r['data_quality_score'] for r in comparison_results.values()]
        
        print(f"   • Mean Confidence: {np.mean(confidences):.1%}")
        print(f"   • Mean Quality: {np.mean(qualities):.1%}")
        print(f"   • Confidence Range: {np.min(confidences):.1%} - {np.max(confidences):.1%}")
        print(f"   • Quality Range: {np.min(qualities):.1%} - {np.max(qualities):.1%}")
        
        # Integration quality assessment
        confidence_cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 0
        quality_cv = np.std(qualities) / np.mean(qualities) if np.mean(qualities) > 0 else 0
        
        print(f"\n🔍 Integration Quality Assessment:")
        print(f"   • Confidence Consistency (CV): {confidence_cv:.1%}")
        print(f"   • Quality Consistency (CV): {quality_cv:.1%}")
        
        if confidence_cv < 0.1 and quality_cv < 0.1:
            print(f"   ✅ High consistency across sites")
        elif confidence_cv < 0.2 and quality_cv < 0.2:
            print(f"   ⚠️  Moderate consistency across sites")
        else:
            print(f"   ❌ Low consistency - investigate site-specific factors")
    
    def run_interactive_demo(self):
        """Run interactive demonstration mode."""
        print("\n" + "="*60)
        print("🎮 INTERACTIVE ANALYTICS EXPLORATION")
        print("="*60)
        
        while True:
            print(f"\n🔍 Available Commands:")
            print(f"   1. Analyze site")
            print(f"   2. Generate stakeholder report")
            print(f"   3. Compare sites")
            print(f"   4. System health check")
            print(f"   5. Custom analysis")
            print(f"   q. Quit interactive mode")
            
            choice = input(f"\nEnter your choice (1-5 or q): ").strip().lower()
            
            if choice == 'q':
                print(f"👋 Exiting interactive mode")
                break
            elif choice == '1':
                self._interactive_analyze_site()
            elif choice == '2':
                self._interactive_stakeholder_report()
            elif choice == '3':
                self._interactive_compare_sites()
            elif choice == '4':
                self._interactive_system_health()
            elif choice == '5':
                self._interactive_custom_analysis()
            else:
                print(f"❌ Invalid choice. Please try again.")
    
    def _interactive_analyze_site(self):
        """Interactive site analysis."""
        print(f"\n📍 Available Test Sites:")
        sites = list(self.test_sites.items())
        for i, (site_id, site_info) in enumerate(sites, 1):
            print(f"   {i}. {site_info['name']}")
        
        try:
            site_choice = int(input(f"Select site (1-{len(sites)}): ")) - 1
            if 0 <= site_choice < len(sites):
                site_id, site_info = sites[site_choice]
                
                print(f"\n🔬 Analyzing {site_info['name']}...")
                result_summary = quick_analysis(
                    latitude=site_info["coordinates"][0],
                    longitude=site_info["coordinates"][1],
                    analysis_type="comprehensive"
                )
                
                print(f"✅ Analysis complete!")
                print(f"   • Site: {site_info['name']}")
                print(f"   • Confidence: {result_summary['overall_confidence']:.1%}")
                print(f"   • Quality: {result_summary['data_quality_score']:.1%}")
                
                print(f"\n💡 Key Findings:")
                for finding in result_summary['key_findings']:
                    print(f"   • {finding}")
            else:
                print(f"❌ Invalid site selection")
        except ValueError:
            print(f"❌ Please enter a valid number")
    
    def _interactive_stakeholder_report(self):
        """Interactive stakeholder report generation."""
        stakeholder_types = ["first_nations", "scientific", "management"]
        
        print(f"\n👥 Available Stakeholder Types:")
        for i, stype in enumerate(stakeholder_types, 1):
            print(f"   {i}. {stype.replace('_', ' ').title()}")
        
        try:
            type_choice = int(input(f"Select stakeholder type (1-{len(stakeholder_types)}): ")) - 1
            if 0 <= type_choice < len(stakeholder_types):
                stakeholder_type = stakeholder_types[type_choice]
                
                print(f"\n📊 Generating {stakeholder_type.replace('_', ' ').title()} report...")
                
                # Use first test site for demo
                site_info = list(self.test_sites.values())[0]
                result = self.run_basic_analysis_for_stakeholder_demo()
                
                report = create_stakeholder_report(
                    stakeholder_type=stakeholder_type,
                    analysis_result=result,
                    format_type=ReportFormat.MARKDOWN
                )
                
                print(f"✅ Report generated!")
                print(f"   • Type: {stakeholder_type.replace('_', ' ').title()}")
                print(f"   • Sections: {len(report['content'])}")
                print(f"   • Key Messages: {len(report['key_messages'])}")
                
                print(f"\n🔑 Key Messages:")
                for msg in report['key_messages'][:3]:
                    print(f"   • {msg}")
            else:
                print(f"❌ Invalid stakeholder type selection")
        except ValueError:
            print(f"❌ Please enter a valid number")
    
    def _interactive_compare_sites(self):
        """Interactive site comparison."""
        print(f"\n🌐 Comparing all test sites...")
        
        for site_id, site_info in self.test_sites.items():
            print(f"\n📍 {site_info['name']}:")
            
            result_summary = quick_analysis(
                latitude=site_info["coordinates"][0],
                longitude=site_info["coordinates"][1],
                analysis_type="validation"
            )
            
            print(f"   • Confidence: {result_summary['overall_confidence']:.1%}")
            print(f"   • Quality: {result_summary['data_quality_score']:.1%}")
            print(f"   • Expected Extent: {site_info['expected_extent']} ha")
            print(f"   • Risk Level: {site_info['risk_level']}")
    
    def _interactive_system_health(self):
        """Interactive system health check."""
        print(f"\n🏥 Checking system health...")
        
        health = self.framework.get_system_health()
        
        print(f"   • Status: {health['status'].upper()}")
        print(f"   • Health Score: {health['health_score']:.1%}")
        
        if health.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in health['recommendations']:
                print(f"   • {rec}")
    
    def _interactive_custom_analysis(self):
        """Interactive custom analysis configuration."""
        print(f"\n⚙️  Custom Analysis Configuration:")
        
        # Get coordinates
        try:
            lat = float(input("Enter latitude (-90 to 90): "))
            lon = float(input("Enter longitude (-180 to 180): "))
            
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                print(f"❌ Invalid coordinates")
                return
            
            # Get analysis types
            available_types = ["validation", "temporal", "species", "historical", "deep_learning"]
            print(f"\nAvailable analysis types: {', '.join(available_types)}")
            types_input = input("Enter analysis types (comma-separated): ")
            analysis_types = [t.strip() for t in types_input.split(",")]
            
            print(f"\n🔬 Running custom analysis...")
            result_summary = quick_analysis(
                latitude=lat,
                longitude=lon,
                analysis_type="comprehensive" if "all" in analysis_types else analysis_types[0]
            )
            
            print(f"✅ Custom analysis complete!")
            print(f"   • Location: ({lat:.4f}, {lon:.4f})")
            print(f"   • Confidence: {result_summary['overall_confidence']:.1%}")
            print(f"   • Quality: {result_summary['data_quality_score']:.1%}")
            
        except ValueError:
            print(f"❌ Please enter valid numeric coordinates")
    
    def run_basic_analysis_for_stakeholder_demo(self):
        """Run basic analysis for stakeholder demo purposes."""
        site_info = self.test_sites["broughton_archipelago"]
        
        request = create_analysis_request(
            analysis_types=["validation", "historical"],
            latitude=site_info["coordinates"][0],
            longitude=site_info["coordinates"][1],
            start_date=(datetime.now() - timedelta(days=365)).isoformat(),
            end_date=datetime.now().isoformat()
        )
        
        return self.framework.execute_analysis(request)

def main():
    """Main demonstration function."""
    print("🌊 Kelpie Carbon v1 - Advanced Analytics Framework Demo")
    print("="*60)
    
    # Determine demo mode
    mode = "basic"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    # Initialize demo
    try:
        demo = AnalyticsFrameworkDemo()
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        return
    
    # Run appropriate demo mode
    try:
        if mode == "basic":
            demo.run_basic_demo()
        elif mode == "stakeholder":
            demo.run_stakeholder_demo()
        elif mode == "performance":
            demo.run_performance_demo()
        elif mode == "integration":
            demo.run_integration_demo()
        elif mode == "interactive":
            demo.run_interactive_demo()
        else:
            print(f"❌ Unknown demo mode: {mode}")
            print(f"Available modes: basic, stakeholder, performance, integration, interactive")
            return
        
        print(f"\n✅ Analytics Framework Demo completed successfully!")
        print(f"📊 Mode: {mode}")
        print(f"⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 