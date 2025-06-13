"""Professional Report Templates for Regulatory Compliance.

This module provides comprehensive report templates for regulatory submissions,
stakeholder communications, and scientific documentation with professional
formatting and VERA compliance.

Features:
- HTML/PDF report generation
- Regulatory compliance formatting
- Multi-stakeholder template variants
- Professional styling and layout
- Mathematical transparency integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from jinja2 import BaseLoader, DictLoader, Environment, FileSystemLoader

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    logging.warning("Jinja2 not available - template rendering disabled")

try:
    import weasyprint

    HAS_WEASYPRINT = True
except (ImportError, OSError) as e:
    HAS_WEASYPRINT = False
    logging.warning(f"WeasyPrint not available - PDF generation disabled: {e}")

    # Create mock weasyprint for fallback
    class MockWeasyPrint:
        class HTML:
            def __init__(self, *args, **kwargs):
                pass

            def write_pdf(self, *args, **kwargs):
                raise RuntimeError("WeasyPrint not available")

    weasyprint = MockWeasyPrint()

logger = logging.getLogger(__name__)


@dataclass
class ReportConfiguration:
    """Configuration for professional report generation."""

    title: str
    site_name: str
    organization: str = "Kelpie Carbon v1 System"
    contact_email: str = ""
    analysis_date: str = ""
    report_type: str = "scientific"
    vera_compliance: bool = True
    include_mathematical_details: bool = True
    include_uncertainty_analysis: bool = True
    logo_path: str | None = None
    custom_css: str | None = None


class ProfessionalReportTemplate:
    """Base class for professional report templates."""

    def __init__(self, config: ReportConfiguration):
        """Initialize professional report template."""
        self.config = config
        if not config.analysis_date:
            self.config.analysis_date = datetime.now().strftime("%Y-%m-%d")

    def get_base_css(self) -> str:
        """Get base CSS styling for professional reports."""
        return """
        /* Professional Report Styling */
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }

        .header {
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        .logo {
            float: left;
            max-height: 60px;
            margin-right: 20px;
        }

        .title-section {
            overflow: hidden;
        }

        h1 {
            color: #2c5aa0;
            font-size: 2.2em;
            margin: 0;
            font-weight: 300;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
            margin: 5px 0;
        }

        .report-metadata {
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #2c5aa0;
            margin: 20px 0;
        }

        .report-metadata table {
            width: 100%;
            border-collapse: collapse;
        }

        .report-metadata td {
            padding: 5px 10px;
            border-bottom: 1px solid #e9ecef;
        }

        .report-metadata td:first-child {
            font-weight: bold;
            color: #2c5aa0;
            width: 200px;
        }

        h2 {
            color: #2c5aa0;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-top: 40px;
        }

        h3 {
            color: #444;
            margin-top: 30px;
        }

        .executive-summary {
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #d6e9ff;
            margin: 20px 0;
        }

        .key-findings {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .key-findings ul {
            margin: 0;
            padding-left: 20px;
        }

        .key-findings li {
            margin: 10px 0;
        }

        .mathematical-section {
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            margin: 20px 0;
        }

        .formula {
            background: #fff;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-size: 1.1em;
        }

        .uncertainty-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .uncertainty-table th,
        .uncertainty-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }

        .uncertainty-table th {
            background: #f8f9fa;
            font-weight: bold;
            color: #2c5aa0;
        }

        .uncertainty-table tr:nth-child(even) {
            background: #f9f9f9;
        }

        .vera-compliance {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .vera-compliance h3 {
            color: #155724;
            margin-top: 0;
        }

        .compliance-badge {
            display: inline-block;
            padding: 5px 15px;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .error-box {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .figure {
            text-align: center;
            margin: 30px 0;
        }

        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }

        .figure-caption {
            font-style: italic;
            color: #666;
            margin-top: 10px;
            font-size: 0.9em;
        }

        .footer {
            border-top: 2px solid #e9ecef;
            margin-top: 50px;
            padding-top: 20px;
            color: #666;
            font-size: 0.9em;
        }

        .signature-section {
            margin-top: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        .signature-line {
            border-bottom: 1px solid #333;
            width: 300px;
            margin: 20px 0 5px 0;
        }

        @media print {
            body {
                margin: 0;
                padding: 15px;
            }

            .no-print {
                display: none;
            }

            h1, h2 {
                page-break-after: avoid;
            }

            .mathematical-section,
            .executive-summary,
            .vera-compliance {
                page-break-inside: avoid;
            }
        }
        """

    def generate_html_report(self, data: dict[str, Any]) -> str:
        """Generate HTML report from data."""
        if not HAS_JINJA2:
            return self._generate_simple_html_report(data)

        template_content = self._get_html_template()

        # Create Jinja2 environment
        env = Environment(loader=DictLoader({"report": template_content}))
        template = env.get_template("report")

        # Prepare template data
        template_data = {
            "config": self.config,
            "data": data,
            "css": self.get_base_css() + (self.config.custom_css or ""),
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return template.render(**template_data)

    def _generate_simple_html_report(self, data: dict[str, Any]) -> str:
        """Generate simple HTML report without Jinja2."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{self.config.title}</title>
            <style>{self.get_base_css()}</style>
        </head>
        <body>
            <div class="header">
                <div class="title-section">
                    <h1>{self.config.title}</h1>
                    <div class="subtitle">{self.config.site_name} - {self.config.analysis_date}</div>
                </div>
            </div>

            <div class="report-metadata">
                <table>
                    <tr><td>Organization:</td><td>{self.config.organization}</td></tr>
                    <tr><td>Site Name:</td><td>{self.config.site_name}</td></tr>
                    <tr><td>Analysis Date:</td><td>{self.config.analysis_date}</td></tr>
                    <tr><td>Report Type:</td><td>{self.config.report_type.title()}</td></tr>
                    <tr><td>VERA Compliance:</td><td>{"✅ Yes" if self.config.vera_compliance else "❌ No"}</td></tr>
                </table>
            </div>

            <div class="executive-summary">
                <h3>Executive Summary</h3>
                <p>This report presents the results of kelp carbon monitoring analysis conducted for {self.config.site_name}.</p>
            </div>

            <h2>Analysis Results</h2>
            <p>Detailed results would be populated here based on analysis data.</p>

            <div class="footer">
                <p>Generated by {self.config.organization} on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </body>
        </html>
        """
        return html_content

    def _get_html_template(self) -> str:
        """Get Jinja2 HTML template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{ config.title }}</title>
            <style>{{ css }}</style>
        </head>
        <body>
            <!-- Header Section -->
            <div class="header">
                {% if config.logo_path %}
                <img src="{{ config.logo_path }}" alt="Logo" class="logo">
                {% endif %}
                <div class="title-section">
                    <h1>{{ config.title }}</h1>
                    <div class="subtitle">{{ config.site_name }} - {{ config.analysis_date }}</div>
                </div>
            </div>

            <!-- Report Metadata -->
            <div class="report-metadata">
                <table>
                    <tr><td>Organization:</td><td>{{ config.organization }}</td></tr>
                    <tr><td>Site Name:</td><td>{{ config.site_name }}</td></tr>
                    <tr><td>Analysis Date:</td><td>{{ config.analysis_date }}</td></tr>
                    <tr><td>Report Type:</td><td>{{ config.report_type.title() }}</td></tr>
                    <tr><td>VERA Compliance:</td><td>{% if config.vera_compliance %}<span class="compliance-badge">COMPLIANT</span>{% else %}❌ Non-Compliant{% endif %}</td></tr>
                    <tr><td>Generated:</td><td>{{ generation_timestamp }}</td></tr>
                </table>
            </div>

            <!-- Executive Summary -->
            <div class="executive-summary">
                <h3>Executive Summary</h3>
                {% if data.carbon_results %}
                <p><strong>Total Carbon Content:</strong> {{ "%.4f"|format(data.carbon_results.total_carbon) }} ± {{ "%.4f"|format(data.carbon_results.total_uncertainty) }} kg C/m²</p>
                <p><strong>Relative Uncertainty:</strong> {{ "%.1f"|format((data.carbon_results.total_uncertainty / data.carbon_results.total_carbon) * 100) }}%</p>
                {% endif %}
                <p>This report presents comprehensive kelp carbon monitoring results for {{ config.site_name }}, conducted using satellite remote sensing and advanced analytical methods.</p>
            </div>

            <!-- Key Findings -->
            {% if data.key_findings %}
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                {% for finding in data.key_findings %}
                    <li>{{ finding }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}

            <!-- Methodology Section -->
            <h2>Methodology</h2>
            <p>The analysis employs satellite remote sensing data processed through the Kelpie Carbon v1 system, utilizing advanced spectral analysis and machine learning algorithms for kelp detection and biomass estimation.</p>

            {% if config.include_mathematical_details and data.carbon_calculation %}
            <!-- Mathematical Documentation -->
            <div class="mathematical-section">
                <h3>Mathematical Framework</h3>
                <p>All calculations follow peer-reviewed methodologies with complete mathematical transparency:</p>

                {% for doc in data.carbon_calculation.formula_documentation %}
                <h4>{{ doc.name }}</h4>
                <div class="formula">{{ doc.formula_latex }}</div>
                <p><strong>Description:</strong> {{ doc.description }}</p>
                <p><strong>SKEMA Equivalence:</strong> {{ "%.1f"|format(doc.skema_equivalence * 100) }}%</p>
                {% endfor %}
            </div>
            {% endif %}

            <!-- Results Section -->
            <h2>Analysis Results</h2>

            {% if data.carbon_results %}
            <h3>Carbon Quantification Results</h3>
            <table class="uncertainty-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Uncertainty</th>
                        <th>Units</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Carbon Content</td>
                        <td>{{ "%.6f"|format(data.carbon_results.total_carbon) }}</td>
                        <td>± {{ "%.6f"|format(data.carbon_results.total_uncertainty) }}</td>
                        <td>kg C/m²</td>
                    </tr>
                    <tr>
                        <td>Relative Uncertainty</td>
                        <td>{{ "%.2f"|format((data.carbon_results.total_uncertainty / data.carbon_results.total_carbon) * 100) }}</td>
                        <td>-</td>
                        <td>%</td>
                    </tr>
                </tbody>
            </table>
            {% endif %}

            {% if data.spectral_analysis %}
            <h3>Spectral Analysis Results</h3>
            <p>Comprehensive spectral signature analysis was performed to identify and quantify kelp biomass using multiple vegetation indices.</p>
            {% endif %}

            <!-- VERA Compliance Section -->
            {% if config.vera_compliance %}
            <div class="vera-compliance">
                <h3>VERA Carbon Standard Compliance</h3>
                <p>This analysis meets the requirements of the Verified Carbon Standard (VERA) for:</p>
                <ul>
                    <li><strong>Additionality:</strong> Kelp carbon sequestration represents additional carbon storage</li>
                    <li><strong>Permanence:</strong> Long-term carbon storage in kelp biomass and sediments</li>
                    <li><strong>Measurability:</strong> Quantified using validated remote sensing methods</li>
                    <li><strong>Verification:</strong> Independent third-party verification capability</li>
                    <li><strong>Uniqueness:</strong> No double-counting with other carbon projects</li>
                </ul>
                {% if data.carbon_calculation %}
                <p><strong>Mathematical Transparency:</strong> Complete step-by-step calculations with uncertainty propagation</p>
                <p><strong>SKEMA Compatibility:</strong> {{ "%.1f"|format(data.carbon_calculation.metadata.skema_compatibility * 100) }}% equivalence with established methods</p>
                {% endif %}
            </div>
            {% endif %}

            <!-- Quality Assurance -->
            <h2>Quality Assurance and Validation</h2>
            {% if data.carbon_results %}
            {% set quality_level = 'HIGH' if (data.carbon_results.total_uncertainty / data.carbon_results.total_carbon) < 0.2 else 'MODERATE' if (data.carbon_results.total_uncertainty / data.carbon_results.total_carbon) < 0.4 else 'LOW' %}
            <p><strong>Data Quality Level:</strong> {{ quality_level }}</p>
            <p><strong>Recommendation:</strong>
            {% if quality_level == 'HIGH' %}
                Results are suitable for regulatory reporting and carbon credit applications.
            {% elif quality_level == 'MODERATE' %}
                Additional validation recommended before regulatory submission.
            {% else %}
                Extensive validation required before use in carbon credit applications.
            {% endif %}
            </p>
            {% endif %}

            <!-- Conclusions -->
            <h2>Conclusions and Recommendations</h2>
            <p>The kelp carbon monitoring analysis provides scientifically robust quantification of carbon content with comprehensive uncertainty analysis. The methodology demonstrates strong compliance with VERA carbon standards and is suitable for regulatory applications.</p>

            <!-- Recommendations -->
            <h3>Recommendations</h3>
            <ul>
                <li>Continue regular monitoring to establish temporal trends</li>
                <li>Implement field validation program for continued accuracy assurance</li>
                <li>Consider expansion to additional sites for regional assessment</li>
                <li>Maintain mathematical transparency for all future analyses</li>
            </ul>

            <!-- Signature Section -->
            <div class="signature-section">
                <h3>Certification</h3>
                <p>This report has been prepared in accordance with VERA carbon standard requirements and peer-reviewed scientific methods.</p>

                <div style="margin-top: 40px;">
                    <div class="signature-line"></div>
                    <p><strong>Authorized Signatory</strong><br>
                    {{ config.organization }}<br>
                    Date: {{ config.analysis_date }}</p>
                </div>

                {% if config.contact_email %}
                <p><strong>Contact:</strong> {{ config.contact_email }}</p>
                {% endif %}
            </div>

            <!-- Footer -->
            <div class="footer">
                <p>This report was generated by the Kelpie Carbon v1 System on {{ generation_timestamp }}.</p>
                <p>For technical questions or verification requests, please contact the generating organization.</p>
            </div>
        </body>
        </html>
        """


class RegulatoryComplianceReport(ProfessionalReportTemplate):
    """Specialized template for regulatory compliance reporting."""

    def __init__(self, config: ReportConfiguration):
        """Initialize regulatory compliance template."""
        super().__init__(config)
        self.config.vera_compliance = True
        self.config.include_mathematical_details = True
        self.config.include_uncertainty_analysis = True

    def generate_vera_compliance_section(self, data: dict[str, Any]) -> str:
        """Generate VERA-specific compliance documentation."""
        compliance_html = """
        <div class="vera-compliance">
            <h2>VERA Carbon Standard Compliance Documentation</h2>

            <h3>1. Additionality Assessment</h3>
            <p>The kelp carbon sequestration project demonstrates clear additionality through:</p>
            <ul>
                <li>Baseline scenario documentation</li>
                <li>Demonstration of additional carbon storage beyond business-as-usual</li>
                <li>Regulatory surplus verification</li>
            </ul>

            <h3>2. Permanence Documentation</h3>
            <p>Carbon storage permanence is ensured through:</p>
            <ul>
                <li>Long-term kelp forest monitoring protocols</li>
                <li>Risk assessment and mitigation strategies</li>
                <li>Buffer pool contributions for uncertainty management</li>
            </ul>

            <h3>3. Measurability and Monitoring</h3>
            <p>Quantification methodology includes:</p>
            <ul>
                <li>Satellite-based biomass estimation with validated algorithms</li>
                <li>Comprehensive uncertainty quantification</li>
                <li>Regular monitoring and verification protocols</li>
                <li>Mathematical transparency with peer-reviewed methods</li>
            </ul>

            <h3>4. Verification Requirements</h3>
            <p>Third-party verification is facilitated through:</p>
            <ul>
                <li>Complete documentation of methodologies</li>
                <li>Reproducible calculation frameworks</li>
                <li>Quality assurance and control procedures</li>
                <li>Independent data validation capabilities</li>
            </ul>
        </div>
        """
        return compliance_html


class ProfessionalReportGenerator:
    """Main generator for professional reports."""

    def __init__(self):
        """Initialize professional report generator."""
        self.templates = {
            "scientific": ProfessionalReportTemplate,
            "regulatory": RegulatoryComplianceReport,
        }

    def generate_html_report(
        self,
        data: dict[str, Any],
        config: ReportConfiguration,
        output_path: str | None = None,
    ) -> str:
        """Generate HTML report.

        Args:
            data: Analysis data to include in report
            config: Report configuration
            output_path: Optional path to save HTML file

        Returns:
            HTML content or file path

        """
        template_class = self.templates.get(
            config.report_type, ProfessionalReportTemplate
        )
        template = template_class(config)

        html_content = template.generate_html_report(data)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"HTML report saved to {output_path}")
            return output_path

        return html_content

    def generate_pdf_report(
        self, data: dict[str, Any], config: ReportConfiguration, output_path: str
    ) -> str:
        """Generate PDF report from HTML.

        Args:
            data: Analysis data to include in report
            config: Report configuration
            output_path: Path to save PDF file

        Returns:
            Path to generated PDF file

        """
        if not HAS_WEASYPRINT:
            raise ImportError("WeasyPrint required for PDF generation")

        # Generate HTML content
        html_content = self.generate_html_report(data, config)

        # Create PDF
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            html_doc = weasyprint.HTML(string=html_content)
            html_doc.write_pdf(output_path)
            logger.info(f"PDF report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise

    def generate_stakeholder_reports(
        self, data: dict[str, Any], base_config: ReportConfiguration, output_dir: str
    ) -> dict[str, str]:
        """Generate reports for multiple stakeholder types.

        Args:
            data: Analysis data
            base_config: Base configuration to modify for each stakeholder
            output_dir: Directory to save reports

        Returns:
            Dictionary mapping stakeholder types to file paths

        """
        stakeholder_configs = {
            "scientific": {
                "title": f"Scientific Analysis Report - {base_config.site_name}",
                "report_type": "scientific",
                "include_mathematical_details": True,
                "include_uncertainty_analysis": True,
            },
            "regulatory": {
                "title": f"VERA Compliance Report - {base_config.site_name}",
                "report_type": "regulatory",
                "vera_compliance": True,
                "include_mathematical_details": True,
                "include_uncertainty_analysis": True,
            },
            "management": {
                "title": f"Management Summary - {base_config.site_name}",
                "report_type": "scientific",
                "include_mathematical_details": False,
                "include_uncertainty_analysis": False,
            },
        }

        generated_reports = {}

        for stakeholder_type, config_updates in stakeholder_configs.items():
            # Create modified config for this stakeholder
            stakeholder_config = ReportConfiguration(
                title=config_updates["title"],
                site_name=base_config.site_name,
                organization=base_config.organization,
                contact_email=base_config.contact_email,
                analysis_date=base_config.analysis_date,
                report_type=config_updates["report_type"],
                vera_compliance=config_updates.get(
                    "vera_compliance", base_config.vera_compliance
                ),
                include_mathematical_details=config_updates[
                    "include_mathematical_details"
                ],
                include_uncertainty_analysis=config_updates[
                    "include_uncertainty_analysis"
                ],
                logo_path=base_config.logo_path,
                custom_css=base_config.custom_css,
            )

            # Generate HTML report
            html_path = (
                Path(output_dir)
                / f"{base_config.site_name}_{stakeholder_type}_report_{base_config.analysis_date}.html"
            )
            self.generate_html_report(data, stakeholder_config, str(html_path))
            generated_reports[stakeholder_type] = str(html_path)

            # Generate PDF if possible
            if HAS_WEASYPRINT:
                try:
                    pdf_path = html_path.with_suffix(".pdf")
                    self.generate_pdf_report(data, stakeholder_config, str(pdf_path))
                    generated_reports[f"{stakeholder_type}_pdf"] = str(pdf_path)
                except Exception as e:
                    logger.warning(f"PDF generation failed for {stakeholder_type}: {e}")

        return generated_reports


def create_professional_report_generator() -> ProfessionalReportGenerator:
    """Create professional report generator."""
    return ProfessionalReportGenerator()
