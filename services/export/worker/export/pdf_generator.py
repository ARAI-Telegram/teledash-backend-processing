"""
PDF export generation logic using weasyprint.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import os

logger = logging.getLogger(__name__)


def generate_pdf_export(
    data: Dict[str, Any],
    analysis_types: List[str],
    filters: Dict[str, Any],
    include_visualizations: bool = True,
    template_name: Optional[str] = None
) -> bytes:
    """
    Generate PDF report from analysis data.

    Args:
        data: Analysis data dictionary containing results for each analysis type
        analysis_types: List of analysis types included
        filters: Filters applied to the data
        include_visualizations: Whether to include charts
        template_name: Optional custom template name

    Returns:
        PDF data as bytes
    """
    logger.info(f"Generating PDF export for analysis types: {analysis_types}")

    try:
        # TODO: Import weasyprint (will be installed via requirements)
        # from weasyprint import HTML, CSS

        # Load Jinja2 template
        template_dir = os.path.join(os.path.dirname(__file__), '../templates/pdf')
        env = Environment(loader=FileSystemLoader(template_dir))

        # Use custom template or default
        template_file = template_name or 'default_report.html'
        template = env.get_template(template_file)

        # Prepare context for template
        context = {
            'title': 'ARAI Text Analysis Report',
            'generated_at': datetime.utcnow().isoformat(),
            'analysis_types': analysis_types,
            'filters': filters,
            'include_visualizations': include_visualizations,
            'data': data,
            'summary': _generate_summary(data, analysis_types)
        }

        # Render HTML
        html_content = template.render(context)

        # Convert HTML to PDF using weasyprint
        # TODO: Uncomment when weasyprint is installed
        # pdf_bytes = HTML(string=html_content).write_pdf()

        # Placeholder return
        pdf_bytes = html_content.encode('utf-8')

        logger.info(f"PDF generated successfully: {len(pdf_bytes)} bytes")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        raise


def _generate_summary(data: Dict[str, Any], analysis_types: List[str]) -> Dict[str, Any]:
    """Generate summary statistics for report."""
    summary = {
        'total_items': 0,
        'by_type': {}
    }

    for analysis_type in analysis_types:
        type_data = data.get(analysis_type, {})

        if isinstance(type_data, list):
            count = len(type_data)
        elif isinstance(type_data, dict):
            count = type_data.get('total', len(type_data.get('results', [])))
        else:
            count = 0

        summary['by_type'][analysis_type] = count
        summary['total_items'] += count

    return summary
