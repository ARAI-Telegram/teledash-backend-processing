"""
Interactive HTML dashboard generation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import json
import os

logger = logging.getLogger(__name__)


def generate_html_export(
    data: Dict[str, Any],
    analysis_types: List[str],
    filters: Dict[str, Any],
    interactive: bool = True
) -> bytes:
    """
    Generate interactive HTML dashboard.

    Args:
        data: Analysis data dictionary
        analysis_types: List of analysis types included
        filters: Filters applied
        interactive: Whether to include interactive features (charts, filters)

    Returns:
        HTML data as bytes
    """
    logger.info(f"Generating HTML export for analysis types: {analysis_types}")

    try:
        # Load Jinja2 template
        template_dir = os.path.join(os.path.dirname(__file__), '../templates/html')
        env = Environment(loader=FileSystemLoader(template_dir))

        template = env.get_template('dashboard.html')

        # Prepare chart data for Plotly.js/ECharts
        chart_data = _prepare_chart_data(data, analysis_types) if interactive else {}

        # Prepare context
        context = {
            'title': 'ARAI Text Analysis Dashboard',
            'generated_at': datetime.utcnow().isoformat(),
            'analysis_types': analysis_types,
            'filters': filters,
            'interactive': interactive,
            'data': data,
            'chart_data': chart_data,
            'summary': _generate_summary_stats(data, analysis_types),
            # Embed data as JSON for client-side processing
            'data_json': json.dumps(data, default=str, ensure_ascii=False)
        }

        # Render HTML
        html_content = template.render(context)

        logger.info(f"HTML generated successfully: {len(html_content)} bytes")
        return html_content.encode('utf-8')

    except Exception as e:
        logger.error(f"Error generating HTML: {e}", exc_info=True)
        raise


def _prepare_chart_data(data: Dict[str, Any], analysis_types: List[str]) -> Dict[str, Any]:
    """Prepare data structures for interactive charts."""
    chart_data = {}

    # Sentiment timeline chart
    if 'sentiment' in analysis_types and 'sentiment' in data:
        sentiment_data = data['sentiment']
        if isinstance(sentiment_data, dict) and 'timeline' in sentiment_data:
            timeline = sentiment_data['timeline']
            chart_data['sentiment_timeline'] = {
                'type': 'line',
                'x': [point['timestamp'] for point in timeline],
                'series': [
                    {'name': 'Positive', 'data': [point['positive'] for point in timeline]},
                    {'name': 'Neutral', 'data': [point['neutral'] for point in timeline]},
                    {'name': 'Negative', 'data': [point['negative'] for point in timeline]}
                ]
            }

    # Entity network graph
    if 'entities' in analysis_types and 'network' in data.get('entities', {}):
        network = data['entities']['network']
        chart_data['entity_network'] = {
            'type': 'network',
            'nodes': network.get('nodes', []),
            'edges': network.get('edges', [])
        }

    # N-gram word cloud
    if 'ngrams' in analysis_types and 'ngrams' in data:
        ngrams_list = data['ngrams']
        if isinstance(ngrams_list, list):
            chart_data['ngram_wordcloud'] = {
                'type': 'wordcloud',
                'words': [
                    {'text': ng['text'], 'weight': ng['frequency']}
                    for ng in ngrams_list[:100]  # Limit to top 100
                ]
            }

    # Topic distribution
    if 'topics' in analysis_types and 'topics' in data:
        topics = data['topics']
        if isinstance(topics, list):
            chart_data['topic_distribution'] = {
                'type': 'bar',
                'labels': [t.get('topic_number', i) for i, t in enumerate(topics[:20])],
                'data': [t['size'] for t in topics[:20]]
            }

    return chart_data


def _generate_summary_stats(data: Dict[str, Any], analysis_types: List[str]) -> Dict[str, Any]:
    """Generate summary statistics for dashboard."""
    summary = {
        'total_analyzed': 0,
        'by_type': {}
    }

    for analysis_type in analysis_types:
        type_data = data.get(analysis_type, {})

        if isinstance(type_data, list):
            count = len(type_data)
        elif isinstance(type_data, dict):
            count = type_data.get('total', 0)
        else:
            count = 0

        summary['by_type'][analysis_type] = {
            'count': count,
            'label': analysis_type.capitalize()
        }
        summary['total_analyzed'] += count

    return summary
