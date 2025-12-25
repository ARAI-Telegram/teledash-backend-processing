"""
Temporal Analysis Implementation

Analyzes temporal patterns in Telegram messaging data.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from celery.utils.log import get_task_logger
from scipy import stats
from sklearn.ensemble import IsolationForest

logger = get_task_logger(__name__)


class TemporalAnalyzer:
    """Analyze temporal patterns in Telegram data."""

    def __init__(self):
        logger.info("Initialized Temporal Analyzer")

    def detect_posting_patterns(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze when channels typically post.

        Returns:
            Dictionary with posting patterns
        """
        logger.info(f"Analyzing posting patterns for {len(messages)} messages")

        if not messages:
            return {}

        try:
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['date'])
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day_name()
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            # Find peak hours
            hourly_counts = df['hour'].value_counts()
            peak_hours = hourly_counts.nlargest(3).index.tolist()

            # Find peak days
            daily_counts = df['day'].value_counts()
            peak_days = daily_counts.nlargest(3).index.tolist()

            # Calculate posting frequencies
            time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            hours = time_range / 3600
            days = time_range / 86400
            weeks = days / 7

            posting_frequency = {
                'hourly_avg': len(messages) / hours if hours > 0 else 0,
                'daily_avg': len(messages) / days if days > 0 else 0,
                'weekly_avg': len(messages) / weeks if weeks > 0 else 0
            }

            # Detect periodicity using autocorrelation
            hourly_series = df.groupby(df['timestamp'].dt.floor('H')).size()
            periodicity_detected = False
            period_type = None

            if len(hourly_series) > 48:  # Need at least 2 days of data
                from statsmodels.tsa.stattools import acf

                try:
                    autocorr = acf(hourly_series.values, nlags=min(168, len(hourly_series) // 2))

                    # Check for daily pattern (lag ~24)
                    if len(autocorr) > 24 and autocorr[24] > 0.3:
                        periodicity_detected = True
                        period_type = 'daily'

                    # Check for weekly pattern (lag ~168)
                    elif len(autocorr) > 168 and autocorr[168] > 0.3:
                        periodicity_detected = True
                        period_type = 'weekly'
                except:
                    logger.warning("Could not compute autocorrelation")

            result = {
                'peak_hours': peak_hours,
                'peak_days': peak_days,
                'posting_frequency': posting_frequency,
                'periodicity': {
                    'has_pattern': periodicity_detected,
                    'period': period_type
                },
                'total_messages': len(messages),
                'time_span_days': days
            }

            logger.info("Posting pattern analysis complete")
            return result

        except Exception as e:
            logger.error(f"Error analyzing posting patterns: {e}")
            return {}

    def detect_anomalies(
        self,
        messages: List[Dict[str, Any]],
        contamination: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect unusual spikes or drops in activity.

        Args:
            messages: List of message dictionaries
            contamination: Expected proportion of outliers

        Returns:
            List of anomaly events
        """
        logger.info(f"Detecting anomalies in {len(messages)} messages")

        if not messages:
            return []

        try:
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['date'])

            # Group by hour
            time_series = df.groupby(df['timestamp'].dt.floor('H')).size()

            if len(time_series) < 24:  # Need at least 24 hours of data
                logger.warning("Insufficient data for anomaly detection")
                return []

            # Prepare data for anomaly detection
            values = time_series.values.reshape(-1, 1)

            # Use Isolation Forest for anomaly detection
            model = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            predictions = model.fit_predict(values)

            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]

            anomalies = []
            for idx in anomaly_indices:
                timestamp = time_series.index[idx]
                message_count = time_series.iloc[idx]

                # Calculate how much it deviates from mean
                mean_count = time_series.mean()
                std_count = time_series.std()
                z_score = (message_count - mean_count) / std_count if std_count > 0 else 0

                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'message_count': int(message_count),
                    'mean_count': float(mean_count),
                    'z_score': float(z_score),
                    'type': 'spike' if message_count > mean_count else 'drop',
                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                })

            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []

    def forecast_activity(
        self,
        messages: List[Dict[str, Any]],
        periods: int = 7
    ) -> Dict[str, Any]:
        """
        Forecast future activity levels.

        Args:
            messages: List of message dictionaries
            periods: Number of days to forecast

        Returns:
            Dictionary with forecast data
        """
        logger.info(f"Forecasting activity for {periods} days")

        if not messages:
            return {}

        try:
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['date'])

            # Group by day
            daily_series = df.groupby(df['timestamp'].dt.floor('D')).size()

            if len(daily_series) < 14:  # Need at least 2 weeks for forecasting
                logger.warning("Insufficient data for forecasting")
                return {
                    'error': 'insufficient_data',
                    'min_days_required': 14
                }

            # Simple linear trend forecast
            x = np.arange(len(daily_series))
            y = daily_series.values

            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Generate forecast
            future_x = np.arange(len(daily_series), len(daily_series) + periods)
            forecast_values = slope * future_x + intercept

            # Generate forecast dates
            last_date = daily_series.index[-1]
            forecast_dates = [
                (last_date + timedelta(days=i+1)).isoformat()
                for i in range(periods)
            ]

            forecast = {
                'forecast_dates': forecast_dates,
                'forecast_values': [max(0, float(v)) for v in forecast_values],  # Can't have negative messages
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': float(abs(r_value)),
                'confidence': 'high' if abs(r_value) > 0.7 else 'medium' if abs(r_value) > 0.4 else 'low'
            }

            logger.info("Activity forecast complete")
            return forecast

        except Exception as e:
            logger.error(f"Error forecasting activity: {e}")
            return {'error': str(e)}

    def analyze_time_series(
        self,
        messages: List[Dict[str, Any]],
        granularity: str = 'hour'
    ) -> pd.DataFrame:
        """
        Create time series analysis of message volume.

        Args:
            messages: List of message dictionaries
            granularity: Time granularity ('hour', 'day', 'week')

        Returns:
            DataFrame with time series data
        """
        logger.info(f"Creating time series with {granularity} granularity")

        try:
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['date'])

            # Group by specified granularity
            freq_map = {
                'hour': 'H',
                'day': 'D',
                'week': 'W'
            }

            freq = freq_map.get(granularity, 'H')
            time_series = df.groupby(df['timestamp'].dt.floor(freq)).agg({
                'id': 'count',
                'chat_id': lambda x: x.nunique()
            }).rename(columns={'id': 'message_count', 'chat_id': 'unique_chats'})

            logger.info(f"Time series created with {len(time_series)} periods")
            return time_series

        except Exception as e:
            logger.error(f"Error creating time series: {e}")
            return pd.DataFrame()

    def detect_event_correlations(
        self,
        messages: List[Dict[str, Any]],
        external_events: List[Dict[str, Any]],
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Correlate message spikes with external events.

        Args:
            messages: List of message dictionaries
            external_events: List of external events with timestamps
            time_window_hours: Hours before/after event to check

        Returns:
            List of correlations
        """
        logger.info("Detecting event correlations")

        if not messages or not external_events:
            return []

        try:
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['date'])

            hourly_series = df.groupby(df['timestamp'].dt.floor('H')).size()
            mean_count = hourly_series.mean()
            std_count = hourly_series.std()

            correlations = []

            for event in external_events:
                event_time = pd.to_datetime(event['timestamp'])

                # Get messages in time window around event
                start_time = event_time - timedelta(hours=time_window_hours)
                end_time = event_time + timedelta(hours=time_window_hours)

                window_messages = df[
                    (df['timestamp'] >= start_time) &
                    (df['timestamp'] <= end_time)
                ]

                if len(window_messages) == 0:
                    continue

                # Calculate activity increase
                window_hourly = window_messages.groupby(
                    window_messages['timestamp'].dt.floor('H')
                ).size()

                max_activity = window_hourly.max()
                z_score = (max_activity - mean_count) / std_count if std_count > 0 else 0

                if z_score > 2:  # Significant increase
                    correlations.append({
                        'event': event.get('name', 'Unknown'),
                        'event_time': event_time.isoformat(),
                        'peak_activity_time': window_hourly.idxmax().isoformat(),
                        'peak_message_count': int(max_activity),
                        'baseline_count': float(mean_count),
                        'z_score': float(z_score),
                        'messages_in_window': len(window_messages)
                    })

            logger.info(f"Found {len(correlations)} event correlations")
            return correlations

        except Exception as e:
            logger.error(f"Error detecting event correlations: {e}")
            return []

    def calculate_engagement_metrics(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate engagement metrics over time.

        Args:
            messages: List of message dictionaries with views/reactions

        Returns:
            Dictionary with engagement metrics
        """
        logger.info("Calculating engagement metrics")

        try:
            df = pd.DataFrame(messages)

            if 'views' not in df.columns:
                logger.warning("No view data available")
                return {}

            df['timestamp'] = pd.to_datetime(df['date'])

            # Calculate engagement rate (views per message)
            total_views = df['views'].sum()
            avg_views_per_message = df['views'].mean()

            # Engagement over time
            daily_engagement = df.groupby(df['timestamp'].dt.floor('D')).agg({
                'views': 'sum',
                'id': 'count'
            })
            daily_engagement['engagement_rate'] = daily_engagement['views'] / daily_engagement['id']

            metrics = {
                'total_views': int(total_views),
                'avg_views_per_message': float(avg_views_per_message),
                'peak_engagement_date': daily_engagement['engagement_rate'].idxmax().isoformat(),
                'peak_engagement_rate': float(daily_engagement['engagement_rate'].max()),
                'engagement_trend': 'increasing' if daily_engagement['engagement_rate'].is_monotonic_increasing else 'decreasing'
            }

            logger.info("Engagement metrics calculated")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return {}
