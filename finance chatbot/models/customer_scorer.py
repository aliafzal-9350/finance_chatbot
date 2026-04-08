"""
Customer Value Scoring System
Weights: Revenue (45%), Frequency (25%), Recency (15%), Reliability (15%)
"""
import pandas as pd
import numpy as np

class CustomerScorer:
    """Customer value scoring optimized for invoice-centric data"""
    
    def __init__(self):
        self.scoring_weights = {
            'revenue': 0.45,      # Total revenue contribution
            'frequency': 0.25,    # Purchase frequency
            'recency': 0.15,      # How recent last purchase
            'reliability': 0.15   # Payment reliability
        }
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive customer scores"""
        
        today = pd.Timestamp.now()
        
        # Customer-level aggregations
        customer_metrics = df.groupby('CUSTOMER_NAME').agg({
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'INVOICE_ID': 'count',
            'ISSUE_DATE': ['min', 'max'],
            'IS_PAID': 'mean',  # Payment rate
            'IS_UNPAID': 'sum'   # Count of unpaid invoices
        }).reset_index()
        
        customer_metrics.columns = [
            'CUSTOMER_NAME',
            'TOTAL_REVENUE',
            'INVOICE_COUNT',
            'FIRST_INVOICE_DATE',
            'LAST_INVOICE_DATE',
            'PAYMENT_RATE',
            'UNPAID_COUNT'
        ]
        
        # Calculate recency (days since last invoice)
        customer_metrics['RECENCY_DAYS'] = (today - customer_metrics['LAST_INVOICE_DATE']).dt.days
        
        # Normalize scores to 0-100
        customer_metrics['REVENUE_SCORE'] = self._normalize(customer_metrics['TOTAL_REVENUE'], inverse=False) * 100
        customer_metrics['FREQUENCY_SCORE'] = self._normalize(customer_metrics['INVOICE_COUNT'], inverse=False) * 100
        customer_metrics['RECENCY_SCORE'] = self._normalize(customer_metrics['RECENCY_DAYS'], inverse=True) * 100
        customer_metrics['RELIABILITY_SCORE'] = customer_metrics['PAYMENT_RATE'] * 100
        
        # Weighted composite score
        customer_metrics['CUSTOMER_SCORE'] = (
            customer_metrics['REVENUE_SCORE'] * self.scoring_weights['revenue'] +
            customer_metrics['FREQUENCY_SCORE'] * self.scoring_weights['frequency'] +
            customer_metrics['RECENCY_SCORE'] * self.scoring_weights['recency'] +
            customer_metrics['RELIABILITY_SCORE'] * self.scoring_weights['reliability']
        )
        
        # Assign tiers
        customer_metrics['TIER'] = pd.cut(
            customer_metrics['CUSTOMER_SCORE'],
            bins=[0, 50, 70, 85, 100],
            labels=['D', 'C', 'B', 'A']
        )
        
        # Reliability grade
        customer_metrics['RELIABILITY_GRADE'] = pd.cut(
            customer_metrics['RELIABILITY_SCORE'],
            bins=[0, 60, 85, 100],
            labels=['Risky', 'Moderate', 'Excellent']
        )
        
        return customer_metrics.sort_values('CUSTOMER_SCORE', ascending=False)
    
    def _normalize(self, series: pd.Series, inverse: bool = False) -> pd.Series:
        """Min-max normalization to 0-1 scale"""
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(series))
        
        normalized = (series - min_val) / (max_val - min_val)
        
        if inverse:
            normalized = 1 - normalized
        
        return normalized