"""
Customer Churn Prediction
Risk based on unpaid invoices count
"""
import pandas as pd
import numpy as np

class ChurnPredictor:
    """Predict churn risk based on unpaid invoice behavior"""
    
    def predict_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn probability for each customer"""
        
        # Customer-level unpaid invoice analysis
        customer_risk = df.groupby('CUSTOMER_NAME').agg({
            'IS_UNPAID': 'sum',  # Count unpaid invoices
            'INVOICE_ID': 'count',  # Total invoices
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'ISSUE_DATE': 'max'
        }).reset_index()
        
        customer_risk.columns = [
            'CUSTOMER_NAME',
            'UNPAID_INVOICE_COUNT',
            'TOTAL_INVOICES',
            'TOTAL_REVENUE',
            'LAST_INVOICE_DATE'
        ]
        
        # Calculate unpaid ratio
        customer_risk['UNPAID_RATIO'] = (
            customer_risk['UNPAID_INVOICE_COUNT'] / customer_risk['TOTAL_INVOICES']
        )
        
        # Calculate days since last invoice
        today = pd.Timestamp.now()
        customer_risk['DAYS_SINCE_LAST_INVOICE'] = (
            today - customer_risk['LAST_INVOICE_DATE']
        ).dt.days
        
        # Churn probability formula:
        # High unpaid ratio (60%) + Long inactivity (40%)
        customer_risk['CHURN_PROBABILITY'] = (
            customer_risk['UNPAID_RATIO'] * 60 +
            np.clip(customer_risk['DAYS_SINCE_LAST_INVOICE'] / 90, 0, 1) * 40
        ).clip(0, 100)
        
        # Risk level classification
        customer_risk['CHURN_RISK_LEVEL'] = pd.cut(
            customer_risk['CHURN_PROBABILITY'],
            bins=[0, 30, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Generate signals
        customer_risk['SIGNALS'] = customer_risk.apply(self._generate_signals, axis=1)
        customer_risk['RECOMMENDED_ACTIONS'] = customer_risk.apply(self._generate_actions, axis=1)
        
        return customer_risk.sort_values('CHURN_PROBABILITY', ascending=False)
    
    def _generate_signals(self, row) -> str:
        """Generate risk signals"""
        signals = []
        
        if row['UNPAID_INVOICE_COUNT'] > 3:
            signals.append(f"{row['UNPAID_INVOICE_COUNT']} unpaid invoices")
        
        if row['DAYS_SINCE_LAST_INVOICE'] > 60:
            signals.append(f"Inactive for {row['DAYS_SINCE_LAST_INVOICE']} days")
        
        if row['UNPAID_RATIO'] > 0.5:
            signals.append(f"{row['UNPAID_RATIO']*100:.0f}% invoices unpaid")
        
        return ', '.join(signals) if signals else 'No critical signals'
    
    def _generate_actions(self, row) -> str:
        """Generate recommended actions"""
        if row['CHURN_PROBABILITY'] > 70:
            return "Immediate follow-up call; Offer payment plan"
        elif row['CHURN_PROBABILITY'] > 40:
            return "Send payment reminder; Schedule check-in"
        else:
            return "Monitor regularly"