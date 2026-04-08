"""
Utility Functions for AI Finance Control Center
Formatting, metrics calculation, and helper functions
FIXED: Smart KPI calculation for historical data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import Config

class Formatter:
    """Data formatting utilities"""
    
    @staticmethod
    def format_currency(value: float, currency: str = Config.CURRENCY) -> str:
        """
        Format number as currency with appropriate scale
        
        Args:
            value: Numeric value to format
            currency: Currency symbol (default from config)
            
        Returns:
            Formatted currency string
        """
        if pd.isna(value):
            return f"{currency} 0"

        abs_value = abs(value)

        if abs_value >= 1_000_000_000:
            return f"{currency} {value/1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:
            return f"{currency} {value/1_000_000:.2f}M"
        elif abs_value >= 1_000:
            return f"{currency} {value/1_000:.1f}K"
        else:
            return f"{currency} {value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """
        Format decimal as percentage
        
        Args:
            value: Decimal value (0.15 = 15%)
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if pd.isna(value):
            return "0%"
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_date(date) -> str:
        """
        Format datetime as readable string
        
        Args:
            date: DateTime object or string
            
        Returns:
            Formatted date string (YYYY-MM-DD)
        """
        if pd.isna(date):
            return "N/A"
        
        if isinstance(date, str):
            try:
                date = pd.to_datetime(date)
            except:
                return date
        
        return date.strftime("%Y-%m-%d")
    
    @staticmethod
    def format_number(value: float, decimals: int = 0) -> str:
        """
        Format number with thousands separator
        
        Args:
            value: Numeric value
            decimals: Number of decimal places
            
        Returns:
            Formatted number string
        """
        if pd.isna(value):
            return "0"
        
        if decimals == 0:
            return f"{int(value):,}"
        else:
            return f"{value:,.{decimals}f}"
    
    @staticmethod
    def get_risk_color(risk_level: str) -> str:
        """
        Get color code for risk level
        
        Args:
            risk_level: Risk level (Low, Medium, High, Critical)
            
        Returns:
            Hex color code
        """
        colors = {
            'Low': '#10b981',      # Green
            'Medium': '#3b82f6',   # Blue
            'High': '#f59e0b',     # Amber
            'Critical': '#dc2626'  # Red
        }
        return colors.get(risk_level, '#64748b')
    
    @staticmethod
    def get_tier_color(tier: str) -> str:
        """
        Get color code for customer tier
        
        Args:
            tier: Customer tier (A, B, C, D)
            
        Returns:
            Hex color code
        """
        colors = {
            'A': '#10b981',    # Green
            'B': '#3b82f6',    # Blue
            'C': '#f59e0b',    # Amber
            'D': '#ef4444'     # Red
        }
        return colors.get(tier, '#64748b')
    
    @staticmethod
    def get_status_emoji(status: str) -> str:
        """
        Get emoji for invoice status
        
        Args:
            status: Invoice status
            
        Returns:
            Emoji string
        """
        emojis = {
            'Cleared': '✅',
            'Cleared-With-Warning': '⚠️',
            'pending': '🕒',
            'Non-Taxable': 'ℹ️',
            'Exception': '❌'
        }
        return emojis.get(status, '❓')


class MetricsCalculator:
    """Business metrics calculations"""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict:
        """
        Calculate executive dashboard KPIs
        FIXED: Smart calculation - uses most recent month with data
        
        Args:
            df: Processed invoice DataFrame
            
        Returns:
            Dictionary of KPI values
        """
        if df.empty:
            return {
                'mtd_revenue': 0,
                'outstanding_invoices': 0,
                'expected_cash_inflow_30d': 0,
                'total_invoices': 0,
                'paid_invoices': 0,
                'unpaid_invoices': 0,
                'avg_invoice_value': 0,
                'total_revenue': 0,
                'total_tax': 0,
                'b2b_revenue': 0,
                'b2c_revenue': 0,
                'invoice_count': 0,
                'credit_memo_count': 0,
                'debit_note_count': 0,
                'mtd_period': 'No Data'
            }
        
        today = pd.Timestamp.now()
        
        # CRITICAL FIX: Find the most recent month with data
        most_recent_date = df['ISSUE_DATE'].max()
        most_recent_month_start = most_recent_date.replace(day=1)
        
        # If data is old (more than 6 months), use the most recent month in data
        # Otherwise use current month
        if (today - most_recent_date).days > 180:
            # Data is historical - use most recent month in dataset
            mtd_start = most_recent_month_start
            mtd_period = f"{mtd_start.strftime('%B %Y')} (Historical)"
            print(f"📅 Using historical data: {mtd_start.strftime('%B %Y')}")
        else:
            # Data is recent - use current month
            mtd_start = today.replace(day=1)
            mtd_period = f"{mtd_start.strftime('%B %Y')} (Current)"
            print(f"📅 Using current month: {mtd_start.strftime('%B %Y')}")
        
        # 1. Month-to-date revenue (invoices issued in the selected month)
        mtd_revenue = df[df['ISSUE_DATE'] >= mtd_start]['PAYABLE_AMOUNT_VALUE'].sum()
        
        # Fallback: If MTD is still zero, use total revenue
        if mtd_revenue == 0:
            mtd_revenue = df['PAYABLE_AMOUNT_VALUE'].sum()
            mtd_period = "All Time"
            print(f"⚠️ No data in selected month, using all-time revenue")
        
        # 2. Outstanding invoices (unpaid balance)
        outstanding = df[df['IS_UNPAID']]['PAYABLE_AMOUNT_VALUE'].sum()
        
        # 3. Expected cash inflow (next 30 days - unpaid invoices)
        expected_inflow = outstanding
        
        # 4. Invoice counts
        total_invoices = len(df)
        paid_invoices = df['IS_PAID'].sum()
        unpaid_invoices = df['IS_UNPAID'].sum()
        
        # 5. Average invoice value
        avg_invoice_value = df['PAYABLE_AMOUNT_VALUE'].mean()
        
        # 6. Total revenue (all time)
        total_revenue = df['PAYABLE_AMOUNT_VALUE'].sum()
        
        # 7. Total tax collected
        total_tax = df['TAXTOTAL_TAX_AMOUNT_VALUE'].sum()
        
        # 8. B2B vs B2C breakdown
        b2b_revenue = df[df['BUSINESS_TYPE'] == 'B2B']['PAYABLE_AMOUNT_VALUE'].sum()
        b2c_revenue = df[df['BUSINESS_TYPE'] == 'B2C']['PAYABLE_AMOUNT_VALUE'].sum()
        
        # 9. Invoice type breakdown
        invoice_count = len(df[df['INVOICE_TYPE_LABEL'] == 'Invoice'])
        credit_memo_count = len(df[df['INVOICE_TYPE_LABEL'] == 'Credit Memo'])
        debit_note_count = len(df[df['INVOICE_TYPE_LABEL'] == 'Debit Note'])
        
        print(f"✅ KPIs calculated:")
        print(f"   - Period: {mtd_period}")
        print(f"   - MTD Revenue: SAR {mtd_revenue:,.2f}")
        print(f"   - Outstanding: SAR {outstanding:,.2f}")
        print(f"   - Total Invoices: {total_invoices:,}")
        print(f"   - Paid: {paid_invoices:,} | Unpaid: {unpaid_invoices:,}")
        
        return {
            'mtd_revenue': mtd_revenue,
            'outstanding_invoices': outstanding,
            'expected_cash_inflow_30d': expected_inflow,
            'total_invoices': total_invoices,
            'paid_invoices': paid_invoices,
            'unpaid_invoices': unpaid_invoices,
            'avg_invoice_value': avg_invoice_value,
            'total_revenue': total_revenue,
            'total_tax': total_tax,
            'b2b_revenue': b2b_revenue,
            'b2c_revenue': b2c_revenue,
            'invoice_count': invoice_count,
            'credit_memo_count': credit_memo_count,
            'debit_note_count': debit_note_count,
            'mtd_period': mtd_period
        }
    
    @staticmethod
    def calculate_revenue_run_rate(mtd_revenue: float, days_elapsed: int = None) -> float:
        """
        Calculate monthly revenue run rate
        
        Args:
            mtd_revenue: Month-to-date revenue
            days_elapsed: Days elapsed in month (optional, auto-calculated if None)
            
        Returns:
            Projected monthly revenue
        """
        if days_elapsed is None:
            today = datetime.now()
            days_elapsed = today.day
        
        days_in_month = pd.Timestamp.now().days_in_month
        
        if days_elapsed == 0:
            return mtd_revenue  # Return as-is if no days elapsed
        
        # For historical data (full month), return as-is
        if days_elapsed >= days_in_month - 2:  # If near end of month
            return mtd_revenue
        
        return (mtd_revenue / days_elapsed) * days_in_month
    
    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """
        Calculate growth rate between two periods
        
        Args:
            current: Current period value
            previous: Previous period value
            
        Returns:
            Growth rate as percentage
        """
        if previous == 0:
            return 0
        
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def calculate_trends(df: pd.DataFrame, period: str = 'M') -> pd.DataFrame:
        """
        Calculate revenue trends over time
        
        Args:
            df: Invoice DataFrame
            period: Pandas period code ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Trend DataFrame with aggregated metrics
        """
        df_trend = df.copy()
        df_trend['PERIOD'] = df_trend['ISSUE_DATE'].dt.to_period(period).astype(str)
        
        trends = df_trend.groupby('PERIOD').agg({
            'PAYABLE_AMOUNT_VALUE': ['sum', 'mean', 'count'],
            'IS_PAID': 'mean',
            'IS_UNPAID': 'sum',
            'TAXTOTAL_TAX_AMOUNT_VALUE': 'sum'
        }).reset_index()
        
        trends.columns = [
            'PERIOD',
            'TOTAL_REVENUE',
            'AVG_INVOICE',
            'INVOICE_COUNT',
            'PAYMENT_RATE',
            'UNPAID_COUNT',
            'TOTAL_TAX'
        ]
        
        return trends.sort_values('PERIOD')
    
    @staticmethod
    def calculate_customer_concentration(df: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Calculate customer revenue concentration
        
        Args:
            df: Invoice DataFrame
            top_n: Number of top customers to analyze
            
        Returns:
            Dictionary with concentration metrics
        """
        customer_revenue = df.groupby('CUSTOMER_NAME')['PAYABLE_AMOUNT_VALUE'].sum().sort_values(ascending=False)
        
        total_revenue = customer_revenue.sum()
        top_n_revenue = customer_revenue.head(top_n).sum()
        top_n_percentage = (top_n_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        return {
            'total_customers': len(customer_revenue),
            'top_n': top_n,
            'top_n_revenue': top_n_revenue,
            'top_n_percentage': top_n_percentage,
            'revenue_concentration': 'High' if top_n_percentage > 70 else 'Medium' if top_n_percentage > 40 else 'Low'
        }
    
    @staticmethod
    def calculate_geographic_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate revenue distribution by city
        
        Args:
            df: Invoice DataFrame
            
        Returns:
            DataFrame with city-level metrics
        """
        geo_dist = df.groupby('CUSTOMER_CITY').agg({
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'INVOICE_ID': 'count',
            'CUSTOMER_NAME': 'nunique'
        }).reset_index()
        
        geo_dist.columns = ['CITY', 'TOTAL_REVENUE', 'INVOICE_COUNT', 'CUSTOMER_COUNT']
        
        # Calculate percentages
        total_revenue = geo_dist['TOTAL_REVENUE'].sum()
        geo_dist['REVENUE_PERCENTAGE'] = (geo_dist['TOTAL_REVENUE'] / total_revenue * 100) if total_revenue > 0 else 0
        
        return geo_dist.sort_values('TOTAL_REVENUE', ascending=False)


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_fields: List[str]) -> Dict:
        """
        Validate DataFrame has required fields and data quality
        
        Args:
            df: DataFrame to validate
            required_fields: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'missing_fields': [],
            'empty_fields': [],
            'data_quality_issues': []
        }
        
        # Check for missing fields
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            validation['is_valid'] = False
            validation['missing_fields'] = missing
        
        # Check for empty fields
        for field in required_fields:
            if field in df.columns and df[field].isna().all():
                validation['empty_fields'].append(field)
        
        # Check data quality
        if 'PAYABLE_AMOUNT_VALUE' in df.columns:
            if (df['PAYABLE_AMOUNT_VALUE'] < 0).any():
                validation['data_quality_issues'].append('Negative invoice amounts detected')
        
        if 'ISSUE_DATE' in df.columns:
            if df['ISSUE_DATE'].isna().any():
                validation['data_quality_issues'].append('Missing invoice dates detected')
        
        return validation
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame, id_field: str = 'INVOICE_ID') -> pd.DataFrame:
        """
        Detect duplicate invoices
        
        Args:
            df: Invoice DataFrame
            id_field: Field to check for duplicates
            
        Returns:
            DataFrame of duplicate records
        """
        duplicates = df[df.duplicated(subset=[id_field], keep=False)]
        return duplicates.sort_values(id_field)
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, field: str = 'PAYABLE_AMOUNT_VALUE', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect statistical anomalies using z-score
        
        Args:
            df: Invoice DataFrame
            field: Numeric field to analyze
            threshold: Z-score threshold (default 3.0 = 99.7%)
            
        Returns:
            DataFrame of anomalous records
        """
        mean = df[field].mean()
        std = df[field].std()
        
        if std == 0:
            return pd.DataFrame()
        
        df['z_score'] = np.abs((df[field] - mean) / std)
        anomalies = df[df['z_score'] > threshold].copy()
        
        return anomalies.drop(columns=['z_score'])


class ExportHelper:
    """Data export utilities"""
    
    @staticmethod
    def export_to_excel(data: Dict[str, pd.DataFrame], filename: str):
        """
        Export multiple DataFrames to Excel with multiple sheets
        
        Args:
            data: Dictionary of {sheet_name: DataFrame}
            filename: Output filename
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    # Truncate sheet name to 31 characters (Excel limit)
                    safe_sheet_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
            print(f"✅ Exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str):
        """
        Export DataFrame to CSV
        
        Args:
            df: DataFrame to export
            filename: Output filename
        """
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ Exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    @staticmethod
    def generate_report_summary(kpis: Dict, forecast: Optional[pd.DataFrame] = None) -> str:
        """
        Generate executive summary text report
        
        Args:
            kpis: Dictionary of KPIs
            forecast: Optional forecast DataFrame
            
        Returns:
            Markdown-formatted report
        """
        today = datetime.now()
        
        summary = f"""
# Executive Financial Summary
**Report Generated:** {today.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Key Performance Indicators

### Revenue Metrics
- **Total Revenue:** {Formatter.format_currency(kpis['total_revenue'])}
- **Revenue (Period):** {Formatter.format_currency(kpis['mtd_revenue'])} - {kpis.get('mtd_period', 'N/A')}
- **Average Invoice Value:** {Formatter.format_currency(kpis['avg_invoice_value'])}

### Outstanding Receivables
- **Total Outstanding:** {Formatter.format_currency(kpis['outstanding_invoices'])}
- **Unpaid Invoices:** {kpis['unpaid_invoices']:,}
- **Expected Cash Inflow (30d):** {Formatter.format_currency(kpis['expected_cash_inflow_30d'])}

### Invoice Portfolio
- **Total Invoices:** {kpis['total_invoices']:,}
- **Paid Invoices:** {kpis['paid_invoices']:,} ({kpis['paid_invoices']/kpis['total_invoices']*100:.1f}%)
- **Unpaid Invoices:** {kpis['unpaid_invoices']:,}

### Business Type Breakdown
- **B2B Revenue:** {Formatter.format_currency(kpis['b2b_revenue'])}
- **B2C Revenue:** {Formatter.format_currency(kpis['b2c_revenue'])}

### Tax Collection
- **Total Tax Collected:** {Formatter.format_currency(kpis['total_tax'])}

---

## 📈 Revenue Forecast

"""
        
        if forecast is not None and not forecast.empty:
            total_forecast = forecast['Estimation'].sum()
            avg_daily = forecast['Estimation'].mean()
            summary += f"""
- **Forecast Period:** {len(forecast)} days
- **Total Expected Revenue:** {Formatter.format_currency(total_forecast)}
- **Average Daily Revenue:** {Formatter.format_currency(avg_daily)}
"""
        else:
            summary += "_No forecast data available_\n"
        
        summary += f"""

---

_Report generated by AI Finance Control Center_
        """
        
        return summary.strip()


class DateHelper:
    """Date and time utilities"""
    
    @staticmethod
    def get_period_range(period: str) -> tuple:
        """
        Get start and end dates for a named period
        
        Args:
            period: Period name ('current_month', 'last_month', 'current_quarter', 'current_year')
            
        Returns:
            Tuple of (start_date, end_date)
        """
        today = pd.Timestamp.now()
        
        if period == 'current_month':
            start = today.replace(day=1)
            end = today
        
        elif period == 'last_month':
            end = today.replace(day=1) - timedelta(days=1)
            start = end.replace(day=1)
        
        elif period == 'current_quarter':
            quarter_start_month = ((today.month - 1) // 3) * 3 + 1
            start = today.replace(month=quarter_start_month, day=1)
            end = today
        
        elif period == 'last_quarter':
            current_quarter_start_month = ((today.month - 1) // 3) * 3 + 1
            last_quarter_start_month = current_quarter_start_month - 3
            if last_quarter_start_month < 1:
                last_quarter_start_month += 12
                start = today.replace(year=today.year - 1, month=last_quarter_start_month, day=1)
            else:
                start = today.replace(month=last_quarter_start_month, day=1)
            end = today.replace(month=current_quarter_start_month, day=1) - timedelta(days=1)
        
        elif period == 'current_year':
            start = today.replace(month=1, day=1)
            end = today
        
        elif period == 'last_year':
            start = today.replace(year=today.year - 1, month=1, day=1)
            end = today.replace(year=today.year - 1, month=12, day=31)
        
        else:
            # Default to current month
            start = today.replace(day=1)
            end = today
        
        return start, end
    
    @staticmethod
    def get_period_label(period: str) -> str:
        """Get human-readable label for period"""
        labels = {
            'current_month': 'This Month',
            'last_month': 'Last Month',
            'current_quarter': 'This Quarter',
            'last_quarter': 'Last Quarter',
            'current_year': 'This Year',
            'last_year': 'Last Year'
        }
        return labels.get(period, period)