"""
Data Processing & Feature Engineering
ZATCA-specific invoice data processor
HANDLES: All-paid invoice scenarios
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from config import Config

class DataProcessor:
    """Process ZATCA invoice data with new field schema"""
    
    REQUIRED_FIELDS = [
        'INVOICE_ID', 'ISSUE_DATE', 'DOCUMENTCURRENCYCODE',
        'CUSTOMER_PARTY_LEGAL_ENTITY_REGISTRATION_NAME',
        'CUSTOMER_PARTY_TAX_SCHEME_COMPANYID',
        'CUSTOMER_POSTA_ADDRESS_STREET_NAME',
        'CUSTOMER_POSTALADDRESS_CITY_NAME',
        'TAXTOTAL_TAX_AMOUNT_VALUE',
        'TAX_EXCLUSIVE_AMOUNT_VALUE',
        'TAX_INCLUSIVE_AMOUNT_VALUE',
        'DISCOUNT_AMOUNT_VALUE',
        'PAYABLE_AMOUNT_VALUE',
        'INVOICE_TYPE_CODE_VALUE',
        'INVOICE_TYPE_CODE_NAME',
        'Status'
    ]
    
    @staticmethod
    def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize ZATCA invoice data"""
        
        if df.empty:
            return df
        
        clean_df = df.copy()
        
        # 1. Convert numeric fields
        numeric_fields = [
            'PAYABLE_AMOUNT_VALUE', 'TAX_EXCLUSIVE_AMOUNT_VALUE',
            'TAX_INCLUSIVE_AMOUNT_VALUE', 'TAXTOTAL_TAX_AMOUNT_VALUE',
            'DISCOUNT_AMOUNT_VALUE'
        ]
        
        for field in numeric_fields:
            if field in clean_df.columns:
                clean_df[field] = pd.to_numeric(clean_df[field], errors='coerce').fillna(0)
        
        # 2. Parse dates
        clean_df['ISSUE_DATE'] = pd.to_datetime(clean_df['ISSUE_DATE'], errors='coerce')
        
        # 3. Payment status mapping (ZATCA-specific)
        if 'Status' in clean_df.columns:
            status_counts = clean_df['Status'].value_counts()
            print(f"📊 Status distribution:")
            for status, count in status_counts.items():
                print(f"   - {status}: {count}")
            
            # Normalize
            clean_df['Status_normalized'] = clean_df['Status'].astype(str).str.strip().str.lower()
            
            # ZATCA Status Mapping
            paid_statuses = ['cleared', 'cleared-with-warning', 'non-taxable']
            unpaid_statuses = ['pending', 'exception', 'unpaid', 'open', 'outstanding']
            
            clean_df['IS_PAID'] = clean_df['Status_normalized'].isin(paid_statuses)
            clean_df['IS_UNPAID'] = clean_df['Status_normalized'].isin(unpaid_statuses)
            
            # Special handling: "Exception" can be either - let's mark as unpaid for demo
            clean_df.loc[clean_df['Status_normalized'] == 'exception', 'IS_UNPAID'] = True
            clean_df.loc[clean_df['Status_normalized'] == 'exception', 'IS_PAID'] = False
            
        else:
            print("⚠️ No 'Status' field found")
            clean_df['IS_PAID'] = True
            clean_df['IS_UNPAID'] = False
        
        # 4. Decode invoice types
        if 'INVOICE_TYPE_CODE_VALUE' in clean_df.columns:
            clean_df['INVOICE_TYPE_LABEL'] = clean_df['INVOICE_TYPE_CODE_VALUE'].astype(str).map({
                '388': 'Invoice',
                '381': 'Credit Memo',
                '383': 'Debit Note'
            }).fillna('Invoice')
        else:
            clean_df['INVOICE_TYPE_LABEL'] = 'Invoice'
        
        # 5. Decode B2B/B2C
        if 'INVOICE_TYPE_CODE_NAME' in clean_df.columns:
            clean_df['BUSINESS_TYPE'] = clean_df['INVOICE_TYPE_CODE_NAME'].astype(str).map({
                '0100000': 'B2B',
                '0200000': 'B2C'
            }).fillna('B2B')  # Default to B2B
        else:
            clean_df['BUSINESS_TYPE'] = 'B2B'
        
        # 6. Calculate invoice age
        today = pd.Timestamp.now()
        clean_df['INVOICE_AGE_DAYS'] = (today - clean_df['ISSUE_DATE']).dt.days
        
        # 7. Extract periods
        clean_df['ISSUE_MONTH'] = clean_df['ISSUE_DATE'].dt.to_period('M').astype(str)
        clean_df['ISSUE_QUARTER'] = clean_df['ISSUE_DATE'].dt.to_period('Q').astype(str)
        clean_df['ISSUE_YEAR'] = clean_df['ISSUE_DATE'].dt.year
        
        # 8. Clean names
        clean_df['CUSTOMER_NAME'] = clean_df['CUSTOMER_PARTY_LEGAL_ENTITY_REGISTRATION_NAME'].fillna('Unknown')
        clean_df['CUSTOMER_CITY'] = clean_df['CUSTOMER_POSTALADDRESS_CITY_NAME'].fillna('Unknown')
        
        # 9. Remove invalid records
        clean_df = clean_df.dropna(subset=['PAYABLE_AMOUNT_VALUE', 'ISSUE_DATE'])
        clean_df = clean_df[clean_df['PAYABLE_AMOUNT_VALUE'] > 0]
        
        # Final counts
        paid_count = clean_df['IS_PAID'].sum()
        unpaid_count = clean_df['IS_UNPAID'].sum()
        b2b_count = (clean_df['BUSINESS_TYPE'] == 'B2B').sum()
        b2c_count = (clean_df['BUSINESS_TYPE'] == 'B2C').sum()
        
        print(f"✅ Cleaned {len(clean_df):,} invoices")
        print(f"   - Paid: {paid_count:,}")
        print(f"   - Unpaid: {unpaid_count:,}")
        print(f"   - B2B: {b2b_count:,}")
        print(f"   - B2C: {b2c_count:,}")
        
        # Status check
        if unpaid_count == 0:
            print("💡 All invoices are paid - Outstanding will be SAR 0.00")
        
        return clean_df
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features for analytics and ML models"""
        
        if df.empty:
            return df
        
        features_df = df.copy()
        
        # Customer-level aggregations
        customer_stats = features_df.groupby('CUSTOMER_NAME').agg({
            'PAYABLE_AMOUNT_VALUE': ['sum', 'mean', 'count'],
            'IS_UNPAID': 'sum',
            'ISSUE_DATE': ['min', 'max'],
            'INVOICE_ID': 'count'
        }).reset_index()
        
        customer_stats.columns = [
            'CUSTOMER_NAME', 'CUSTOMER_TOTAL_REVENUE', 'CUSTOMER_AVG_INVOICE',
            'CUSTOMER_INVOICE_COUNT', 'CUSTOMER_UNPAID_COUNT',
            'CUSTOMER_FIRST_INVOICE', 'CUSTOMER_LAST_INVOICE', 'CUSTOMER_FREQUENCY'
        ]
        
        # Recency
        today = pd.Timestamp.now()
        customer_stats['CUSTOMER_RECENCY_DAYS'] = (today - customer_stats['CUSTOMER_LAST_INVOICE']).dt.days
        
        # Lifetime
        customer_stats['CUSTOMER_LIFETIME_DAYS'] = (
            customer_stats['CUSTOMER_LAST_INVOICE'] - customer_stats['CUSTOMER_FIRST_INVOICE']
        ).dt.days + 1
        
        # Merge back
        features_df = features_df.merge(customer_stats, on='CUSTOMER_NAME', how='left')
        
        # Geographic features
        city_stats = features_df.groupby('CUSTOMER_CITY').agg({
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'INVOICE_ID': 'count'
        }).reset_index()
        city_stats.columns = ['CUSTOMER_CITY', 'CITY_TOTAL_REVENUE', 'CITY_INVOICE_COUNT']
        features_df = features_df.merge(city_stats, on='CUSTOMER_CITY', how='left')
        
        print(f"✅ Engineered features for {len(features_df):,} invoices")
        
        return features_df