"""
Advanced AI Financial Copilot
Intelligent NLP with intent recognition and context-aware responses
Inspired by warehouse-Copilot architecture
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class AICopilot:
    """Advanced natural language financial analytics assistant"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.query_log = []
        self.context = {}
        
        # Intent patterns (regex-based)
        self.intent_patterns = {
            'revenue_query': [
                r'(total|what|show|get).*revenue',
                r'how much.*earned',
                r'sales.*amount'
            ],
            'top_customers': [
                r'top.*customer',
                r'best.*customer',
                r'highest.*customer',
                r'largest.*customer'
            ],
            'unpaid_invoices': [
                r'unpaid',
                r'outstanding',
                r'pending.*invoice',
                r'not.*paid'
            ],
            'geographic_query': [
                r'in\s+(\w+)',  # "in Riyadh"
                r'from\s+(\w+)',
                r'city.*(\w+)'
            ],
            'business_type': [
                r'b2b',
                r'b2c',
                r'business.*business',
                r'business.*consumer'
            ],
            'tax_query': [
                r'tax',
                r'vat',
                r'collected'
            ],
            'credit_memo': [
                r'credit.*memo',
                r'refund'
            ],
            'debit_note': [
                r'debit.*note'
            ],
            'comparison': [
                r'compare',
                r'versus',
                r'vs',
                r'difference.*between'
            ],
            'trend': [
                r'trend',
                r'over.*time',
                r'monthly',
                r'quarterly'
            ]
        }
    
    def process_query(self, user_query: str) -> Dict:
        """
        Process natural language query with intent recognition
        
        Args:
            user_query: Natural language question
            
        Returns:
            Dict with results, explanation, and visualization data
        """
        # Log query
        self.query_log.append({
            'timestamp': datetime.now(),
            'query': user_query
        })
        
        # Normalize query
        query_lower = user_query.lower().strip()
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract parameters
        params = self._extract_parameters(query_lower)
        
        # Route to appropriate handler
        try:
            if intent == 'revenue_query':
                return self._handle_revenue_query(params)
            
            elif intent == 'top_customers':
                return self._handle_top_customers(params)
            
            elif intent == 'unpaid_invoices':
                return self._handle_unpaid_invoices(params)
            
            elif intent == 'geographic_query':
                return self._handle_geographic_query(params)
            
            elif intent == 'business_type':
                return self._handle_business_type_query(params)
            
            elif intent == 'tax_query':
                return self._handle_tax_query(params)
            
            elif intent == 'credit_memo':
                return self._handle_credit_memo_query(params)
            
            elif intent == 'debit_note':
                return self._handle_debit_note_query(params)
            
            elif intent == 'comparison':
                return self._handle_comparison_query(params)
            
            elif intent == 'trend':
                return self._handle_trend_query(params)
            
            else:
                return self._handle_unknown_query(user_query)
        
        except Exception as e:
            return {
                'success': False,
                'message': f"Sorry, I encountered an error: {str(e)}",
                'suggestions': self._get_sample_queries()
            }
    
    def _detect_intent(self, query: str) -> str:
        """Detect user intent from query"""
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        
        return 'unknown'
    
    def _extract_parameters(self, query: str) -> Dict:
        """Extract parameters from query"""
        
        params = {}
        
        # Extract time periods
        if 'this month' in query or 'current month' in query:
            params['period'] = 'current_month'
        elif 'last month' in query:
            params['period'] = 'last_month'
        elif 'this quarter' in query:
            params['period'] = 'current_quarter'
        elif 'this year' in query:
            params['period'] = 'current_year'
        
        # Extract numbers (for "top N" queries)
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            params['limit'] = int(numbers[0])
        else:
            params['limit'] = 10  # Default
        
        # Extract amount thresholds
        amount_match = re.search(r'above\s+(\d+(?:,\d+)*)', query)
        if amount_match:
            params['min_amount'] = float(amount_match.group(1).replace(',', ''))
        
        # Extract cities
        city_match = re.search(r'(?:in|from)\s+([A-Za-z]+)', query)
        if city_match:
            params['city'] = city_match.group(1).title()
        
        # Extract business type
        if 'b2b' in query:
            params['business_type'] = 'B2B'
        elif 'b2c' in query:
            params['business_type'] = 'B2C'
        
        return params
    
    # ========================================================================
    # QUERY HANDLERS
    # ========================================================================
    
    def _handle_revenue_query(self, params: Dict) -> Dict:
        """Handle revenue queries"""
        
        filtered_df = self._apply_period_filter(self.df, params.get('period'))
        
        if params.get('business_type'):
            filtered_df = filtered_df[filtered_df['BUSINESS_TYPE'] == params['business_type']]
        
        if params.get('city'):
            filtered_df = filtered_df[filtered_df['CUSTOMER_CITY'].str.contains(params['city'], case=False, na=False)]
        
        total_revenue = filtered_df['PAYABLE_AMOUNT_VALUE'].sum()
        invoice_count = len(filtered_df)
        
        # Breakdown by B2B/B2C
        breakdown = filtered_df.groupby('BUSINESS_TYPE')['PAYABLE_AMOUNT_VALUE'].sum()
        
        explanation = f"**Total Revenue:** SAR {total_revenue:,.2f}\n\n"
        explanation += f"**Invoices:** {invoice_count:,}\n\n"
        
        if not breakdown.empty:
            explanation += "**Breakdown:**\n"
            for btype, amount in breakdown.items():
                pct = (amount / total_revenue * 100) if total_revenue > 0 else 0
                explanation += f"- {btype}: SAR {amount:,.2f} ({pct:.1f}%)\n"
        
        return {
            'success': True,
            'query_type': 'revenue',
            'result': {
                'total_revenue': total_revenue,
                'invoice_count': invoice_count,
                'breakdown': breakdown.to_dict() if not breakdown.empty else {}
            },
            'explanation': explanation,
            'data': filtered_df[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE', 'BUSINESS_TYPE']].head(20)
        }
    
    def _handle_top_customers(self, params: Dict) -> Dict:
        """Handle top customers query"""
        
        filtered_df = self._apply_period_filter(self.df, params.get('period'))
        
        if params.get('city'):
            filtered_df = filtered_df[filtered_df['CUSTOMER_CITY'].str.contains(params['city'], case=False, na=False)]
        
        top_customers = filtered_df.groupby('CUSTOMER_NAME').agg({
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'INVOICE_ID': 'count'
        }).reset_index()
        
        top_customers.columns = ['Customer', 'Total_Revenue', 'Invoice_Count']
        top_customers = top_customers.sort_values('Total_Revenue', ascending=False).head(params.get('limit', 10))
        
        explanation = f"**Top {len(top_customers)} Customers by Revenue:**\n\n"
        for idx, row in top_customers.iterrows():
            explanation += f"{idx+1}. {row['Customer'][:50]} — SAR {row['Total_Revenue']:,.2f}\n"
        
        return {
            'success': True,
            'query_type': 'top_customers',
            'result': top_customers.to_dict('records'),
            'explanation': explanation,
            'data': top_customers
        }
    
    def _handle_unpaid_invoices(self, params: Dict) -> Dict:
        """Handle unpaid invoices query"""
        
        unpaid = self.df[self.df['IS_UNPAID']].copy()
        
        if params.get('min_amount'):
            unpaid = unpaid[unpaid['PAYABLE_AMOUNT_VALUE'] > params['min_amount']]
        
        if params.get('city'):
            unpaid = unpaid[unpaid['CUSTOMER_CITY'].str.contains(params['city'], case=False, na=False)]
        
        total_unpaid = unpaid['PAYABLE_AMOUNT_VALUE'].sum()
        count = len(unpaid)
        
        explanation = f"**Unpaid Invoices:** {count:,}\n\n"
        explanation += f"**Total Outstanding:** SAR {total_unpaid:,.2f}\n\n"
        
        if not unpaid.empty:
            explanation += "**Top Unpaid Amounts:**\n"
            top_unpaid = unpaid.nlargest(5, 'PAYABLE_AMOUNT_VALUE')
            for _, row in top_unpaid.iterrows():
                explanation += f"- {row['CUSTOMER_NAME'][:40]}: SAR {row['PAYABLE_AMOUNT_VALUE']:,.2f}\n"
        
        return {
            'success': True,
            'query_type': 'unpaid_invoices',
            'result': {'total': total_unpaid, 'count': count},
            'explanation': explanation,
            'data': unpaid[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE', 'CUSTOMER_CITY']].head(20)
        }
    
    def _handle_geographic_query(self, params: Dict) -> Dict:
        """Handle geographic/city-based queries"""
        
        city = params.get('city', 'Riyadh')
        
        city_data = self.df[self.df['CUSTOMER_CITY'].str.contains(city, case=False, na=False)]
        
        total_revenue = city_data['PAYABLE_AMOUNT_VALUE'].sum()
        invoice_count = len(city_data)
        customer_count = city_data['CUSTOMER_NAME'].nunique()
        
        explanation = f"**{city} Analysis:**\n\n"
        explanation += f"**Total Revenue:** SAR {total_revenue:,.2f}\n"
        explanation += f"**Invoices:** {invoice_count:,}\n"
        explanation += f"**Unique Customers:** {customer_count:,}\n"
        
        return {
            'success': True,
            'query_type': 'geographic',
            'result': {'city': city, 'revenue': total_revenue, 'invoices': invoice_count},
            'explanation': explanation,
            'data': city_data[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE']].head(20)
        }
    
    def _handle_business_type_query(self, params: Dict) -> Dict:
        """Handle B2B vs B2C queries"""
        
        btype = params.get('business_type', 'B2B')
        
        filtered = self.df[self.df['BUSINESS_TYPE'] == btype]
        
        total_revenue = filtered['PAYABLE_AMOUNT_VALUE'].sum()
        invoice_count = len(filtered)
        avg_invoice = filtered['PAYABLE_AMOUNT_VALUE'].mean()
        
        explanation = f"**{btype} Analysis:**\n\n"
        explanation += f"**Total Revenue:** SAR {total_revenue:,.2f}\n"
        explanation += f"**Invoices:** {invoice_count:,}\n"
        explanation += f"**Average Invoice:** SAR {avg_invoice:,.2f}\n"
        
        return {
            'success': True,
            'query_type': 'business_type',
            'result': {'business_type': btype, 'revenue': total_revenue},
            'explanation': explanation,
            'data': filtered[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE']].head(20)
        }
    
    def _handle_tax_query(self, params: Dict) -> Dict:
        """Handle tax/VAT queries"""
        
        filtered_df = self._apply_period_filter(self.df, params.get('period'))
        
        total_tax = filtered_df['TAXTOTAL_TAX_AMOUNT_VALUE'].sum()
        total_revenue = filtered_df['PAYABLE_AMOUNT_VALUE'].sum()
        avg_tax_rate = (total_tax / total_revenue * 100) if total_revenue > 0 else 0
        
        explanation = f"**Tax Analysis:**\n\n"
        explanation += f"**Total Tax Collected:** SAR {total_tax:,.2f}\n"
        explanation += f"**Total Revenue:** SAR {total_revenue:,.2f}\n"
        explanation += f"**Effective Tax Rate:** {avg_tax_rate:.2f}%\n"
        
        return {
            'success': True,
            'query_type': 'tax',
            'result': {'total_tax': total_tax, 'tax_rate': avg_tax_rate},
            'explanation': explanation,
            'data': filtered_df[['INVOICE_ID', 'CUSTOMER_NAME', 'TAXTOTAL_TAX_AMOUNT_VALUE']].head(20)
        }
    
    def _handle_credit_memo_query(self, params: Dict) -> Dict:
        """Handle credit memo queries"""
        
        credit_memos = self.df[self.df['INVOICE_TYPE_LABEL'] == 'Credit Memo']
        
        total_amount = credit_memos['PAYABLE_AMOUNT_VALUE'].sum()
        count = len(credit_memos)
        
        explanation = f"**Credit Memo Analysis:**\n\n"
        explanation += f"**Total Credit Memos:** {count:,}\n"
        explanation += f"**Total Amount:** SAR {total_amount:,.2f}\n"
        
        return {
            'success': True,
            'query_type': 'credit_memo',
            'result': {'count': count, 'total_amount': total_amount},
            'explanation': explanation,
            'data': credit_memos[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE']].head(20)
        }
    
    def _handle_debit_note_query(self, params: Dict) -> Dict:
        """Handle debit note queries"""
        
        debit_notes = self.df[self.df['INVOICE_TYPE_LABEL'] == 'Debit Note']
        
        total_amount = debit_notes['PAYABLE_AMOUNT_VALUE'].sum()
        count = len(debit_notes)
        
        explanation = f"**Debit Note Analysis:**\n\n"
        explanation += f"**Total Debit Notes:** {count:,}\n"
        explanation += f"**Total Amount:** SAR {total_amount:,.2f}\n"
        
        return {
            'success': True,
            'query_type': 'debit_note',
            'result': {'count': count, 'total_amount': total_amount},
            'explanation': explanation,
            'data': debit_notes[['INVOICE_ID', 'CUSTOMER_NAME', 'PAYABLE_AMOUNT_VALUE']].head(20)
        }
    
    def _handle_comparison_query(self, params: Dict) -> Dict:
        """Handle comparison queries (B2B vs B2C, etc.)"""
        
        b2b = self.df[self.df['BUSINESS_TYPE'] == 'B2B']
        b2c = self.df[self.df['BUSINESS_TYPE'] == 'B2C']
        
        b2b_revenue = b2b['PAYABLE_AMOUNT_VALUE'].sum()
        b2c_revenue = b2c['PAYABLE_AMOUNT_VALUE'].sum()
        
        explanation = f"**B2B vs B2C Comparison:**\n\n"
        explanation += f"**B2B Revenue:** SAR {b2b_revenue:,.2f} ({len(b2b):,} invoices)\n"
        explanation += f"**B2C Revenue:** SAR {b2c_revenue:,.2f} ({len(b2c):,} invoices)\n\n"
        
        if b2b_revenue > b2c_revenue:
            diff_pct = ((b2b_revenue - b2c_revenue) / b2c_revenue * 100) if b2c_revenue > 0 else 0
            explanation += f"**Insight:** B2B revenue is {diff_pct:.1f}% higher than B2C"
        else:
            diff_pct = ((b2c_revenue - b2b_revenue) / b2b_revenue * 100) if b2b_revenue > 0 else 0
            explanation += f"**Insight:** B2C revenue is {diff_pct:.1f}% higher than B2B"
        
        return {
            'success': True,
            'query_type': 'comparison',
            'result': {'b2b_revenue': b2b_revenue, 'b2c_revenue': b2c_revenue},
            'explanation': explanation,
            'data': pd.DataFrame({
                'Type': ['B2B', 'B2C'],
                'Revenue': [b2b_revenue, b2c_revenue],
                'Invoices': [len(b2b), len(b2c)]
            })
        }
    
    def _handle_trend_query(self, params: Dict) -> Dict:
        """Handle trend queries"""
        
        monthly_trend = self.df.groupby('ISSUE_MONTH').agg({
            'PAYABLE_AMOUNT_VALUE': 'sum',
            'INVOICE_ID': 'count'
        }).reset_index()
        
        monthly_trend.columns = ['Month', 'Revenue', 'Invoice_Count']
        
        explanation = f"**Monthly Revenue Trend:**\n\n"
        for _, row in monthly_trend.tail(6).iterrows():
            explanation += f"- {row['Month']}: SAR {row['Revenue']:,.2f} ({row['Invoice_Count']} invoices)\n"
        
        return {
            'success': True,
            'query_type': 'trend',
            'result': monthly_trend.to_dict('records'),
            'explanation': explanation,
            'data': monthly_trend
        }
    
    def _handle_unknown_query(self, query: str) -> Dict:
        """Handle unrecognized queries"""
        return {
            'success': False,
            'message': "I couldn't understand that query. Here are some things you can ask me:",
            'suggestions': self._get_sample_queries()
        }
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _apply_period_filter(self, df: pd.DataFrame, period: Optional[str]) -> pd.DataFrame:
        """Apply time period filter"""
        
        if not period:
            return df
        
        today = pd.Timestamp.now()
        
        if period == 'current_month':
            month_start = today.replace(day=1)
            return df[df['ISSUE_DATE'] >= month_start]
        
        elif period == 'last_month':
            last_month_end = today.replace(day=1) - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return df[(df['ISSUE_DATE'] >= last_month_start) & (df['ISSUE_DATE'] <= last_month_end)]
        
        elif period == 'current_quarter':
            quarter_start = today - pd.DateOffset(months=(today.month - 1) % 3)
            return df[df['ISSUE_DATE'] >= quarter_start.replace(day=1)]
        
        elif period == 'current_year':
            year_start = today.replace(month=1, day=1)
            return df[df['ISSUE_DATE'] >= year_start]
        
        return df
    
    def _get_sample_queries(self) -> List[str]:
        """Get list of sample queries"""
        return [
            "What is total revenue this month?",
            "Show me top 10 customers",
            "How many invoices are unpaid?",
            "Which customers in Riyadh have unpaid invoices above 10,000 SAR?",
            "Show me B2B invoices from last month",
            "What's the total tax collected this quarter?",
            "Compare revenue between B2B and B2C customers",
            "Show me all credit memos",
            "Total debit notes this month",
            "Revenue by city",
            "What's the monthly revenue trend?"
        ]
    
    def get_query_suggestions(self) -> List[str]:
        """Get categorized query suggestions"""
        return self._get_sample_queries()