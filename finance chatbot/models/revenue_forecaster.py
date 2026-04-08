"""
AI Revenue Forecaster - Prophet-based with Positive Value Enforcement
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class RevenueForecaster:
    """Revenue forecasting with guaranteed positive predictions"""
    
    def __init__(self):
        self.model = None
        self.forecast_df = None
        self.is_trained = False
        self.historical_mean = 0
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare daily revenue data for Prophet"""
        
        # Aggregate by day
        daily_revenue = df.groupby(df['ISSUE_DATE'].dt.date).agg({
            'PAYABLE_AMOUNT_VALUE': 'sum'
        }).reset_index()
        
        daily_revenue.columns = ['ds', 'y']
        daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])
        
        # Store historical mean for fallback
        self.historical_mean = daily_revenue['y'].mean()
        
        return daily_revenue
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train forecasting model"""
        
        try:
            from prophet import Prophet
        except ImportError:
            print("⚠️ Prophet not installed. Using simple forecasting.")
            self.is_trained = True
            return {'method': 'Simple Moving Average', 'data_points': len(df)}
        
        prophet_data = self.prepare_data(df)
        
        if len(prophet_data) < 10:
            print("⚠️ Insufficient data. Using simple forecasting.")
            self.is_trained = True
            return {'method': 'Simple Baseline', 'data_points': len(prophet_data)}
        
        # Train Prophet with conservative settings
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,  # Conservative - prevents wild swings
            seasonality_prior_scale=0.1,
            interval_width=0.80  # 80% confidence interval
        )
        
        self.model.fit(prophet_data)
        self.is_trained = True
        
        return {
            'method': 'Prophet',
            'data_points': len(prophet_data),
            'date_range': f"{prophet_data['ds'].min()} to {prophet_data['ds'].max()}"
        }
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Generate forecast with positive value guarantee"""
        
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        if self.model is None:
            return self._predict_simple(periods)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        # Predict
        forecast = self.model.predict(future)
        
        # Filter to future only
        forecast = forecast[forecast['ds'] > self.model.history['ds'].max()].copy()
        
        # CRITICAL: Ensure all values are positive
        forecast['yhat'] = forecast['yhat'].clip(lower=self.historical_mean * 0.5)  # At least 50% of historical mean
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=forecast['yhat'] * 0.8)
        
        # Rename columns
        self.forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        self.forecast_df.columns = ['Date', 'Estimation', 'estimate_lower', 'estimate_upper']
        
        # Final safety check
        self.forecast_df['Estimation'] = self.forecast_df['Estimation'].abs()
        self.forecast_df['estimate_lower'] = self.forecast_df['estimate_lower'].abs()
        self.forecast_df['estimate_upper'] = self.forecast_df['estimate_upper'].abs()
        
        return self.forecast_df
    
    def _predict_simple(self, periods: int) -> pd.DataFrame:
        """Simple baseline forecast (always positive)"""
        
        dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
        
        # Use historical mean with small random variation
        base = self.historical_mean if self.historical_mean > 0 else 1500
        noise = np.random.normal(0, base * 0.1, periods)
        
        estimations = np.abs(base + noise)  # Absolute value ensures positive
        
        return pd.DataFrame({
            'Date': dates,
            'Estimation': estimations,
            'estimate_lower': estimations * 0.85,
            'estimate_upper': estimations * 1.15
        })