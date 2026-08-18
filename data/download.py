import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('pandas')
install_package('numpy')
install_package('requests')

try:
    import pandas_datareader.data as web
except ImportError:
    print("Installing pandas-datareader...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-datareader"])
    import pandas_datareader.data as web

import pandas as pd
import numpy as np
import datetime

print("✅ All packages installed successfully!")
print("="*60)

class FREDDataLoader:
    """Load economic data from FRED - Simple and Reliable"""
    
    def __init__(self):
        self.data = None
        self.feature_names = []
    
    def fetch_simple_data(self):
        """Fetch essential economic data - minimal, reliable"""
        
        print("📥 Loading economic data from FRED...")
        
        # Set date range
        start = datetime.datetime(2018, 1, 1)
        end = datetime.datetime(2023, 12, 31)
        
        indicators = {
            'SP500': 'S&P 500 Stock Index',           # Target for prediction
            'DGS10': '10-Year Treasury Yield',        # Interest rates
            'UNRATE': 'Unemployment Rate',            # Employment
            'CPIAUCSL': 'Consumer Price Index',       # Inflation
            'DCOILWTICO': 'Crude Oil Price',          # Commodities
            'NASDAQCOM': 'NASDAQ Composite',          # Tech stocks
        }
        
        data_frames = []
        
        for code, description in indicators.items():
            try:
                print(f"   Loading {code}...")
                series_data = web.DataReader(code, 'fred', start, end)
                series_data.columns = [code]
                data_frames.append(series_data)
                
            except Exception as e:
                print(f"   ⚠️  Could not load {code}: {e}")
                continue
        
        # Combine all data
        if data_frames:
            self.data = pd.concat(data_frames, axis=1)
            self.data = self.data.fillna(method='ffill').fillna(method='bfill')
            
            print(f"\n✅ SUCCESS: Loaded {len(self.data)} days of data")
            print(f"   Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"   Available indicators: {list(self.data.columns)}")
        else:
            print("❌ ERROR: Could not load any data. Creating sample data...")
            self._create_sample_data()
        
        return self.data
    
    def _create_sample_data(self):
        """Create realistic economic data if FRED fails"""
        print("Creating realistic economic sample data...")
        
        dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        n_samples = len(dates)
        
        sp500_base = 2500
        sp500_trend = 0.0003
        sp500_vol = 0.015
        
        sp500_returns = np.random.normal(sp500_trend, sp500_vol, n_samples)
        sp500 = sp500_base * np.exp(np.cumsum(sp500_returns))
        
        crash_period = (dates > '2020-02-15') & (dates < '2020-03-30')
        sp500[crash_period] *= np.linspace(1, 0.7, sum(crash_period))
        
        recovery_period = (dates >= '2020-03-30') & (dates < '2020-12-31')
        sp500[recovery_period] *= np.linspace(1, 1.3, sum(recovery_period))
        
        data = {
            'SP500': sp500,
            'DGS10': np.random.uniform(1.5, 3.5, n_samples) + 0.01 * np.sin(np.arange(n_samples) / 50),
            'UNRATE': np.random.uniform(3.5, 8.5, n_samples) + 2 * (dates > '2020-02-01') & (dates < '2021-06-01'),
            'CPIAUCSL': 250 + 0.1 * np.arange(n_samples) + np.random.normal(0, 0.5, n_samples),
            'DCOILWTICO': 50 + 10 * np.sin(np.arange(n_samples) / 100) + np.random.normal(0, 5, n_samples),
            'NASDAQCOM': sp500 * 1.2 + np.random.normal(0, 50, n_samples)
        }
        
        self.data = pd.DataFrame(data, index=dates)
        print(f"✅ Created sample data: {self.data.shape}")
    
    def create_basic_features(self):
        """Create simple but effective features"""
        
        print("\n🔧 Creating features...")
        
        df = self.data.copy()
        
        for col in df.columns:
            df[f'{col}_return'] = df[col].pct_change()
        
        if 'SP500' in df.columns:
            for window in [5, 20, 50]:
                df[f'SP500_MA{window}'] = df['SP500'].rolling(window).mean()
        
        # 3. Economic ratios
        if 'SP500' in df.columns and 'DCOILWTICO' in df.columns:
            df['SP500_Oil_Ratio'] = df['SP500'] / df['DCOILWTICO']
        
        if 'DGS10' in df.columns and 'UNRATE' in df.columns:
            df['Yield_Unemployment'] = df['DGS10'] / (df['UNRATE'] + 1)
        
        # 4. Volatility
        if 'SP500' in df.columns:
            df['SP500_Vol_20'] = df['SP500_return'].rolling(20).std()
        
        # 5. Momentum
        for col in ['SP500', 'NASDAQCOM', 'DCOILWTICO']:
            if col in df.columns:
                df[f'{col}_Momentum_5'] = df[col] / df[col].shift(5) - 1
                df[f'{col}_Momentum_20'] = df[col] / df[col].shift(20) - 1
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature names (exclude target)
        self.feature_names = [col for col in df.columns if col != 'SP500' and 'return' not in col]
        
        print(f"✅ Created {len(self.feature_names)} features")
        return df
    
    def prepare_for_training(self, forecast_days=5, lookback_days=60):
        """Prepare data for deep learning model"""
        
        processed_data = self.create_basic_features()
        
        # Target: S&P 500 price
        target_col = 'SP500'
        
        # Features: everything except target and its returns
        feature_cols = [col for col in processed_data.columns 
                        if col != target_col and f'{target_col}_return' not in col]
        
        X = processed_data[feature_cols].values
        y = processed_data[target_col].values
        
        print(f"\n📊 Training Data Prepared:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Lookback window: {lookback_days} days")
        print(f"   Forecast horizon: {forecast_days} days ahead")
        print(f"\n   Available for sequences: {len(X) - lookback_days - forecast_days} samples")
        
        return X, y, feature_cols


def main():
    """Main function - simple and reliable"""
    
    print("="*60)
    print("FRED ECONOMIC DATA FOR FINANCIAL FORECASTING")
    print("="*60)
    
    print("\n1. LOADING DATA FROM FEDERAL RESERVE (FRED)")
    loader = FREDDataLoader()
    data = loader.fetch_simple_data()
    
    print(f"\n📈 Data Overview:")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    
    print(f"\n📋 First 5 rows:")
    print(data.head())
    
    # Show statistics
    print(f"\n📊 Basic Statistics:")
    print(data.describe())
    
    # Step 2: Prepare for model
    print("\n2. PREPARING DATA FOR DEEP LEARNING MODEL")
    X, y, features = loader.prepare_for_training(
        forecast_days=5,
        lookback_days=60
    )
    
    print(f"\n✅ First 3 feature names: {features[:3]}")
    print(f"✅ X shape: {X.shape}")
    print(f"✅ y shape: {y.shape}")
    
    print("\n3. VISUALIZING THE DATA")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(data.columns[:6]):
            if idx < len(axes):
                axes[idx].plot(data.index, data[col])
                axes[idx].set_title(col)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib not available for plotting")
    
    # Save data
    print("\n4. SAVING DATA")
    data.to_csv('fred_economic_data.csv')
    print("✅ Data saved to 'fred_economic_data.csv'")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use X and y for LSTM/Transformer training")
    print("2. Sequence length: 60 days")
    print("3. Forecast horizon: 5 days ahead")
    print(f"4. Total samples available: {len(X) - 60 - 5}")
    
    return X, y, features, data

if __name__ == "__main__":
    X, y, features, data = main()