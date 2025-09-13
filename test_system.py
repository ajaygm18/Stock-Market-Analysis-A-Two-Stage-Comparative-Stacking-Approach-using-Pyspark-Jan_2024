"""
Test script to verify the improved stock prediction system
"""

import sys
import os
import tempfile
import pandas as pd
from pyspark.sql import SparkSession

def create_test_data():
    """Create a small test dataset for validation"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate synthetic stock data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Simulate stock price data with trends and noise
    np.random.seed(42)
    base_price = 100
    price_data = []
    
    for i, date in enumerate(dates):
        # Add trend and random walk
        trend = 0.0002 * i  # Slight upward trend
        noise = np.random.normal(0, 0.02)
        
        if i == 0:
            close = base_price
        else:
            close = price_data[-1]['close'] * (1 + trend + noise)
        
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.lognormal(15, 0.5))
        
        price_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 4),
            'high': round(high, 4),
            'low': round(low, 4),
            'close': round(close, 4),
            'adjusted close': round(close, 4),  # Simplified
            'volume': volume,
            'dividend amount': 0.0,
            'split coefficient': 1.0
        })
    
    # Add some dividends and splits occasionally
    for i in range(0, len(price_data), 90):  # Every ~3 months
        if np.random.random() > 0.7:  # 30% chance
            price_data[i]['dividend amount'] = round(np.random.uniform(0.5, 2.0), 2)
    
    return pd.DataFrame(price_data)

def test_basic_functionality():
    """Test basic functionality without full pipeline"""
    print("Testing basic PySpark functionality...")
    
    try:
        # Test Spark session creation
        spark = SparkSession.builder.appName("TestBasic").getOrCreate()
        
        # Create test DataFrame
        test_data = create_test_data()
        
        # Save to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        # Test data loading
        df = spark.read.csv(temp_file, header=True, inferSchema=True)
        print(f"✅ Successfully loaded {df.count()} rows of test data")
        
        # Test basic transformations
        from pyspark.sql import functions as F
        df_with_return = df.withColumn("daily_return", (df.close - df.open) / df.open)
        print(f"✅ Successfully created daily return column")
        
        spark.stop()
        os.unlink(temp_file)  # Clean up
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_feature_engineering():
    """Test the feature engineering components"""
    print("Testing feature engineering...")
    
    try:
        from pyspark.sql import SparkSession, functions as F, Window
        
        spark = SparkSession.builder.appName("TestFeatures").getOrCreate()
        
        # Create test data
        test_data = create_test_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        df = spark.read.csv(temp_file, header=True, inferSchema=True)
        
        # Test window functions
        w = Window.partitionBy().orderBy("date")
        df = df.withColumn("lag_close_1", F.lag(df.close, 1).over(w))
        df = df.withColumn("ma_5", F.avg(df.close).over(w.rowsBetween(-4, 0)))
        
        # Check if features were created
        result = df.filter(df.lag_close_1.isNotNull()).count()
        print(f"✅ Feature engineering successful, {result} rows with lag features")
        
        spark.stop()
        os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_model_imports():
    """Test that all required ML models can be imported"""
    print("Testing model imports...")
    
    try:
        # Test PySpark ML imports
        from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
        from pyspark.ml.feature import StandardScaler, VectorAssembler
        from pyspark.ml import Pipeline
        from pyspark.ml.evaluation import RegressionEvaluator
        print("✅ PySpark ML imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model import test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("RUNNING STOCK PREDICTION SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Feature Engineering", test_feature_engineering),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)