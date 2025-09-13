# Workspace Setup Guide for Stock Market Prediction System

## Overview

This guide provides step-by-step instructions to set up the development workspace for the Stock Market Prediction system using PySpark and machine learning models.

## Prerequisites

- Python 3.8 or higher
- Java 8 or 11 (required for PySpark)
- Git
- At least 8GB RAM (recommended for PySpark operations)
- 10GB+ free disk space

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ajaygm18/Stock-Market-Analysis-A-Two-Stage-Comparative-Stacking-Approach-using-Pyspark-Jan_2024.git
cd Stock-Market-Analysis-A-Two-Stage-Comparative-Stacking-Approach-using-Pyspark-Jan_2024
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv stock_prediction_env

# Activate virtual environment
# On Linux/Mac:
source stock_prediction_env/bin/activate
# On Windows:
stock_prediction_env\Scripts\activate
```

### 3. Install Java (for PySpark)

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
```

#### macOS:
```bash
brew install openjdk@11
export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home
echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home' >> ~/.zshrc
```

#### Windows:
1. Download Java 11 from Oracle or OpenJDK
2. Install and set JAVA_HOME environment variable
3. Add JAVA_HOME/bin to PATH

### 4. Install Python Dependencies

Create a `requirements.txt` file:

```bash
cat > requirements.txt << EOF
pyspark==3.5.0
pandas==2.1.4
numpy==1.24.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
yfinance==0.2.28
streamlit==1.29.0
tensorflow==2.15.0
keras==2.15.0
jupyterlab==4.0.9
plotly==5.17.0
scipy==1.11.4
joblib==1.3.2
ta==0.10.2
arch==5.6.0
pyfolio==0.9.2
EOF
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 5. Environment Variables

Create a `.env` file:

```bash
cat > .env << EOF
# PySpark Configuration
PYSPARK_PYTHON=python3
PYSPARK_DRIVER_PYTHON=python3
SPARK_HOME=/path/to/spark  # Update with actual Spark installation path
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Update with actual Java path

# Data Configuration
DATA_PATH=./data_stocks.csv
MODEL_SAVE_PATH=./models/
LOGS_PATH=./logs/

# Trading Configuration
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
EOF
```

### 6. Create Project Structure

```bash
mkdir -p {models,logs,data,tests,notebooks,scripts,config}
```

## Project Structure

```
Stock-Market-Analysis-A-Two-Stage-Comparative-Stacking-Approach-using-Pyspark-Jan_2024/
├── README.md
├── MODEL_ARCHITECTURE_ANALYSIS.md
├── WORKSPACE_SETUP.md
├── requirements.txt
├── .env
├── .gitignore
├── finalStockPrediction.py
├── app.py
├── data_stocks.csv
├── tesla.csv
├── index.html
├── predict.php
├── predict1.html
├── adi.sh
├── adifoot.css
├── models/
│   ├── __init__.py
│   ├── ensemble_model.py
│   ├── base_models.py
│   └── model_utils.py
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── data_validation.py
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── trading_config.py
├── scripts/
│   ├── train_model.py
│   ├── backtest.py
│   └── deploy_model.py
├── tests/
│   ├── test_models.py
│   ├── test_data.py
│   └── test_trading.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_development.ipynb
│   └── backtesting_analysis.ipynb
└── logs/
```

## Verification Steps

### 1. Test PySpark Installation

```python
# test_spark.py
from pyspark.sql import SparkSession

def test_spark():
    try:
        spark = SparkSession.builder \
            .appName("TestSpark") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        # Create a simple DataFrame
        data = [("Alice", 34), ("Bob", 45), ("Charlie", 56)]
        columns = ["Name", "Age"]
        df = spark.createDataFrame(data, columns)
        
        df.show()
        print("✅ PySpark is working correctly!")
        
        spark.stop()
        return True
    except Exception as e:
        print(f"❌ PySpark setup failed: {e}")
        return False

if __name__ == "__main__":
    test_spark()
```

Run the test:
```bash
python test_spark.py
```

### 2. Test Data Loading

```python
# test_data.py
import pandas as pd
from pyspark.sql import SparkSession

def test_data_loading():
    try:
        # Test with pandas
        df_pandas = pd.read_csv('data_stocks.csv')
        print(f"✅ Pandas loaded {len(df_pandas)} rows")
        
        # Test with PySpark
        spark = SparkSession.builder.appName("TestData").getOrCreate()
        df_spark = spark.read.csv("data_stocks.csv", header=True, inferSchema=True)
        print(f"✅ PySpark loaded {df_spark.count()} rows")
        
        spark.stop()
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
```

### 3. Test Model Dependencies

```python
# test_models.py
def test_model_imports():
    try:
        # Test PySpark ML
        from pyspark.ml.regression import LinearRegression, RandomForestRegressor
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.ml import Pipeline
        print("✅ PySpark ML imports successful")
        
        # Test scikit-learn
        from sklearn.ensemble import RandomForestRegressor as SKRandomForest
        from sklearn.linear_model import LinearRegression as SKLinearRegression
        print("✅ Scikit-learn imports successful")
        
        # Test TensorFlow/Keras
        import tensorflow as tf
        from tensorflow import keras
        print("✅ TensorFlow/Keras imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False

if __name__ == "__main__":
    test_model_imports()
```

## Development Workflow

### 1. Start Development Environment

```bash
# Activate virtual environment
source stock_prediction_env/bin/activate

# Start Jupyter Lab for development
jupyter lab

# Or start Streamlit app
streamlit run app.py
```

### 2. Run Main Prediction Script

```bash
# Ensure data file exists
ls -la data_stocks.csv

# Run the main prediction script
python finalStockPrediction.py
```

### 3. Development Best Practices

1. **Use version control**: Commit changes frequently
2. **Write tests**: Add tests for new functionality
3. **Use logging**: Replace print statements with proper logging
4. **Configuration management**: Use config files instead of hardcoded values
5. **Documentation**: Update documentation for new features

## Troubleshooting

### Common Issues

1. **Java not found**:
   - Verify JAVA_HOME is set correctly
   - Ensure Java is in PATH

2. **PySpark memory errors**:
   - Reduce dataset size for testing
   - Increase driver memory: `--driver-memory 4g`

3. **Module not found errors**:
   - Verify virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

4. **Data loading failures**:
   - Check file paths are correct
   - Verify data file format and encoding

### Performance Optimization

1. **Spark Configuration**:
   ```python
   spark = SparkSession.builder \
       .appName("StockPrediction") \
       .config("spark.sql.adaptive.enabled", "true") \
       .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
       .config("spark.driver.memory", "4g") \
       .config("spark.driver.maxResultSize", "2g") \
       .getOrCreate()
   ```

2. **Data Partitioning**:
   - Partition data by date for time-series operations
   - Use appropriate file formats (Parquet instead of CSV)

3. **Memory Management**:
   - Use `cache()` for frequently accessed DataFrames
   - Use `persist()` with appropriate storage levels

## Next Steps

1. Review the `MODEL_ARCHITECTURE_ANALYSIS.md` for detailed architecture assessment
2. Run the verification tests to ensure everything is working
3. Start with the notebook examples in the `notebooks/` directory
4. Begin implementing the recommended improvements from the analysis

## Support

For issues with the workspace setup:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Review error logs in the `logs/` directory
4. Consult PySpark documentation for Spark-specific issues