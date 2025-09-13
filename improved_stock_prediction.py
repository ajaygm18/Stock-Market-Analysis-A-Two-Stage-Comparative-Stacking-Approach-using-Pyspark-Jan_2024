"""
Improved Stock Market Prediction with Two-Stage Stacking Ensemble
Addresses critical issues identified in the original implementation
"""

import logging
import sys
from typing import List, Tuple, Dict
from datetime import datetime

from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, when, isnan, isinf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StockPredictionPipeline:
    """
    Improved stock prediction pipeline with proper two-stage stacking ensemble
    """
    
    def __init__(self, app_name: str = "ImprovedStockPrediction"):
        """Initialize the prediction pipeline"""
        self.spark = None
        self.app_name = app_name
        self.models_stage1 = {}
        self.models_stage2 = {}
        self.meta_model = None
        self.feature_columns = []
        self.evaluators = {
            'rmse': RegressionEvaluator(labelCol="adjusted close", metricName="rmse"),
            'mae': RegressionEvaluator(labelCol="adjusted close", metricName="mae"),
            'r2': RegressionEvaluator(labelCol="adjusted close", metricName="r2")
        }
        
    def initialize_spark(self) -> None:
        """Initialize Spark session with optimized configurations"""
        try:
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.driver.memory", "4g") \
                .config("spark.driver.maxResultSize", "2g") \
                .getOrCreate()
            
            logger.info("Spark session initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spark session: {e}")
            raise
    
    def load_data(self, file_path: str):
        """Load and validate data"""
        try:
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            
            # Data validation
            if df.count() == 0:
                raise ValueError("Dataset is empty")
            
            required_columns = ["date", "open", "high", "low", "close", "adjusted close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Data loaded successfully: {df.count()} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def engineer_features(self, df):
        """Enhanced feature engineering with additional technical indicators"""
        try:
            # Window function for time-based operations
            w = Window.partitionBy().orderBy("date")
            
            # Basic price features
            df = df.withColumn("lag_close_1", F.lag(df.close, 1).over(w))
            df = df.withColumn("lag_close_2", F.lag(df.close, 2).over(w))
            df = df.withColumn("lag_close_3", F.lag(df.close, 3).over(w))
            
            # Returns and volatility
            df = df.withColumn("daily_return", (df.close - df.open) / df.open)
            df = df.withColumn("log_return", F.log(df.close / df.lag_close_1))
            df = df.withColumn("intra_day_volatility", (df.high - df.low) / df.close)
            df = df.withColumn("daily_volatility", F.abs(df.close - df.lag_close_1) / df.lag_close_1)
            
            # Moving averages
            df = df.withColumn("ma_5", F.avg(df.close).over(w.rowsBetween(-4, 0)))
            df = df.withColumn("ma_10", F.avg(df.close).over(w.rowsBetween(-9, 0)))
            df = df.withColumn("ma_20", F.avg(df.close).over(w.rowsBetween(-19, 0)))
            df = df.withColumn("ma_50", F.avg(df.close).over(w.rowsBetween(-49, 0)))
            
            # Price position relative to moving averages
            df = df.withColumn("price_to_ma_5", df.close / df.ma_5)
            df = df.withColumn("price_to_ma_20", df.close / df.ma_20)
            
            # Volume features
            df = df.withColumn("volume_ma_10", F.avg(df.volume).over(w.rowsBetween(-9, 0)))
            df = df.withColumn("volume_ratio", df.volume / df.volume_ma_10)
            
            # Bollinger Bands
            df = df.withColumn("price_std_20", F.stddev(df.close).over(w.rowsBetween(-19, 0)))
            df = df.withColumn("bb_upper", df.ma_20 + 2 * df.price_std_20)
            df = df.withColumn("bb_lower", df.ma_20 - 2 * df.price_std_20)
            df = df.withColumn("bb_position", (df.close - df.bb_lower) / (df.bb_upper - df.bb_lower))
            
            # RSI approximation (simplified)
            df = df.withColumn("price_change", df.close - df.lag_close_1)
            df = df.withColumn("gain", when(df.price_change > 0, df.price_change).otherwise(0))
            df = df.withColumn("loss", when(df.price_change < 0, F.abs(df.price_change)).otherwise(0))
            df = df.withColumn("avg_gain", F.avg(df.gain).over(w.rowsBetween(-13, 0)))
            df = df.withColumn("avg_loss", F.avg(df.loss).over(w.rowsBetween(-13, 0)))
            df = df.withColumn("rsi", 100 - (100 / (1 + df.avg_gain / (df.avg_loss + 0.0001))))
            
            # Corporate actions
            df = df.withColumn("is_dividend", when(col("dividend amount") > 0, 1.0).otherwise(0.0))
            df = df.withColumn("is_split", when(col("split coefficient") != 1, 1.0).otherwise(0.0))
            
            # Clean infinite and null values
            df = df.replace([float('inf'), float('-inf')], None)
            
            # Remove rows with null values
            df = df.dropna()
            
            # Define feature columns for modeling
            self.feature_columns = [
                "open", "high", "low", "close", "volume",
                "daily_return", "log_return", "intra_day_volatility", "daily_volatility",
                "ma_5", "ma_10", "ma_20", "price_to_ma_5", "price_to_ma_20",
                "volume_ratio", "bb_position", "rsi",
                "lag_close_1", "lag_close_2", "lag_close_3",
                "is_dividend", "is_split"
            ]
            
            logger.info(f"Feature engineering completed. Features: {len(self.feature_columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def prepare_data(self, df):
        """Prepare data for machine learning"""
        try:
            # Assemble features
            assembler = VectorAssembler(inputCols=self.feature_columns, outputCol="features")
            df_assembled = assembler.transform(df)
            
            # Time-based split (more appropriate for time series)
            df_sorted = df_assembled.orderBy("date")
            total_count = df_sorted.count()
            train_count = int(total_count * 0.7)
            val_count = int(total_count * 0.15)
            
            train_data = df_sorted.limit(train_count)
            temp_data = df_sorted.subtract(train_data)
            val_data = temp_data.limit(val_count)
            test_data = temp_data.subtract(val_data)
            
            logger.info(f"Data split - Train: {train_data.count()}, "
                       f"Validation: {val_data.count()}, Test: {test_data.count()}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def initialize_models(self):
        """Initialize base models and meta-model"""
        try:
            # Stage 1: Linear models with different regularization
            self.models_stage1 = {
                'linear_reg': LinearRegression(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxIter=100
                ),
                'ridge': LinearRegression(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxIter=100, 
                    regParam=0.01, 
                    elasticNetParam=0.0
                ),
                'lasso': LinearRegression(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxIter=100, 
                    regParam=0.1, 
                    elasticNetParam=1.0
                ),
                'elastic_net': LinearRegression(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxIter=100, 
                    regParam=0.1, 
                    elasticNetParam=0.5
                )
            }
            
            # Stage 2: Non-linear models
            self.models_stage2 = {
                'random_forest': RandomForestRegressor(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    numTrees=100,  # Increased from 5
                    maxDepth=10,
                    subsamplingRate=0.8
                ),
                'decision_tree': DecisionTreeRegressor(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxDepth=15  # Increased from 5
                ),
                'gradient_boosting': GBTRegressor(
                    labelCol="adjusted close", 
                    featuresCol="scaled_features", 
                    maxIter=100,
                    maxDepth=8
                )
            }
            
            # Meta-model for stacking
            self.meta_model = RandomForestRegressor(
                labelCol="adjusted close", 
                featuresCol="meta_features", 
                numTrees=50,
                maxDepth=8
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    def train_stage1_models(self, train_data, val_data):
        """Train Stage 1 models and return predictions"""
        try:
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features", 
                                  withStd=True, withMean=True)
            
            stage1_predictions = {}
            stage1_performance = {}
            
            for name, model in self.models_stage1.items():
                logger.info(f"Training Stage 1 model: {name}")
                
                # Create pipeline
                pipeline = Pipeline(stages=[scaler, model])
                
                # Train model
                model_pipeline = pipeline.fit(train_data)
                
                # Validate
                val_predictions = model_pipeline.transform(val_data)
                
                # Evaluate
                metrics = {}
                for metric_name, evaluator in self.evaluators.items():
                    metrics[metric_name] = evaluator.evaluate(val_predictions)
                
                stage1_performance[name] = metrics
                stage1_predictions[name] = model_pipeline
                
                logger.info(f"{name} - RMSE: {metrics['rmse']:.4f}, "
                           f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
            
            return stage1_predictions, stage1_performance
            
        except Exception as e:
            logger.error(f"Stage 1 training failed: {e}")
            raise
    
    def train_stage2_models(self, train_data, val_data):
        """Train Stage 2 models and return predictions"""
        try:
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features", 
                                  withStd=True, withMean=True)
            
            stage2_predictions = {}
            stage2_performance = {}
            
            for name, model in self.models_stage2.items():
                logger.info(f"Training Stage 2 model: {name}")
                
                # Create pipeline
                pipeline = Pipeline(stages=[scaler, model])
                
                # Train model
                model_pipeline = pipeline.fit(train_data)
                
                # Validate
                val_predictions = model_pipeline.transform(val_data)
                
                # Evaluate
                metrics = {}
                for metric_name, evaluator in self.evaluators.items():
                    metrics[metric_name] = evaluator.evaluate(val_predictions)
                
                stage2_performance[name] = metrics
                stage2_predictions[name] = model_pipeline
                
                logger.info(f"{name} - RMSE: {metrics['rmse']:.4f}, "
                           f"MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
            
            return stage2_predictions, stage2_performance
            
        except Exception as e:
            logger.error(f"Stage 2 training failed: {e}")
            raise
    
    def create_meta_features(self, data, stage1_models, stage2_models):
        """Create meta-features from base model predictions"""
        try:
            # Get predictions from all base models
            all_predictions = data.select("date", "adjusted close")
            
            prediction_columns = []
            
            # Stage 1 predictions
            for name, model in stage1_models.items():
                pred = model.transform(data)
                pred_col = f"pred_{name}"
                prediction_columns.append(pred_col)
                pred = pred.select("date", col("prediction").alias(pred_col))
                all_predictions = all_predictions.join(pred, on="date", how="inner")
            
            # Stage 2 predictions
            for name, model in stage2_models.items():
                pred = model.transform(data)
                pred_col = f"pred_{name}"
                prediction_columns.append(pred_col)
                pred = pred.select("date", col("prediction").alias(pred_col))
                all_predictions = all_predictions.join(pred, on="date", how="inner")
            
            # Create meta-features vector
            meta_assembler = VectorAssembler(inputCols=prediction_columns, outputCol="meta_features")
            meta_data = meta_assembler.transform(all_predictions)
            
            logger.info(f"Meta-features created with {len(prediction_columns)} base predictions")
            return meta_data
            
        except Exception as e:
            logger.error(f"Meta-feature creation failed: {e}")
            raise
    
    def train_meta_model(self, train_data, val_data, stage1_models, stage2_models):
        """Train the meta-model on base model predictions"""
        try:
            logger.info("Training meta-model for stacking ensemble")
            
            # Create meta-features for training and validation
            train_meta = self.create_meta_features(train_data, stage1_models, stage2_models)
            val_meta = self.create_meta_features(val_data, stage1_models, stage2_models)
            
            # Train meta-model
            meta_model_fitted = self.meta_model.fit(train_meta)
            
            # Validate meta-model
            meta_predictions = meta_model_fitted.transform(val_meta)
            
            # Evaluate meta-model
            meta_performance = {}
            for metric_name, evaluator in self.evaluators.items():
                meta_performance[metric_name] = evaluator.evaluate(meta_predictions)
            
            logger.info(f"Meta-model - RMSE: {meta_performance['rmse']:.4f}, "
                       f"MAE: {meta_performance['mae']:.4f}, R2: {meta_performance['r2']:.4f}")
            
            return meta_model_fitted, meta_performance
            
        except Exception as e:
            logger.error(f"Meta-model training failed: {e}")
            raise
    
    def run_pipeline(self, file_path: str):
        """Run the complete prediction pipeline"""
        try:
            logger.info("Starting stock prediction pipeline")
            
            # Initialize Spark
            self.initialize_spark()
            
            # Load and prepare data
            df = self.load_data(file_path)
            df_features = self.engineer_features(df)
            train_data, val_data, test_data = self.prepare_data(df_features)
            
            # Initialize models
            self.initialize_models()
            
            # Train Stage 1 models
            stage1_models, stage1_performance = self.train_stage1_models(train_data, val_data)
            
            # Train Stage 2 models
            stage2_models, stage2_performance = self.train_stage2_models(train_data, val_data)
            
            # Train meta-model
            meta_model, meta_performance = self.train_meta_model(
                train_data, val_data, stage1_models, stage2_models
            )
            
            # Final evaluation on test set
            test_meta = self.create_meta_features(test_data, stage1_models, stage2_models)
            final_predictions = meta_model.transform(test_meta)
            
            final_performance = {}
            for metric_name, evaluator in self.evaluators.items():
                final_performance[metric_name] = evaluator.evaluate(final_predictions)
            
            logger.info("=== FINAL TEST RESULTS ===")
            logger.info(f"Ensemble Model - RMSE: {final_performance['rmse']:.4f}, "
                       f"MAE: {final_performance['mae']:.4f}, R2: {final_performance['r2']:.4f}")
            
            # Compare with best individual model
            best_individual = min(
                list(stage1_performance.values()) + list(stage2_performance.values()),
                key=lambda x: x['rmse']
            )
            
            improvement = (best_individual['rmse'] - final_performance['rmse']) / best_individual['rmse'] * 100
            logger.info(f"Improvement over best individual model: {improvement:.2f}%")
            
            return {
                'stage1_performance': stage1_performance,
                'stage2_performance': stage2_performance,
                'meta_performance': meta_performance,
                'final_performance': final_performance,
                'models': {
                    'stage1': stage1_models,
                    'stage2': stage2_models,
                    'meta': meta_model
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("Spark session stopped")

def main():
    """Main execution function"""
    try:
        pipeline = StockPredictionPipeline()
        results = pipeline.run_pipeline("data_stocks.csv")
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Check stock_prediction.log for detailed logs")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()