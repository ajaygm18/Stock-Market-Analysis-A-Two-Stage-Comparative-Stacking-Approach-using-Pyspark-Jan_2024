from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, functions as F, Window
from functools import reduce

# Create a Spark session
spark = SparkSession.builder.appName("ImprovedStockPrediction").getOrCreate()

# Load the data from the CSV into a DataFrame
df = spark.read.csv("data_stocks.csv", header=True, inferSchema=True)

# Window function to order by date (for features that require looking at previous rows)
w = Window.partitionBy().orderBy("date")

# Feature Engineering
# 1. Create a lagged column for the 'close' price (i.e., previous day's close)
df = df.withColumn("lag_close_1", F.lag(df.close).over(w))

# 2. Calculate daily return - how much did the price change in percentage from opening to closing
df = df.withColumn("daily_return", (df.close - df.open) / df.open)

# 3. Calculate intra-day volatility - difference between the highest and lowest prices of the day
df = df.withColumn("intra_day_volatility", df.high - df.low)

# 4. Calculate daily volatility - change in closing price from the previous day
df = df.withColumn("daily_volatility", df.close - df.lag_close_1)

# 5. Calculate a 7-day moving average for the closing prices
df = df.withColumn("7_day_avg_close", F.avg(df.close).over(w.rowsBetween(-6, 0)))

# 6. Binary indicator if there was a dividend
df = df.withColumn("is_dividend", F.when(df["dividend amount"] > 0, 1).otherwise(0))

# 7. Binary indicator if there was a stock split
df = df.withColumn("is_split", F.when(df["split coefficient"] != 1, 1).otherwise(0))

# Drop any rows with NA values (which might have been introduced due to lagging operations)
df = df.dropna()

# Assemble the features into a single vector for MLlib consumption
feature_columns = ["open", "high", "low", "close", "volume", "daily_return",
                   "intra_day_volatility", "daily_volatility", "7_day_avg_close",
                   "is_dividend", "is_split"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)

# Split the data into a training set and a test set.
# Here, we're using a time-based split - 80% for training and 20% for testing
train_count = int(df_assembled.count() * 0.8)
train_data = df_assembled.orderBy("date").limit(train_count)
test_data = df_assembled.subtract(train_data)

# Feature scaling for all models
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Stage 1: Simple Linear Models
lasso = LinearRegression(labelCol="adjusted close", featuresCol="scaled_features", maxIter=50, regParam=0.1, elasticNetParam=1.0)
elastic_net = LinearRegression(labelCol="adjusted close", featuresCol="scaled_features", maxIter=50, regParam=0.1, elasticNetParam=0.5)
ridge = LinearRegression(labelCol="adjusted close", featuresCol="scaled_features", maxIter=50, regParam=0.01, elasticNetParam=0.0)
linear_reg = LinearRegression(labelCol="adjusted close", featuresCol="scaled_features", maxIter=50)
stage1_models = [lasso, elastic_net, ridge, linear_reg]

# Stage 2: Nonlinear Regressors
decision_tree = DecisionTreeRegressor(labelCol="adjusted close", featuresCol="scaled_features", maxDepth=5)
knn = RandomForestRegressor(labelCol="adjusted close", featuresCol="scaled_features", numTrees=5)
stage2_models = [decision_tree, knn]

# Evaluate the models and print the results
evaluator_rmse = RegressionEvaluator(labelCol="adjusted close", metricName="rmse")

# Iterate through the stages
for stage, all_models in enumerate([stage1_models, stage2_models], start=1):
    print(f"Stage {stage}:")

    # Create a list to store models' predictions
    stage_predictions = []

    # Create a pipeline for each model
    for model in all_models:
        pipeline = Pipeline(stages=[scaler, model])
        model_pipeline = pipeline.fit(train_data)
        predictions = model_pipeline.transform(test_data)
        stage_predictions.append(predictions)

        # Evaluate the model
        rmse = evaluator_rmse.evaluate(predictions)
        print(f"Model: {model.__class__.__name__}, RMSE: {rmse}")

    # Combine the predictions from all models in the stage using reduce
    combined_predictions = reduce(lambda df1, df2: df1.join(df2, on='date', how='inner'), stage_predictions)
# Your existing Python code for model training

