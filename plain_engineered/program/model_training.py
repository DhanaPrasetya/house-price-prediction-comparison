import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# ---------------------------------------------------------
# 1. LOAD TRAIN & TEST DATASETS FROM CSV
# ---------------------------------------------------------
train_df = pd.read_csv('../output/housing_train.csv')
test_df = pd.read_csv('../output/housing_test.csv')

# Pipeline for Numerical Features:
# - Impute missing values using Column Median
# - Scale features using StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Features (e.g., 'city'):
# - Impute missing values using Column Mode (most frequent)
# - One-Hot Encode text values into binary indicator columns
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Drop corrupted zero or negative prices if present
train_df = train_df[train_df['price'] > 0]
test_df = test_df[test_df['price'] > 0]

# Separate Features (X) and Target (y)
X_train = train_df.drop(columns=['price'])
X_test = test_df.drop(columns=['price'])

# Transform target to log scale for training: y = log(1 + price)
y_train_log = np.log1p(train_df['price'])
y_test_dollars = test_df['price']  # Kept in actual dollars for final evaluation

# ---------------------------------------------------------
# 2. IDENTIFY COLUMN TYPES & BUILD PREPROCESSOR
# ---------------------------------------------------------
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Re-use preprocessor pipelines defined earlier
transformers = [('num', num_pipeline, num_cols)]
if len(cat_cols) > 0:
    transformers.append(('cat', cat_pipeline, cat_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# ---------------------------------------------------------
# 3. FULL MODEL PIPELINE (PREPROCESSING + LINEAR REGRESSION)
# ---------------------------------------------------------
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit on training CSV data using log-transformed price target
model_pipeline.fit(X_train, y_train_log)

# ---------------------------------------------------------
# 4. PREDICT & EVALUATE
# ---------------------------------------------------------
# Model outputs predictions in LOG scale
y_pred_log = model_pipeline.predict(X_test)

# Clip predictions to ensure they are within the range of the training log prices
max_log_limit = y_train_log.max()
y_pred_log_clipped = np.clip(y_pred_log, a_min=y_train_log.min(), a_max=max_log_limit)

# Exponentiate safe clipped predictions
y_pred_dollars = np.expm1(y_pred_log_clipped)

# Calculate evaluation metrics in real dollars against raw test prices
mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
r2 = r2_score(y_test_dollars, y_pred_dollars)

# Save the trained model pipeline to a file
joblib.dump(model_pipeline, '../output/housing_linear_model.pkl')
print("Model saved successfully as 'housing_linear_model.pkl'")

print("--- Linear Regression Evaluation ---")
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")