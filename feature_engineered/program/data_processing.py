import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

os.makedirs('../output', exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA & INITIAL FEATURE ENGINEERING
# ---------------------------------------------------------
df = pd.read_csv('../../input/raw_data.csv')

# Example placeholder structure matching your schema
features_df = df.copy()

# A. Drop text columns with too many unique values (high cardinality)
columns_to_drop = ['street', 'statezip', 'date', 'country']
features_df = features_df.drop(
    columns=[col for col in columns_to_drop if col in features_df.columns]
)

# B. Feature engineering for year columns
if 'yr_built' in features_df.columns:
  features_df['house_age'] = 2026 - features_df['yr_built']
  features_df = features_df.drop(columns=['yr_built'])

if 'yr_renovated' in features_df.columns:
  features_df['is_renovated'] = (features_df['yr_renovated'] > 0).astype(int)
  features_df = features_df.drop(columns=['yr_renovated'])

# Separate Features (X) and Target (y)
X = features_df.drop(columns=['price'])
# Log-transform target variable to stabilize variance and handle price outliers
y = np.log1p(features_df['price'])

# ---------------------------------------------------------
# 2. SPLIT INTO TRAINING & TESTING SETS
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# remove rows where the target variable is zero or negative (if any)
non_zero_mask = y_train > 0

X_train = X_train[non_zero_mask]
y_train = y_train[non_zero_mask]

# Drop rows where living area exists (> 0) but bedrooms or bathrooms are 0
invalid_house_mask = (df['sqft_living'] > 0) & (
    (df['bedrooms'] == 0) | (df['bathrooms'] == 0)
)

df_clean = df[~invalid_house_mask].copy()

print(
    f'\nRemoved {invalid_house_mask.sum()} corrupted 0-bed/0-bath house records.\n'
)

# Identify numerical and categorical column names
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Save original dollar prices instead of log values
train_df = X_train.copy()
train_df['price'] = np.expm1(y_train)

test_df = X_test.copy()
test_df['price'] = np.expm1(y_test)

train_df.to_csv('../output/housing_train.csv', index=False)
test_df.to_csv('../output/housing_test.csv', index=False)

print("Saved 'housing_train.csv' and 'housing_test.csv' successfully!")

# ---------------------------------------------------------
# 3. VISUALIZE FEATURE SUMMARY, CORRELATION, AND TARGET DISTRIBUTION
# ---------------------------------------------------------
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

# Prepare dataframe combining numeric train features and untransformed raw price
analysis_df = X_train[num_cols].copy()
analysis_df['price ($)'] = np.expm1(y_train)

# --- A. FEATURE SUMMARY TABLE VISUAL (Min, Max, Mode, Std Dev, Median) ---
stats_df = pd.DataFrame({
    'Feature': analysis_df.columns,
    'Min': analysis_df.min().values,
    'Max': analysis_df.max().values,
    'Mode': analysis_df.mode().iloc[0].values,
    'Std Dev': analysis_df.std().values,
    'Median': analysis_df.median().values,
})

# Round numeric columns to prevent string formatting issues
numeric_cols = ['Min', 'Max', 'Mode', 'Std Dev', 'Median']
stats_df[numeric_cols] = stats_df[numeric_cols].round(2)

fig, ax = plt.subplots(figsize=(11, len(stats_df) * 0.5 + 1.5))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=stats_df.values,
    colLabels=stats_df.columns,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.3)

plt.title('Train Set Features Overview (Numeric + Price)', pad=20, fontsize=12)
plt.savefig(
    '../output/summary_statistics_table.jpg',
    dpi=300,
    format='jpg',
    bbox_inches='tight',
)
plt.close()

# --- B. CORRELATION MATRIX VISUAL ---
plt.figure(figsize=(10, 8))
corr_matrix = analysis_df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    cbar=True,
)
plt.title('Train Set Pairwise Correlation Matrix', fontsize=12)
plt.tight_layout()
plt.savefig(
    '../output/correlation_matrix.jpg', dpi=300, format='jpg'
)
plt.close()

# --- C. TARGET DISTRIBUTION VISUAL (Raw Price vs Log-Transformed Price) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Raw Price
sns.histplot(analysis_df['price ($)'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Original Target Distribution (Price $)')
axes[0].set_xlabel('Price ($)')
axes[0].ticklabel_format(style='plain', axis='x')

# Plot 2: Log-Transformed Price (y_train)
sns.histplot(y_train, kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Log-Transformed Target Distribution log1p(Price)')
axes[1].set_xlabel('log1p(Price)')

plt.tight_layout()
plt.savefig(
    '../output/target_distribution.jpg', dpi=300, format='jpg'
)
plt.close()

print(
    "Saved visualization outputs to '../output/':"
    "\n - summary_statistics_table.jpg"
    "\n - correlation_matrix.jpg"
    "\n - target_distribution.jpg"
)