import pandas as pd
import numpy as np

# 1. Generate local messy data for execution
data = {
    'Customer_ID': ['C101', 'C102', 'C103', 'C101', 'C104', 'C105', 'C106'],
    'Join_Date': ['2025-01-01', '2025/01/02', '03-01-2025', '2025-01-01', np.nan, '2025-01-05', '2025-01-06'],
    'Gender': ['Male', 'female', 'M', 'Male', 'F', 'Female', np.nan],
    'Age': [28, 34, -5, 28, 45, 120, 23],
    'Income': [55000, np.nan, 62000, 55000, 95000, 70000, 48000]
}
df = pd.DataFrame(data)

# Baseline metrics for required Before vs After summary table
before_rows = len(df)
before_nulls = df.isnull().sum().sum()
before_duplicates = df.duplicated().sum()

# 2. Drop Duplicate Rows
df = df.drop_duplicates()

# 3. Standardize Formatting (Gender strings)
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female', 'Nan': np.nan})

# 4. Correct Data Types (Dates)
df['Join_Date'] = pd.to_datetime(df['Join_Date'], errors='coerce')

# 5. Outlier Detection & Handling (Logical constraints)
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# 6. Missing Data Imputation (Median for numerical, Mode for categorical)
df['Income'] = df['Income'].fillna(df['Income'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# 7. Print Summary Metrics (Before vs After)
print("--- Before vs After Summary Table ---")
print(f"Total Rows: {before_rows} -> {len(df)}")
print(f"Total Null Values: {before_nulls} -> {df.isnull().sum().sum()}")
print(f"Duplicate Count: {before_duplicates} -> {df.duplicated().sum()}")

# 8. Save output asset
df.to_csv('cleaned_customer_data.csv', index=False)
print("\nCleaned file saved as 'cleaned_customer_data.csv'!")