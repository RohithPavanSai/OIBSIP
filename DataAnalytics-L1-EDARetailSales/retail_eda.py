import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. [span_0](start_span)Generate local retail dataset[span_0](end_span)
np.random.seed(42)
n = 300
data = {
    'Date': pd.date_range(start='2025-01-01', periods=n, freq='D'),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Age': np.random.randint(18, 65, n),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Beauty', 'Home'], n),
    'Total_Amount': np.random.randint(20, 500, n)
}
df = pd.DataFrame(data)

sns.set_theme(style="whitegrid")

# 2. [span_1](start_span)Descriptive Statistics[span_1](end_span)
print("--- Descriptive Summary ---")
print(df.describe())

# 3. [span_2](start_span)Time Series Analysis (Monthly Sales Trends)[span_2](end_span)
df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
monthly_sales = df.groupby('Year_Month')['Total_Amount'].sum().reset_index()

plt.figure(figsize=(8, 4))
sns.lineplot(data=monthly_sales, x='Year_Month', y='Total_Amount', marker='o', color='purple')
plt.title('Monthly Sales Revenue Trend')
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
plt.close()

# 4. [span_3](start_span)Demographics Analysis[span_3](end_span)
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', bins=15, kde=True, color='blue')
plt.title('Customer Age Distribution')
plt.tight_layout()
plt.savefig('age_distribution.png')
plt.close()

# 5. [span_4](start_span)Product Category Analysis[span_4](end_span)
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x='Product_Category', y='Total_Amount', estimator=sum, errorbar=None, palette='muted')
plt.title('Revenue by Product Category')
plt.tight_layout()
plt.savefig('category_revenue.png')
plt.close()

print("\nEDA processing complete! Visual charts saved to folder.")