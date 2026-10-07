# OIBSIP-DataAnalytics-L1-EDARetailSales-
Data analytics ,EDA Retailsales Task 1
The dataset contains retail transaction details, including customer information, product details, sales amount, quantity, discounts, payment methods, sales channels, and regions.
 
CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_sales_dataset.csv")

print("=" * 70)
print("DATASET LOADED")
print("=" * 70)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nNull Values:")
print(df.isnull().sum())

print("\nNull Value Percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

duplicates = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    errors="coerce"
)

print("\nMinimum Date:", df["transaction_date"].min())
print("Maximum Date:", df["transaction_date"].max())

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

mean_values = df[numeric_columns].mean()
median_values = df[numeric_columns].median()
mode_values = df[numeric_columns].mode().iloc[0]
std_values = df[numeric_columns].std()

statistics = pd.DataFrame({
    "Mean": mean_values,
    "Median": median_values,
    "Mode": mode_values,
    "Standard Deviation": std_values
})

print("\nDescriptive Statistics:")
print(statistics)

print("\nPandas Descriptive Statistics:")
print(df[numeric_columns].describe())

total_sales = df["sales_amount"].sum()
average_sales = df["sales_amount"].mean()
total_quantity = df["quantity"].sum()
total_transactions = df["transaction_id"].nunique()

print("\nTotal Transactions:", total_transactions)
print("Total Quantity:", total_quantity)
print("Total Sales:", round(total_sales, 2))
print("Average Sales:", round(average_sales, 2))

df["month"] = df["transaction_date"].dt.to_period("M")

monthly_sales = df.groupby(
    "month"
)["sales_amount"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

plt.figure(figsize=(14, 6))
plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df["quarter"] = df["transaction_date"].dt.to_period("Q")

quarterly_sales = df.groupby(
    "quarter"
)["sales_amount"].sum()

print("\nQuarterly Sales:")
print(quarterly_sales)

plt.figure(figsize=(12, 6))
plt.plot(
    quarterly_sales.index.astype(str),
    quarterly_sales.values,
    marker="o"
)
plt.title("Quarterly Sales Trend")
plt.xlabel("Quarter")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

gender_count = df["customer_gender"].value_counts()

print("\nCustomer Gender Distribution:")
print(gender_count)

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="customer_gender",
    order=gender_count.index
)
plt.title("Customer Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

age_group_count = df["customer_age_group"].value_counts()

print("\nCustomer Age Group Distribution:")
print(age_group_count)

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x="customer_age_group",
    order=age_group_count.index
)
plt.title("Customer Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_products = (
    df.groupby("product_name")["quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Best-Selling Products:")
print(top_products)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=top_products.values,
    y=top_products.index
)
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

category_revenue = (
    df.groupby("category")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Product Category:")
print(category_revenue)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=category_revenue.values,
    y=category_revenue.index
)
plt.title("Revenue by Product Category")
plt.xlabel("Revenue")
plt.ylabel("Product Category")
plt.tight_layout()
plt.show()

correlation_columns = [
    "quantity",
    "unit_price",
    "discount_pct",
    "sales_amount"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

segment_sales = (
    df.groupby("customer_segment")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Customer Segment:")
print(segment_sales)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=segment_sales.values,
    y=segment_sales.index
)
plt.title("Revenue by Customer Segment")
plt.xlabel("Sales Amount")
plt.ylabel("Customer Segment")
plt.tight_layout()
plt.show()

channel_sales = (
    df.groupby("sales_channel")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Sales Channel:")
print(channel_sales)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=channel_sales.index,
    y=channel_sales.values
)
plt.title("Revenue by Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Sales Amount")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

best_category = category_revenue.idxmax()
best_product = top_products.idxmax()
best_segment = segment_sales.idxmax()
best_channel = channel_sales.idxmax()

gender_sales = (
    df.groupby("customer_gender")["sales_amount"]
    .sum()
    .sort_values(ascending=False)
)

best_gender = gender_sales.idxmax()

print("\nBest Performing Category:", best_category)
print("Best-Selling Product:", best_product)
print("Best Customer Segment:", best_segment)
print("Best Sales Channel:", best_channel)
print("Best Gender by Revenue:", best_gender)

print("\nKEY INSIGHTS")

print(f"Total Sales: {total_sales:,.2f}")
print(f"Total Quantity Sold: {total_quantity:,}")
print(f"Best-Selling Product: {best_product}")
print(f"Best Product Category: {best_category}")
print(f"Best Customer Segment: {best_segment}")
print(f"Best Sales Channel: {best_channel}")
print(f"Highest-Revenue Gender Group: {best_gender}")

print("\nBUSINESS RECOMMENDATIONS")

print(
    f'1. Maintain sufficient inventory for the best-selling product "{best_product}".'
)

print(
    f'2. Focus marketing activities on the high-revenue "{best_category}" category.'
)

print(
    f'3. Use loyalty programs and personalized offers for the "{best_segment}" segment.'
)

print(
    f'4. Continue improving the "{best_channel}" sales channel.'
)

print(
    "5. Use customer demographic information to create targeted marketing campaigns."
)

df.to_csv(
    "retail_sales_cleaned.csv",
    index=False
)

statistics.to_csv(
    "retail_descriptive_statistics.csv"
)

category_revenue.to_csv(
    "revenue_by_category.csv"
)

top_products.to_csv(
    "top_10_products.csv"
)

print("\nFiles created successfully:")
print("retail_sales_cleaned.csv")
print("retail_descriptive_statistics.csv")
print("revenue_by_category.csv")
print("top_10_products.csv")

print("\nRETAIL SALES EDA COMPLETED SUCCESSFULLY")
 <img width="1920" height="1080" alt="Screenshot 2026-10-07 104240" src="https://github.com/user-attachments/assets/3bf31239-ba6c-4594-8811-3b6470204d17" />

 

 
 

 
 

