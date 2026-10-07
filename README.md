# OIBSIP-DataAnalytics-L1-EDARetailSales-
CODE:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")
df = pd.read_csv("retail_sales_dataset.csv")

print("\n" + "="*70)
print("DATASET LOADED")
print("="*70)

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ================================================================
# 3. INITIAL DATA INSPECTION
# ================================================================

print("\n" + "="*70)
print("COLUMN NAMES")
print("="*70)

print(df.columns.tolist())


print("\n" + "="*70)
print("DATA TYPES")
print("="*70)

print(df.dtypes)


print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)

df.info()


# ================================================================
# 4. NULL VALUE CHECK
# ================================================================

print("\n" + "="*70)
print("NULL VALUE CHECK")
print("="*70)

null_values = df.isnull().sum()

print(null_values)

print("\nNull value percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))


# ================================================================
# 5. DUPLICATE CHECK
# ================================================================

print("\n" + "="*70)
print("DUPLICATE CHECK")
print("="*70)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ================================================================
# 6. CONVERT DATE COLUMN
# ================================================================

df['transaction_date'] = pd.to_datetime(
    df['transaction_date'],
    errors='coerce'
)

print("\nTransaction date converted successfully.")

print("Minimum date:", df['transaction_date'].min())
print("Maximum date:", df['transaction_date'].max())


# ================================================================
# 7. DESCRIPTIVE STATISTICS
# Mean, Median, Mode, Standard Deviation
# ================================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\n" + "="*70)
print("NUMERICAL COLUMNS")
print("="*70)

print(numeric_columns)


# Mean
mean_values = df[numeric_columns].mean()

# Median
median_values = df[numeric_columns].median()

# Mode
mode_values = df[numeric_columns].mode().iloc[0]

# Standard deviation
std_values = df[numeric_columns].std()


statistics = pd.DataFrame({
    "Mean": mean_values,
    "Median": median_values,
    "Mode": mode_values,
    "Standard Deviation": std_values
})

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print(statistics)


# ================================================================
# 8. BASIC DATASET STATISTICS
# ================================================================

print("\n" + "="*70)
print("PANDAS DESCRIPTIVE STATISTICS")
print("="*70)

print(df[numeric_columns].describe())


# ================================================================
# 9. BASIC BUSINESS METRICS
# ================================================================

total_sales = df['sales_amount'].sum()
average_sales = df['sales_amount'].mean()
total_quantity = df['quantity'].sum()
total_transactions = df['transaction_id'].nunique()

print("\n" + "="*70)
print("BASIC BUSINESS METRICS")
print("="*70)

print("Total Transactions :", total_transactions)
print("Total Quantity     :", total_quantity)
print("Total Sales        :", round(total_sales, 2))
print("Average Sales      :", round(average_sales, 2))


# ================================================================
# 10. TIME SERIES ANALYSIS
# MONTHLY SALES
# ================================================================

df['month'] = df['transaction_date'].dt.to_period('M')

monthly_sales = (
    df.groupby('month')['sales_amount']
    .sum()
)

print("\n" + "="*70)
print("MONTHLY SALES")
print("="*70)

print(monthly_sales)


plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker='o'
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("""
OBSERVATION:
The monthly sales line chart shows how sales changed over time.
Peaks indicate months with stronger sales performance, while
lower points indicate relatively weaker sales periods.
""")


# ================================================================
# 11. QUARTERLY SALES TREND
# ================================================================

df['quarter'] = df['transaction_date'].dt.to_period('Q')

quarterly_sales = (
    df.groupby('quarter')['sales_amount']
    .sum()
)

print("\n" + "="*70)
print("QUARTERLY SALES")
print("="*70)

print(quarterly_sales)


plt.figure(figsize=(12, 6))

plt.plot(
    quarterly_sales.index.astype(str),
    quarterly_sales.values,
    marker='o'
)

plt.title("Quarterly Sales Trend")
plt.xlabel("Quarter")
plt.ylabel("Sales Amount")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("""
OBSERVATION:
The quarterly trend provides a higher-level view of sales
performance and helps identify strong and weak business periods.
""")


# ================================================================
# 12. CUSTOMER DEMOGRAPHICS
# GENDER BREAKDOWN
# ================================================================

gender_count = df['customer_gender'].value_counts()

print("\n" + "="*70)
print("CUSTOMER GENDER DISTRIBUTION")
print("="*70)

print(gender_count)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x='customer_gender',
    order=gender_count.index
)

plt.title("Customer Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

print("""
OBSERVATION:
The gender distribution shows the representation of each gender
among the retail transactions. This can help the business understand
the composition of its customer base.
""")


# ================================================================
# 13. CUSTOMER AGE GROUP DISTRIBUTION
# ================================================================

age_group_count = df['customer_age_group'].value_counts()

print("\n" + "="*70)
print("CUSTOMER AGE GROUP DISTRIBUTION")
print("="*70)

print(age_group_count)


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x='customer_age_group',
    order=age_group_count.index
)

plt.title("Customer Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("""
OBSERVATION:
The age-group distribution identifies which customer age groups
are most represented in the dataset. This information can support
age-specific marketing and product strategies.
""")


# ================================================================
# 14. PRODUCT ANALYSIS
# TOP 10 BEST-SELLING PRODUCTS
# ================================================================

top_products = (
    df.groupby('product_name')['quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n" + "="*70)
print("TOP 10 BEST-SELLING PRODUCTS")
print("="*70)

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

print("""
OBSERVATION:
The top 10 products highlight the products with the highest sales
volume. These products may require stronger inventory planning
and continued promotional support.
""")


# ================================================================
# 15. REVENUE BY PRODUCT CATEGORY
# ================================================================

category_revenue = (
    df.groupby('category')['sales_amount']
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "="*70)
print("REVENUE BY PRODUCT CATEGORY")
print("="*70)

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

print("""
OBSERVATION:
The category revenue chart identifies the product categories that
contribute most to total revenue. High-revenue categories should
receive appropriate inventory and marketing attention.
""")


# ================================================================
# 16. CORRELATION HEATMAP
# ================================================================

correlation_columns = [
    'quantity',
    'unit_price',
    'discount_pct',
    'sales_amount'
]

correlation_matrix = df[correlation_columns].corr()

print("\n" + "="*70)
print("CORRELATION MATRIX")
print("="*70)

print(correlation_matrix)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Matrix of Numerical Variables")

plt.tight_layout()
plt.show()

print("""
OBSERVATION:
The correlation heatmap shows the strength and direction of
relationships between numerical variables. Strong positive values
indicate variables that tend to increase together, while negative
values indicate an inverse relationship.
""")


# ================================================================
# 17. ADDITIONAL VISUALIZATION
# SALES BY CUSTOMER SEGMENT
# ================================================================

segment_sales = (
    df.groupby('customer_segment')['sales_amount']
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "="*70)
print("SALES BY CUSTOMER SEGMENT")
print("="*70)

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

print("""
OBSERVATION:
The customer-segment analysis reveals which customer groups
generate more revenue. This can uncover differences in customer
value that may not be visible from demographic distributions alone.
""")


# ================================================================
# 18. ADDITIONAL VISUALIZATION
# SALES CHANNEL ANALYSIS
# ================================================================

channel_sales = (
    df.groupby('sales_channel')['sales_amount']
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "="*70)
print("SALES BY SALES CHANNEL")
print("="*70)

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

print("""
OBSERVATION:
The sales-channel comparison shows which channels contribute
the most revenue. This can help the business prioritize resources
and improve weaker channels.
""")


# ================================================================
# 19. FIND BEST PERFORMING CATEGORY
# ================================================================

best_category = category_revenue.idxmax()
best_category_revenue = category_revenue.max()

print("\nBest Performing Category:", best_category)
print("Revenue:", round(best_category_revenue, 2))


# ================================================================
# 20. FIND BEST SELLING PRODUCT
# ================================================================

best_product = top_products.idxmax()
best_product_quantity = top_products.max()

print("\nBest-Selling Product:", best_product)
print("Quantity Sold:", best_product_quantity)


# ================================================================
# 21. FIND BEST CUSTOMER SEGMENT
# ================================================================

best_segment = segment_sales.idxmax()
best_segment_sales = segment_sales.max()

print("\nBest Customer Segment:", best_segment)
print("Sales:", round(best_segment_sales, 2))


# ================================================================
# 22. FIND BEST SALES CHANNEL
# ================================================================

best_channel = channel_sales.idxmax()
best_channel_sales = channel_sales.max()

print("\nBest Sales Channel:", best_channel)
print("Sales:", round(best_channel_sales, 2))


# ================================================================
# 23. FIND BEST GENDER BY REVENUE
# ================================================================

gender_sales = (
    df.groupby('customer_gender')['sales_amount']
    .sum()
    .sort_values(ascending=False)
)

best_gender = gender_sales.idxmax()

print("\nBest Gender by Revenue:", best_gender)
print("Revenue:", round(gender_sales.max(), 2))


# ================================================================
# 24. FINAL KEY INSIGHTS
# ================================================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print(f"""
1. Total sales generated:
   {total_sales:,.2f}

2. Total quantity sold:
   {total_quantity:,}

3. Best-selling product:
   {best_product}

4. Best product category by revenue:
   {best_category}

5. Highest-value customer segment:
   {best_segment}

6. Best-performing sales channel:
   {best_channel}

7. Highest-revenue gender group:
   {best_gender}
""")


# ================================================================
# 25. BUSINESS RECOMMENDATIONS
# ================================================================

print("\n" + "="*70)
print("BUSINESS RECOMMENDATIONS")
print("="*70)

print(f"""
1. PRIORITIZE HIGH-PERFORMING PRODUCTS
   Focus inventory planning and promotional activities on the
   best-selling products such as "{best_product}". Maintaining
   adequate stock can help prevent lost sales opportunities.

2. INVEST IN HIGH-REVENUE CATEGORIES
   The "{best_category}" category generates the highest revenue.
   The business should maintain strong inventory availability
   and targeted promotions for this category.

3. TARGET HIGH-VALUE CUSTOMER SEGMENTS
   The "{best_segment}" customer segment generates the highest
   revenue. Personalized promotions, loyalty programs and
   targeted marketing can be used to increase customer retention.

4. STRENGTHEN THE BEST SALES CHANNEL
   The "{best_channel}" channel contributes the highest revenue.
   The company should continue improving customer experience
   and marketing investment in this channel.

5. USE DEMOGRAPHIC INSIGHTS FOR MARKETING
   Customer age and gender distributions can be used to create
   targeted campaigns and offers for the most important customer
   groups.
""")


# ================================================================
# 26. CONCLUSION
# ================================================================

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("""
The Exploratory Data Analysis of the retail sales dataset was
successfully completed using Python, Pandas, Matplotlib and Seaborn.

The analysis covered dataset inspection, missing values,
descriptive statistics, monthly and quarterly sales trends,
customer demographics, product performance, revenue by category,
correlation analysis and additional business visualizations.

The analysis helps identify important customer behaviour,
product performance and sales patterns. These insights can support
better inventory management, targeted marketing, customer
segmentation and sales-channel decisions.

The findings provide actionable information that can be used
by the business to improve sales performance and customer
engagement.
""")


# ================================================================
# 27. SAVE RESULTS
# ================================================================

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

print("\n" + "="*70)
print("OUTPUT FILES CREATED")
print("="*70)

print("""
1. retail_sales_cleaned.csv
2. retail_descriptive_statistics.csv
3. revenue_by_category.csv
4. top_10_products.csv
""")

print("="*70)
print("OASIS INFOBYTE LEVEL 1 TASK 1 COMPLETED")
print("="*70)
Data analytics ,EDA Retailsales Task 1
The dataset contains retail transaction details, including customer information, product details, sales amount, quantity, discounts, payment methods, sales channels, and regions.
It is used to perform Exploratory Data Analysis (EDA) to understand sales trends, customer behavior, product performance, and overall business performance.<img width="1920" height="1080" alt="Screenshot 2026-10-07 104240" src="https://github.com/user-attachments/assets/1fc7e9aa-bbd3-4f5a-b1a5-b82c0c3881ca" />


