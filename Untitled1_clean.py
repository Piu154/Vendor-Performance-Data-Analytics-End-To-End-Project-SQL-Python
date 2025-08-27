import os, glob, shutil
import logging
import time

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/ingestion_db.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# Ensure data directory exists and copy CSVs into it
os.makedirs('data', exist_ok=True)
for path in glob.glob('*.csv'):
    dest = os.path.join('data', os.path.basename(path))
    if not os.path.exists(dest):
        shutil.copy(path, dest)

print("Files in data/:", os.listdir('data'))

# 1) Imports + SQLite engine
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///inventory.db')

# ✅ define the ingest function first
def ingest_db(df, table_name, engine):
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    logging.info(f"{table_name} ingested into SQLite, shape={df.shape}")

# 2) load_raw_data function
def load_raw_data():
    start = time.time()
    for file in os.listdir('data'):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join('data', file))
            table_name = os.path.splitext(file)[0]
            print(f"Ingesting {table_name}, shape={df.shape}")
            ingest_db(df, table_name, engine)
    end = time.time()
    total_time = (end - start) / 60
    logging.info('-----Ingestion complete-----')
    logging.info(f'Total Time Taken: {total_time:.2f} minutes')

# 3) Call the function to actually ingest data
load_raw_data()

import pandas as pd
import sqlite3
conn=sqlite3.connect('inventory.db')
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)
df = pd.read_sql("SELECT COUNT(*) FROM purchases", conn)
print(df)
for table in tables['name']:
    print('-' * 50, f"Table: {table}", '-' * 50)

    # ✅ Use proper f-string placement
    count = pd.read_sql(f"SELECT COUNT(*) AS count FROM {table}", conn)['count'].values[0]
    print("Count of records:", count)
    display(pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn))

purchases = pd.read_sql_query(
    "SELECT * FROM purchases WHERE VendorNumber = 4466", conn
)

purchase_price = pd.read_sql_query(
    "SELECT * FROM purchase_prices WHERE VendorNumber = 4466", conn
)

vendor_invoice = pd.read_sql_query(
    "SELECT * FROM vendor_invoice WHERE VendorNumber = 4466", conn
)


sales = pd.read_sql_query(
    "SELECT * FROM sales WHERE VendorNo = 4466", conn
)


# display results
print("\n=== purchases ===")
print(purchases)

print("\n=== purchase_price ===")
print(purchase_price)

print("\n=== vendor_invoice ===")
print(vendor_invoice)


print("\n=== sales ===")
print(sales)

purchases.groupby(['Brand', 'PurchasePrice'])[['Quantity', 'Dollars']].sum()
# print(purchases.columns)
purchase_price
vendor_invoice['PONumber'].nunique()
vendor_invoice.shape
sales.groupby('Brand')[['SalesDollars', 'SalesPrice', 'SalesQuantity']].sum()

freight_summary = pd.read_sql_query("""
    SELECT VendorNumber, SUM(Freight) AS FreightCost
    FROM Vendor_invoice
    GROUP BY VendorNumber
""", conn)
freight_summary

pd.read_sql_query("""SELECT
    p.VendorNumber,
    p.VendorName,
    p.Brand,
    p.PurchasePrice,
    pd.Volume,
    pd.Price as ActualPrice,
    SUM(p.Quantity) as TotalPurchaseQuantity,
    SUM(p.Dollars) as TotalPurchaseDollars
FROM purchases p
JOIN purchase_prices pd
ON p.Brand = pd.Brand
where p.PurchasePrice>0
GROUP BY p.VendorNumber, p.VendorName, p.Brand
ORDER BY TotalPurchaseDollars DESC""", conn)

pd.read_sql_query("""SELECT
    VendorNo,
    Brand,
    SUM(SalesDollars) as TotalSalesDollars,
    SUM(SalesPrice) as TotalSalesPrice,
    SUM(SalesQuantity) as TotalSalesQuantity,
    SUM(ExciseTax) as TotalExciseTax
FROM sales
GROUP BY VendorNo, Brand""", conn)

import pandas as pd
import sqlite3
conn=sqlite3.connect('inventory.db')
vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
    SELECT
        VendorNumber,
        SUM(Freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
),

PurchaseSummary AS (
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
),

SalesSummary AS (
    SELECT
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars) AS TotalSalesDollars,
        SUM(SalesPrice) AS TotalSalesPrice,
        SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
)

SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC""", conn)
print(vendor_sales_summary)
vendor_sales_summary.dtypes
vendor_sales_summary.isnull().sum()
vendor_sales_summary['Volume']=vendor_sales_summary['Volume'].astype('float64')
vendor_sales_summary.fillna(0,inplace=True)
vendor_sales_summary['VendorName']=vendor_sales_summary['VendorName'].str.strip()

vendor_sales_summary['GrossProfit'] = (
    vendor_sales_summary['TotalSalesDollars']
    - vendor_sales_summary['TotalPurchaseDollars']
)

vendor_sales_summary['GrossProfit'].min()

vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']

vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars']) * 100

vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']

vendor_sales_summary['SalestoPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']


cursor=conn.cursor()
cursor.execute("""CREATE TABLE vendor_sales_summary (
    VendorNumber INT,
    VendorName VARCHAR(100),
    Brand INT,
    Description VARCHAR(100),
    PurchasePrice DECIMAL(10,2),
    ActualPrice DECIMAL(10,2),
    Volume,
    TotalPurchaseQuantity INT,
    TotalPurchaseDollars DECIMAL(15,2),
    TotalSalesQuantity INT,
    TotalSalesDollars DECIMAL(15,2),
    TotalSalesPrice DECIMAL(15,2),
    TotalExciseTax DECIMAL(15,2),
    FreightCost DECIMAL(15,2),
    GrossProfit DECIMAL(15,2),
    ProfitMargin DECIMAL(15,2),
    StockTurnover DECIMAL(15,2),
    SalesToPurchaseRatio DECIMAL(15,2),
    PRIMARY KEY (VendorNumber, Brand)
)""")


pd.read_sql_query("select * from vendor_sales_summary", conn)

vendor_sales_summary.to_sql('vendor_sales_summary', conn, if_exists='replace',index=False)

import sqlite3
import pandas as pd
import logging

# Logging configuration
logging.basicConfig(
    filename="logs/get_vendor_summary.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# -----------------------------
# Helper: Ingest dataframe into SQLite
# -----------------------------
def ingest_db(df, table_name, conn):
    df.to_sql(table_name, con=conn, if_exists='replace', index=False)
    logging.info(f"{table_name} ingested into SQLite, shape={df.shape}")

# -----------------------------
# Create vendor summary
# -----------------------------
def create_vendor_summary(conn):
    """
    Merge the different tables to get the overall vendor summary.
    """
    vendor_sales_summary = pd.read_sql_query("""
        WITH FreightSummary AS (
            SELECT
                VendorNumber,
                SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber
        ),
        PurchaseSummary AS (
            SELECT
                p.VendorNumber,
                p.VendorName,
                p.Brand,
                p.Description,
                p.PurchasePrice,
                pp.Price AS ActualPrice,
                pp.Volume,
                SUM(p.Quantity) AS TotalPurchaseQuantity,
                SUM(p.Dollars) AS TotalPurchaseDollars
            FROM purchases p
            JOIN purchase_prices pp
                ON p.Brand = pp.Brand
            WHERE p.PurchasePrice > 0
            GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description,
                     p.PurchasePrice, pp.Price, pp.Volume
        ),
        SalesSummary AS (
            SELECT
                VendorNo,
                Brand,
                SUM(SalesQuantity) AS TotalSalesQuantity,
                SUM(SalesDollars) AS TotalSalesDollars,
                SUM(SalesPrice) AS TotalSalesPrice,
                SUM(ExciseTax) AS TotalExciseTax
            FROM sales
            GROUP BY VendorNo, Brand
        )
        SELECT
            ps.VendorNumber,
            ps.VendorName,
            ps.Brand,
            ps.Description,
            ps.PurchasePrice,
            ps.ActualPrice,
            ps.Volume,
            ps.TotalPurchaseQuantity,
            ps.TotalPurchaseDollars,
            ss.TotalSalesQuantity,
            ss.TotalSalesDollars,
            ss.TotalSalesPrice,
            ss.TotalExciseTax,
            fs.FreightCost
        FROM PurchaseSummary ps
        LEFT JOIN SalesSummary ss
            ON ps.VendorNumber = ss.VendorNo
           AND ps.Brand = ss.Brand
        LEFT JOIN FreightSummary fs
            ON ps.VendorNumber = fs.VendorNumber
        ORDER BY ps.TotalPurchaseDollars DESC
    """, conn)

    return vendor_sales_summary

# -----------------------------
# Clean the data
# -----------------------------
def clean_data(df):
    """Clean and enrich the vendor sales summary dataframe."""

    # Convert Volume to float
    df['Volume'] = df['Volume'].astype('float')

    # Fill missing values
    df.fillna(0, inplace=True)

    # Strip spaces from categorical columns
    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()

    # Add new calculated columns
    df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
    df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars'].replace(0, 1)) * 100
    df['StockTurnover'] = df['TotalSalesQuantity'] / df['TotalPurchaseQuantity'].replace(0, 1)
    df['SalesToPurchaseRatio'] = df['TotalSalesDollars'] / df['TotalPurchaseDollars'].replace(0, 1)

    return df

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Create database connection
    conn = sqlite3.connect('inventory.db')

    logging.info('Creating Vendor Summary Table.....')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())

    logging.info('Cleaning Data.....')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('Ingesting data.....')
    ingest_db(clean_df, 'vendor_sales_summary', conn)

    logging.info('Completed')



import pandas as pd
import numpy as np
import warnings
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats
warnings.filterwarnings('ignore')
conn = sqlite3.connect('inventory.db')
query = "SELECT * FROM vendor_sales_summary"
df = pd.read_sql_query(query, conn)
df.head()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

# plot histograms for each numerical column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)   # adjust rows/cols depending on no. of features
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

# plot histograms for each numerical column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)   # adjust rows/cols depending on no. of features
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()


df = pd.read_sql_query("""
    SELECT *
    FROM vendor_sales_summary
    WHERE GrossProfit > 0
      AND ProfitMargin > 0
      AND TotalSalesQuantity > 0
""", conn)
df

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

# plot histograms for each numerical column
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)   # adjust rows/cols depending on no. of features
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

categorial_cols = ['VendorName', 'Description']

plt.figure(figsize=(12, 5))
for i, col in enumerate(categorial_cols):
    plt.subplot(1, 2, i + 1)
    sns.countplot(
        y=df[col],
        order=df[col].value_counts().index[:10]  # fixed typo here
    )
    plt.title(f"Count Plot of {col}")

plt.tight_layout()
plt.show()  # fixed typo here


plt.figure(figsize=(12, 8))

# Correct way to compute correlation only for numerical columns
correlation_matrix = df[numerical_cols].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


brand_performance = df.groupby('Description').agg({
    'TotalSalesDollars': 'sum',
    'ProfitMargin': 'mean'
})

low_sales_threshold = brand_performance['TotalSalesDollars'].quantile(0.15)
high_margin_threshold = brand_performance['ProfitMargin'].quantile(0.85)

print("Low Sales Threshold:", low_sales_threshold)
print("High Margin Threshold:", high_margin_threshold)

target_brands = brand_performance[
    (brand_performance['TotalSalesDollars'] <= low_sales_threshold) &
    (brand_performance['ProfitMargin'] >= high_margin_threshold)
]

print("Brands with Low Sales but High Profit Margins:")
display(target_brands.sort_values('TotalSalesDollars'))

brand_performance = brand_performance[brand_performance['TotalSalesDollars'] < 10000]  # for better visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=brand_performance, x='TotalSalesDollars', y='ProfitMargin', color='blue', label='All Brands', alpha=0.7)
sns.scatterplot(data=target_brands, x='TotalSalesDollars', y='ProfitMargin', color='red', label='Target Brands')

plt.axhline(high_margin_threshold, linestyle='--', color='black', label='High Margin Threshold')
plt.axvline(low_sales_threshold, linestyle='--', color='black', label='Low Sales Threshold')

plt.xlabel('Total Sales ($)')
plt.ylabel('Profit Margin (%)')
plt.title('Brands for Promotional or Pricing Adjustments')
plt.legend()
plt.grid(True)
plt.show()

#whih vendors and brands demostrate the highest sales performane?
# Top Vendors & Brands by Sales Performance
def format_dollars(value):
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return str(value)
top_vendors = df.groupby('VendorName')['TotalSalesDollars'].sum().nlargest(10)
top_brands = df.groupby('Description')['TotalSalesDollars'].sum().nlargest(10)
top_vendors

top_brands.apply(lambda a :format_dollars(a))

plt.figure(figsize=(15, 5))

# Plot for Top Vendors
plt.subplot(1, 2, 1)
ax1 = sns.barplot(y=top_vendors.index, x=top_vendors.values, palette="Blues_r")
plt.title("Top 10 Vendors by Sales")

for bar in ax1.patches:
    ax1.text(bar.get_width() + (bar.get_width() * 0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

# Plot for Top Brands
plt.subplot(1, 2, 2)
ax2 = sns.barplot(y=top_brands.index.astype(str), x=top_brands.values, palette="Reds_r")
plt.title("Top 10 Brands by Sales")

for bar in ax2.patches:
    ax2.text(bar.get_width() + (bar.get_width() * 0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.show()

vendor_performance = df.groupby('VendorName').agg({
    'TotalPurchaseDollars': 'sum',
    'GrossProfit': 'sum',
    'TotalSalesDollars': 'sum'
}).reset_index()
vendor_performance.shape

vendor_performance['PurchaseContribution%'] = (
    vendor_performance['TotalPurchaseDollars'] / vendor_performance['TotalPurchaseDollars'].sum()*100
)
vendor_performance

vendor_performance=round(vendor_performance.sort_values('PurchaseContribution%', ascending=False))


# Display Top 10 Vendors
top_vendors = vendor_performance.head(10)
top_vendors['TotalSalesDollars'] = top_vendors['TotalSalesDollars'].apply(format_dollars)
top_vendors['TotalPurchaseDollars'] = top_vendors['TotalPurchaseDollars'].apply(format_dollars)
top_vendors['GrossProfit'] = top_vendors['GrossProfit'].apply(format_dollars)
top_vendors


top_vendors['PurchaseContribution%'].sum()

top_vendors['Cumulative_Contribution%'] = top_vendors['PurchaseContribution%'].cumsum()
top_vendors

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Purchase Contribution %
sns.barplot(
    x=top_vendors['VendorName'],
    y=top_vendors['PurchaseContribution%'],
    palette="mako", ax=ax1
)

# Add text labels on bars
for i, value in enumerate(top_vendors['PurchaseContribution%']):
    ax1.text(i, value, str(value), ha='center', fontsize=10, color='white')

# Line plot for Cumulative Contribution %
ax2 = ax1.twinx()
ax2.plot(
    top_vendors['VendorName'],
    top_vendors['Cumulative_Contribution%'],
    color='red', marker='o', linestyle='dashed', label='Cumulative Contribution'
)

# X-axis vendor names
ax1.set_xticklabels(top_vendors['VendorName'], rotation=90)

# Axis labels
ax1.set_ylabel('Purchase Contribution %', color='blue')
ax2.set_ylabel('Cumulative Contribution %', color='red')
ax1.set_xlabel('Vendors')

# Title
ax1.set_title('Pareto Chart: Vendor Contribution to Total Purchases')

# Horizontal line at 100%
ax2.axhline(100, color='gray', linestyle='dashed', alpha=0.7)

# Legend for cumulative line
ax2.legend(loc='upper right')

plt.show()


#how muh of total prourement is dependent on the top vendors?
print(f"Total Purchase Contribution of top 10 vendors is {round(top_vendors['PurchaseContribution%'].sum(),2)} %")

vendors = list(top_vendors['VendorName'].values)
purchase_contributions = list(top_vendors['PurchaseContribution%'].values)
total_contribution = sum(purchase_contributions)
remaining_contribution = 100 - total_contribution

# Append "Other Vendors" category
vendors.append("Other Vendors")
purchase_contributions.append(remaining_contribution)

# Donut Chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    purchase_contributions,
    labels=vendors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.85,
    colors=plt.cm.Paired.colors
)

# Draw a white circle in the center to create a "donut" effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Add Total Contribution annotation in the center
plt.text(0, 0, f"Top 10 Total:\n{total_contribution:.2f}%",
         fontsize=14, fontweight='bold', ha='center', va='center')


#does purhasing in bulk redue the unit prie,and what is the optimal purhasevolumn for ost savings?
df['UnitPurhasePrie']=df['TotalPurchaseDollars']/df['TotalPurchaseQuantity']
df["OrderSize"] = pd.qcut(df["TotalPurchaseQuantity"], q=3, labels=["Small", "Medium", "Large"])
df[['OrderSize','TotalPurchaseQuantity']]
df.groupby('OrderSize')[['UnitPurhasePrie']].mean()

plt.figure(figsize=(10, 6))  # set figure size

sns.boxplot(
    data=df,
    x="OrderSize",             # categorical variable (order size groups)
    y="UnitPurhasePrie",     # numerical variable (price per unit)
    palette="Set2"             # color palette
)

plt.title("Impact of Bulk Purchasing on Unit Price")
plt.xlabel("Order Size")
plt.ylabel("Average Unit Purchase Price")

plt.show()


#whih vendors have low inventory turnover,indiating eess stok and slow-moving produts?
df[df['StockTurnover']<1].groupby('VendorName')[['StockTurnover']].mean().sort_values('StockTurnover',ascending=True).head(10)

#how muh apital is loked in unsold inventory per vendor, and whih vendors ontribute the most to ot?
df["UnsoldInventoryValue"] = (df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"]) * df["PurchasePrice"]
print('Total Unsold Capital:', format_dollars(df["UnsoldInventoryValue"].sum()))


# Aggregate Capital Locked per Vendor
inventory_value_per_vendor = df.groupby("VendorName")["UnsoldInventoryValue"].sum().reset_index()

# Sort Vendors with the Highest Locked Capital
inventory_value_per_vendor = inventory_value_per_vendor.sort_values(by="UnsoldInventoryValue", ascending=False)

# Format column values as dollars
inventory_value_per_vendor["UnsoldInventoryValue"] = inventory_value_per_vendor["UnsoldInventoryValue"].apply(format_dollars)

# Display top 10 vendors
inventory_value_per_vendor.head(10)


#what is 95% onfidene intervals for profit margins of top-performing and low-performing vendors?
top_threshold = df["TotalSalesDollars"].quantile(0.75)
low_threshold = df["TotalSalesDollars"].quantile(0.25)

top_vendors = df[df["TotalSalesDollars"] >= top_threshold]["ProfitMargin"].dropna()
low_vendors = df[df["TotalSalesDollars"] <= low_threshold]["ProfitMargin"].dropna()


def confidence_interval(data, confidence=0.95):
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))   # Standard error
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)
    margin_of_error = t_critical * std_err
    return mean_val, mean_val - margin_of_error, mean_val + margin_of_error


top_mean, top_lower, top_upper = confidence_interval(top_vendors)
low_mean, low_lower, low_upper = confidence_interval(low_vendors)

print(f"Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")
plt.figure(figsize=(12, 6))

# Top Vendors Plot
sns.histplot(top_vendors, kde=True, color="blue", bins=30, alpha=0.5, label="Top Vendors")
plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2f}")
plt.axvline(top_mean, color="blue", linestyle="-", label=f"Top Mean: {top_mean:.2f}")

# Low Vendors Plot
sns.histplot(low_vendors, kde=True, color="red", bins=30, alpha=0.5, label="Low Vendors")
plt.axvline(low_lower, color="red", linestyle="--", label=f"Low Lower: {low_lower:.2f}")
plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
plt.axvline(low_mean, color="red", linestyle="-", label=f"Low Mean: {low_mean:.2f}")

plt.title("Profit Margin Distribution: Top vs Low Vendors")
plt.legend()
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")

plt.grid(True)
plt.show()


#is there a signifiant differene in profit margins between top-performing and low-performing vendors?
top_threshold = df["TotalSalesDollars"].quantile(0.75)
low_threshold = df["TotalSalesDollars"].quantile(0.25)

top_vendors = df[df["TotalSalesDollars"] >= top_threshold]["ProfitMargin"].dropna()
low_vendors = df[df["TotalSalesDollars"] <= low_threshold]["ProfitMargin"].dropna()

# Perform Two-Sample T-Test
t_stat, p_value = ttest_ind(top_vendors, low_vendors, equal_var=False)

# Print results
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("Reject H₀: There is a significant difference in profit margins between top and low-performing vendors.")
else:
    print("Fail to Reject H₀: No significant difference in profit margins.")

# ⚠️ IMPORTANT: Replace this with your NEW GitHub token (not the old one you pasted before!)
token = "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"

# 1. Remove mistakenly copied folder
# !rm -rf Vendor-Performance-Data-Analytics-End-To-End-Project-SQL-Python

# 2. Create README.md
# !echo "# Vendor-Performance-Data-Analytics-End-To-End-Project-SQL-Python" > README.md

# 3. Stage and commit
# !git add .
# !git commit -m "Initial commit with README"

# 4. Ensure branch is main
# !git branch -M main

# 5. Reset the remote and authenticate with token
# !git remote remove origin
# !git remote add origin https://Piu154:{token}@github.com/Piu154/Vendor-Performance-Data-Analytics-End-To-End-Project-SQL-Python.git

# 6. Set Git identity
# !git config --global user.name "Piu154"
# !git config --global user.email "samantapriya154@gmail.com"

# 7. Push to GitHub
# !git push -u origin main




# 1. Stage the notebook
# !git add Untitled1.ipynb

# 2. Commit the notebook with a message
# !git commit -m "Add notebook Untitled1.ipynb with full code"

# 3. Push to GitHub
# !git push origin main


from google.colab import drive
drive.mount('/content/drive')
