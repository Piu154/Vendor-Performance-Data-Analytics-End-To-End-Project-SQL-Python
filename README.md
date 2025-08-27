🧾🧾 Vendor-Performance-Data-Analytics-End-To-End-Project-SQL-Python– Retail Inventory & Sales

Analyzing vendor efficiency and profitability to support strategic purchasing and inventory decisions using SQL, Python, and Power BI.

📌 Table of Contents

Overview

Business Problem

Dataset

Tools & Technologies

Project Structure

Data Cleaning & Preparation

Exploratory Data Analysis (EDA)

Research Questions & Key Findings

Dashboard

How to Run This Project

Final Recommendations

Author & Contact

🔎 Overview

This project evaluates vendor performance and retail inventory dynamics to drive strategic insights for purchasing, pricing, and inventory optimization.

A complete data pipeline was built using:

SQL for ETL (data ingestion, joins, filtering, CTEs).

Python for EDA, statistical testing, and visualization.

Power BI for interactive dashboards.

🏬 Business Problem

Effective inventory and vendor management are critical in retail.
This project aims to:

Identify underperforming brands needing pricing or promotional adjustments.

Measure vendor contributions to sales and profitability.

Analyze the cost-benefit of bulk purchasing.

Investigate inventory turnover inefficiencies.

Statistically validate differences in vendor profitability.

📂 Dataset

Multiple CSV files (/data/) → sales, vendors, inventory.

Ingested into database via Python + SQL scripts.

Aggregated into vendor-level summary tables for analysis.

⚙️ Tools & Technologies

SQL → Common Table Expressions (CTEs), joins, filtering, aggregation.

Python → Pandas, NumPy, Matplotlib, Seaborn, SciPy.

GitHub → Version control, project management.

📁 Project Structure
vendor-performance-analysis/
│── README.md
│── requirements.txt
│── Vendor Performance Report.pdf
│── data/                       # Raw CSV data
│── notebooks/                  # Jupyter notebooks
│   ├── exploratory_data_analysis.ipynb
│   ├── vendor_performance_analysis.ipynb
│── scripts/                    # Python scripts
│   ├── ingestion_db.py
│   ├── get_vendor_summary.py
│── dashboard/                  # Power BI file
│   └── vendor_performance_dashboard.pbix

🧹 Data Cleaning & Preparation

Removed transactions with:

Gross Profit ≤ 0

Profit Margin ≤ 0

Sales Quantity = 0

Created vendor-level summary tables.

Handled outliers & converted data types.

Merged lookup tables for vendor analysis.

📊 Exploratory Data Analysis (EDA)

Negative or Zero Values:

Gross Profit: min -52,002.78 → loss-making sales.

Profit Margin: negative margins (sales below cost).

Unsold inventory detected (slow-moving stock).

Outliers:

Freight costs up to 257K.

Unusually large purchases.

Correlations:

Purchase Price ↔ Profit → Weak.

Purchase Qty ↔ Sales Qty → Strong (0.999).

Profit Margin ↔ Sales Price → Negative (-0.179).

❓ Research Questions & Key Findings

Brands for Promotions:
198 brands → low sales but high margins.

Top Vendors:
Top 10 vendors = 65.7% of purchases → dependency risk.

Bulk Purchasing Impact:
Up to 72% per-unit savings in large orders.

Inventory Turnover:
$2.71M unsold stock → inventory inefficiency.

Vendor Profitability:

High Vendors: Mean Margin = 31.17%

Low Vendors: Mean Margin = 41.55%

Hypothesis Testing:
Statistically significant difference in profit margins → distinct vendor strategies exist.

📊 Dashboard (Power BI)

The interactive dashboard highlights:

Vendor-wise sales & margins.

Inventory turnover & unsold stock.

Bulk purchasing impact.

Vendor performance heatmaps.

📌 File: dashboard/vendor_performance_dashboard.pbix

🚀 How to Run This Project

Clone the repo:

git clone https://github.com/Piu154/Vendor-Performance-Data-Analytics-End-To-End-Project-SQL-Python.git


Load raw CSVs & ingest into DB:

python scripts/ingestion_db.py
python scripts/get_vendor_summary.py


Run notebooks:

notebooks/exploratory_data_analysis.ipynb

notebooks/vendor_performance_analysis.ipynb



✅ Final Recommendations

Diversify vendor base → reduce over-reliance on top vendors.

Optimize bulk purchasing to capture cost savings.

Reprice & promote slow-moving, high-margin brands.

Strategically clear $2.71M unsold inventory.

Improve marketing efforts for underperforming vendors.
Improve marketing for underperforming vendors




