import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now use absolute imports
from storage.duckdb_manager import DuckDBManager
from storage.vector_store import MetadataIndexer

import pandas as pd

db = DuckDBManager()
db.load_csv_files("/Users/shashank/Desktop/AI/Blend360Assignment/Sales Dataset")

print("=" * 80)
print("Comprehensive Amazon Sales Analysis")
print("=" * 80)

# 1. Total by different status combinations
print("\n1. DIFFERENT STATUS COMBINATIONS:")
print("-" * 80)

scenarios = {
    "All Orders": "",
    "Only Shipped": "WHERE Status = 'Shipped'",
    "Only Delivered": "WHERE Status = 'Shipped - Delivered to Buyer'",
    "Shipped + Delivered": "WHERE Status IN ('Shipped', 'Shipped - Delivered to Buyer')",
    "Exclude Cancelled": "WHERE Status != 'Cancelled'",
    "Exclude Cancelled & Returns": """WHERE Status NOT IN ('Cancelled', 'Shipped - Returned to Seller', 
                                      'Shipped - Returning to Seller', 'Shipped - Rejected by Buyer',
                                      'Shipped - Damaged', 'Shipped - Lost in Transit')""",
    "Only Completed (Delivered)": "WHERE Status = 'Shipped - Delivered to Buyer'",
}

for scenario, where_clause in scenarios.items():
    query = f"SELECT SUM(Amount) as total, COUNT(*) as count FROM amazon_sale_report {where_clause}"
    result = db.execute_query(query)
    total = result['total'].iloc[0]
    count = result['count'].iloc[0]
    if pd.notna(total):
        print(f"{scenario:35s}: ₹{total:15,.2f} ({count:,} orders)")
    else:
        print(f"{scenario:35s}: ₹           0.00 ({count:,} orders)")

# 2. By date ranges
print("\n\n2. SALES BY MONTH:")
print("-" * 80)

query_monthly = """
SELECT 
    strftime(Date, '%Y-%m') as month,
    SUM(Amount) as total,
    COUNT(*) as orders
FROM amazon_sale_report
WHERE Status IN ('Shipped', 'Shipped - Delivered to Buyer')
GROUP BY month
ORDER BY month
"""
result_monthly = db.execute_query(query_monthly)
print(result_monthly.to_string(index=False))

# 3. By category
print("\n\n3. TOP 10 CATEGORIES BY SALES:")
print("-" * 80)

query_category = """
SELECT 
    Category,
    SUM(Amount) as total,
    COUNT(*) as orders
FROM amazon_sale_report
WHERE Status IN ('Shipped', 'Shipped - Delivered to Buyer')
GROUP BY Category
ORDER BY total DESC
LIMIT 10
"""
result_category = db.execute_query(query_category)
print(result_category.to_string(index=False))

# 4. Find combinations that equal ~2.49M
print("\n\n4. SEARCHING FOR ₹2.49M COMBINATIONS:")
print("-" * 80)

target = 2492135.28
tolerance = 100000  # Within ₹100K

# Try specific months
for month in ['2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09']:
    query = f"""
    SELECT SUM(Amount) as total 
    FROM amazon_sale_report 
    WHERE strftime(Date, '%Y-%m') = '{month}'
    AND Status IN ('Shipped', 'Shipped - Delivered to Buyer')
    """
    result = db.execute_query(query)
    total = result['total'].iloc[0]
    if pd.notna(total):
        diff = abs(total - target)
        if diff < tolerance:
            print(f"✓ MATCH: {month} = ₹{total:,.2f} (diff: ₹{diff:,.2f})")
        else:
            print(f"  {month} = ₹{total:,.2f}")

# Try specific categories
print("\n\n5. CHECKING SPECIFIC CATEGORIES:")
print("-" * 80)

query_all_categories = """
SELECT 
    Category,
    SUM(Amount) as total
FROM amazon_sale_report
WHERE Status IN ('Shipped', 'Shipped - Delivered to Buyer')
GROUP BY Category
ORDER BY total DESC
"""
result_all_cat = db.execute_query(query_all_categories)

for _, row in result_all_cat.iterrows():
    total = row['total']
    if pd.notna(total):
        diff = abs(total - target)
        if diff < tolerance:
            print(f"✓ MATCH: {row['Category']} = ₹{total:,.2f} (diff: ₹{diff:,.2f})")

# 6. Try specific styles/SKUs
print("\n\n6. TOP STYLES BY SALES:")
print("-" * 80)

query_styles = """
SELECT 
    Style,
    SUM(Amount) as total,
    COUNT(*) as orders
FROM amazon_sale_report
WHERE Status IN ('Shipped', 'Shipped - Delivered to Buyer')
GROUP BY Style
ORDER BY total DESC
LIMIT 20
"""
result_styles = db.execute_query(query_styles)
print(result_styles.to_string(index=False))

# 7. Check if it's a date range
print("\n\n7. CHECKING DATE RANGES:")
print("-" * 80)

query_date_range = """
SELECT 
    MIN(Date) as first_date,
    MAX(Date) as last_date,
    COUNT(DISTINCT strftime(Date, '%Y-%m')) as num_months
FROM amazon_sale_report
"""
result_dates = db.execute_query(query_date_range)
print(result_dates.to_string(index=False))

# 8. Net sales (after deductions)
print("\n\n8. NET SALES CALCULATION:")
print("-" * 80)

query_net = """
SELECT 
    SUM(CASE WHEN Status IN ('Shipped', 'Shipped - Delivered to Buyer') THEN Amount ELSE 0 END) as revenue,
    SUM(CASE WHEN Status = 'Cancelled' THEN Amount ELSE 0 END) as cancelled,
    SUM(CASE WHEN Status LIKE '%Return%' THEN Amount ELSE 0 END) as returns,
    SUM(CASE WHEN Status IN ('Shipped', 'Shipped - Delivered to Buyer') THEN Amount ELSE 0 END) -
    SUM(CASE WHEN Status LIKE '%Return%' THEN Amount ELSE 0 END) as net_sales
FROM amazon_sale_report
"""
result_net = db.execute_query(query_net)
print(result_net.to_string(index=False))

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Target amount you mentioned: ₹{target:,.2f}")
print(f"Total all orders: ₹{78592678.30:,.2f}")
print(f"Shipped + Delivered: ₹{68975070.00:,.2f}")
print(f"\nPossible explanations:")
print("1. Specific month/date range")
print("2. Specific category or product")
print("3. After specific deductions")
print("4. Different sales report (not Amazon Sale Report.csv)")

db.close()