"""Reusable transformation functions for the gold layer ETL pipeline.
Extracted from 05_load_gold notebook.
"""
from pyspark.sql import SparkSession


def build_dim_date(spark: SparkSession, start_date: str, end_date: str):
    """Build a date dimension with date_key, day-of-week, month, quarter, year, and weekend flag."""
    return spark.sql(f"""
        SELECT CAST(date_format(d, 'yyyyMMdd') AS INT)  AS date_key,
               d                                         AS full_date,
               dayofweek(d)                              AS day_of_week,
               date_format(d, 'EEEE')                    AS day_name,
               month(d)                                  AS month_num,
               date_format(d, 'MMMM')                    AS month_name,
               quarter(d)                                AS quarter_num,
               year(d)                                   AS year_num,
               dayofweek(d) IN (1, 7)                    AS is_weekend
        FROM (SELECT explode(sequence(DATE'{start_date}', DATE'{end_date}', INTERVAL 1 DAY)) AS d)
    """)


def build_dim_geography(spark: SparkSession, nations_table: str, regions_table: str):
    """Join nations with regions to produce the geography dimension."""
    return spark.sql(f"""
        SELECT n.nation_id, n.nation_name, r.region_id, r.region_name
        FROM {nations_table} n
        JOIN {regions_table} r ON n.region_id = r.region_id
    """)


def build_fact_orders(spark: SparkSession, orders_table: str, order_lines_table: str):
    """Build fact_orders with line_count derived from order_lines (COALESCE to 0)."""
    return spark.sql(f"""
        SELECT o.order_id, o.customer_id,
               CAST(date_format(o.order_date, 'yyyyMMdd') AS INT) AS order_date_key,
               o.order_date, o.order_status, o.order_priority, o.clerk,
               o.total_price, COALESCE(l.line_count, 0) AS line_count
        FROM {orders_table} o
        LEFT JOIN (SELECT order_id, COUNT(*) AS line_count
                   FROM {order_lines_table} GROUP BY order_id) l
          ON o.order_id = l.order_id
    """)


def build_fact_order_lines(spark: SparkSession, order_lines_table: str):
    """Build fact_order_lines with computed net_amount and ship_date_key."""
    return spark.sql(f"""
        SELECT order_id, line_number, product_id, supplier_id, quantity,
               extended_price,
               discount_pct,
               ROUND(extended_price * (1 - discount_pct), 2) AS net_amount,
               tax_pct, return_flag, line_status, ship_date,
               CAST(date_format(ship_date, 'yyyyMMdd') AS INT) AS ship_date_key,
               ship_mode
        FROM {order_lines_table}
    """)


def load_table(spark: SparkSession, target: str, select_sql: str) -> int:
    """Execute INSERT OVERWRITE into target and return row count."""
    spark.sql(f"INSERT OVERWRITE {target} {select_sql}")
    return spark.table(target).count()
