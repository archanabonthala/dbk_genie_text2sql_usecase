"""Tests for the gold layer ETL pipeline (05_load_gold).

Unit tests use synthetic temp views to validate transformation logic.
Integration tests (marked @pytest.mark.integration) validate data quality
against actual dev_silver / dev_gold tables.
"""
import pytest
from datetime import date
from pyspark.sql import SparkSession
from gold_transforms import (
    build_dim_date,
    build_dim_geography,
    build_fact_orders,
    build_fact_order_lines,
)

# ── Configuration ─────────────────────────────────────────────────────

ENV = "dev"
SILVER = f"{ENV}_silver"
GOLD = f"{ENV}_gold"
REGIONS = ["africa", "america", "asia", "europe", "middle_east"]


@pytest.fixture(scope="module")
def spark():
    # Reuse the active session when running inside a Databricks notebook/REPL;
    # fall back to a local session for CI environments.
    session = SparkSession.getActiveSession()
    if session is not None:
        return session
    return SparkSession.builder.master("local[1]").appName("test_load_gold").getOrCreate()


# ═══════════════════════════════════════════════════════════════════════
# UNIT TESTS — synthetic data, no dependency on real tables
# ═══════════════════════════════════════════════════════════════════════


class TestDimDate:
    """Validate date dimension generation logic."""

    def test_row_count_1992_to_1999(self, spark):
        """8 years (2 leap) = 2922 days."""
        df = build_dim_date(spark, "1992-01-01", "1999-12-31")
        assert df.count() == 2922

    def test_single_day_range(self, spark):
        df = build_dim_date(spark, "1995-06-15", "1995-06-15")
        assert df.count() == 1

    def test_date_key_is_yyyymmdd_int(self, spark):
        row = build_dim_date(spark, "1992-01-01", "1992-01-01").collect()[0]
        assert row.date_key == 19920101

    def test_date_key_dec_31(self, spark):
        row = build_dim_date(spark, "1999-12-31", "1999-12-31").collect()[0]
        assert row.date_key == 19991231

    def test_first_and_last_dates(self, spark):
        df = build_dim_date(spark, "1992-01-01", "1992-01-03")
        dates = sorted(row.full_date for row in df.collect())
        assert dates[0] == date(1992, 1, 1)
        assert dates[-1] == date(1992, 1, 3)

    def test_saturday_is_weekend(self, spark):
        """1992-01-04 is a Saturday."""
        row = build_dim_date(spark, "1992-01-04", "1992-01-04").collect()[0]
        assert row.is_weekend is True

    def test_sunday_is_weekend(self, spark):
        """1992-01-05 is a Sunday."""
        row = build_dim_date(spark, "1992-01-05", "1992-01-05").collect()[0]
        assert row.is_weekend is True

    def test_monday_is_weekday(self, spark):
        """1992-01-06 is a Monday."""
        row = build_dim_date(spark, "1992-01-06", "1992-01-06").collect()[0]
        assert row.is_weekend is False

    def test_no_duplicate_date_keys(self, spark):
        df = build_dim_date(spark, "1992-01-01", "1999-12-31")
        assert df.select("date_key").distinct().count() == df.count()

    def test_all_twelve_months(self, spark):
        df = build_dim_date(spark, "1992-01-01", "1992-12-31")
        months = {r.month_num for r in df.select("month_num").distinct().collect()}
        assert months == set(range(1, 13))

    def test_four_quarters(self, spark):
        df = build_dim_date(spark, "1992-01-01", "1992-12-31")
        quarters = {r.quarter_num for r in df.select("quarter_num").distinct().collect()}
        assert quarters == {1, 2, 3, 4}

    def test_day_name_english(self, spark):
        """1992-01-01 is a Wednesday."""
        row = build_dim_date(spark, "1992-01-01", "1992-01-01").collect()[0]
        assert row.day_name == "Wednesday"

    def test_month_name_english(self, spark):
        row = build_dim_date(spark, "1992-03-15", "1992-03-15").collect()[0]
        assert row.month_name == "March"

    def test_year_column(self, spark):
        row = build_dim_date(spark, "1995-06-15", "1995-06-15").collect()[0]
        assert row.year_num == 1995

    def test_leap_year_feb_29(self, spark):
        """1992 and 1996 are leap years — Feb 29 must exist."""
        df = build_dim_date(spark, "1996-02-28", "1996-03-01")
        assert df.count() == 3  # 28, 29, 1


class TestFactOrderLines:
    """Validate net_amount and ship_date_key computation."""

    @pytest.fixture(autouse=True)
    def _seed(self, spark):
        spark.createDataFrame(
            [
                (1, 1, 10, 20, 5, 100.00, 0.10, 0.05, "R", "F", date(1995, 3, 15), "AIR"),
                (1, 2, 11, 21, 3, 200.00, 0.00, 0.08, "N", "O", date(1996, 6, 20), "SHIP"),
                (2, 1, 12, 22, 1, 50.00, 1.00, 0.02, "A", "F", date(1994, 1, 1), "TRUCK"),
                (3, 1, 13, 23, 10, 999.99, 0.05, 0.07, "N", "O", date(1998, 12, 31), "RAIL"),
            ],
            ["order_id", "line_number", "product_id", "supplier_id", "quantity",
             "extended_price", "discount_pct", "tax_pct", "return_flag", "line_status",
             "ship_date", "ship_mode"],
        ).createOrReplaceTempView("test_order_lines")

    def test_net_amount_with_discount(self, spark):
        """100 * (1 - 0.10) = 90.00."""
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 1 AND line_number = 1"
        ).collect()[0]
        assert float(row.net_amount) == 90.00

    def test_net_amount_zero_discount(self, spark):
        """200 * (1 - 0.00) = 200.00."""
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 1 AND line_number = 2"
        ).collect()[0]
        assert float(row.net_amount) == 200.00

    def test_net_amount_full_discount(self, spark):
        """50 * (1 - 1.00) = 0.00."""
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 2 AND line_number = 1"
        ).collect()[0]
        assert float(row.net_amount) == 0.00

    def test_net_amount_rounding(self, spark):
        """999.99 * (1 - 0.05) = 949.9905 → rounds to 949.99."""
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 3 AND line_number = 1"
        ).collect()[0]
        assert float(row.net_amount) == 949.99

    def test_ship_date_key_format(self, spark):
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 1 AND line_number = 1"
        ).collect()[0]
        assert row.ship_date_key == 19950315

    def test_ship_date_key_year_end(self, spark):
        row = build_fact_order_lines(spark, "test_order_lines").filter(
            "order_id = 3 AND line_number = 1"
        ).collect()[0]
        assert row.ship_date_key == 19981231

    def test_row_count_preserved(self, spark):
        assert build_fact_order_lines(spark, "test_order_lines").count() == 4

    def test_expected_columns(self, spark):
        cols = set(build_fact_order_lines(spark, "test_order_lines").columns)
        expected = {
            "order_id", "line_number", "product_id", "supplier_id", "quantity",
            "extended_price", "discount_pct", "net_amount", "tax_pct",
            "return_flag", "line_status", "ship_date", "ship_date_key", "ship_mode",
        }
        assert cols == expected


class TestFactOrders:
    """Validate line_count aggregation and order_date_key."""

    @pytest.fixture(autouse=True)
    def _seed(self, spark):
        spark.createDataFrame(
            [
                (1, 100, date(1995, 6, 15), "F", "1-URGENT", "Clerk#001", 25000.00),
                (2, 101, date(1996, 3, 1), "O", "5-LOW", "Clerk#002", 15000.00),
                (3, 102, date(1994, 12, 31), "P", "3-MEDIUM", "Clerk#003", 500.00),
            ],
            ["order_id", "customer_id", "order_date", "order_status",
             "order_priority", "clerk", "total_price"],
        ).createOrReplaceTempView("test_orders")

        spark.createDataFrame(
            [(1, 1), (1, 2), (1, 3), (2, 1)],
            ["order_id", "line_number"],
        ).createOrReplaceTempView("test_lines_agg")

    def test_line_count_multiple(self, spark):
        row = build_fact_orders(spark, "test_orders", "test_lines_agg").filter(
            "order_id = 1"
        ).collect()[0]
        assert row.line_count == 3

    def test_line_count_single(self, spark):
        row = build_fact_orders(spark, "test_orders", "test_lines_agg").filter(
            "order_id = 2"
        ).collect()[0]
        assert row.line_count == 1

    def test_line_count_zero_when_no_lines(self, spark):
        """COALESCE should yield 0 for orders with no matching lines."""
        row = build_fact_orders(spark, "test_orders", "test_lines_agg").filter(
            "order_id = 3"
        ).collect()[0]
        assert row.line_count == 0

    def test_order_date_key(self, spark):
        row = build_fact_orders(spark, "test_orders", "test_lines_agg").filter(
            "order_id = 1"
        ).collect()[0]
        assert row.order_date_key == 19950615

    def test_all_orders_preserved(self, spark):
        assert build_fact_orders(spark, "test_orders", "test_lines_agg").count() == 3


class TestDimGeography:
    """Validate the nation-region join."""

    @pytest.fixture(autouse=True)
    def _seed(self, spark):
        spark.createDataFrame(
            [(0, "ALGERIA", 0), (1, "ARGENTINA", 1), (2, "BRAZIL", 1)],
            ["nation_id", "nation_name", "region_id"],
        ).createOrReplaceTempView("test_nations")
        spark.createDataFrame(
            [(0, "AFRICA"), (1, "AMERICA")],
            ["region_id", "region_name"],
        ).createOrReplaceTempView("test_regions")

    def test_all_nations_present(self, spark):
        assert build_dim_geography(spark, "test_nations", "test_regions").count() == 3

    def test_region_joined_correctly(self, spark):
        row = build_dim_geography(spark, "test_nations", "test_regions").filter(
            "nation_name = 'ALGERIA'"
        ).collect()[0]
        assert row.region_name == "AFRICA"

    def test_multiple_nations_per_region(self, spark):
        df = build_dim_geography(spark, "test_nations", "test_regions")
        assert df.filter("region_name = 'AMERICA'").count() == 2


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — validate actual dev_silver ↔ dev_gold tables
# Run with:  pytest test_load_gold.py -m integration
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSilverGoldRowCounts:
    """Gold tables must contain the same number of rows as their silver source."""

    def test_dim_product_count(self, spark):
        silver = spark.table(f"{SILVER}.common.products").count()
        gold = spark.table(f"{GOLD}.common.dim_product").count()
        assert gold == silver, f"dim_product: gold={gold} != silver={silver}"

    def test_dim_supplier_count(self, spark):
        silver = spark.table(f"{SILVER}.common.suppliers").count()
        gold = spark.table(f"{GOLD}.common.dim_supplier").count()
        assert gold == silver, f"dim_supplier: gold={gold} != silver={silver}"

    @pytest.mark.parametrize("region", REGIONS)
    def test_dim_customer_count(self, spark, region):
        silver = spark.table(f"{SILVER}.{region}.customers").count()
        gold = spark.table(f"{GOLD}.{region}.dim_customer").count()
        assert gold == silver, f"{region}.dim_customer: gold={gold} != silver={silver}"

    @pytest.mark.parametrize("region", REGIONS)
    def test_fact_orders_count(self, spark, region):
        silver = spark.table(f"{SILVER}.{region}.orders").count()
        gold = spark.table(f"{GOLD}.{region}.fact_orders").count()
        assert gold == silver, f"{region}.fact_orders: gold={gold} != silver={silver}"

    @pytest.mark.parametrize("region", REGIONS)
    def test_fact_order_lines_count(self, spark, region):
        silver = spark.table(f"{SILVER}.{region}.order_lines").count()
        gold = spark.table(f"{GOLD}.{region}.fact_order_lines").count()
        assert gold == silver, f"{region}.fact_order_lines: gold={gold} != silver={silver}"


@pytest.mark.integration
class TestReferentialIntegrity:
    """Every foreign key in fact tables must resolve to a dimension row."""

    def test_no_orphan_products(self, spark):
        cnt = spark.sql(f"""
            SELECT COUNT(*) AS cnt FROM {GOLD}.common.vw_order_lines_all f
            LEFT ANTI JOIN {GOLD}.common.dim_product p ON f.product_id = p.product_id
        """).collect()[0].cnt
        assert cnt == 0, f"{cnt} order lines reference missing products"

    def test_no_orphan_suppliers(self, spark):
        cnt = spark.sql(f"""
            SELECT COUNT(*) AS cnt FROM {GOLD}.common.vw_order_lines_all f
            LEFT ANTI JOIN {GOLD}.common.dim_supplier s ON f.supplier_id = s.supplier_id
        """).collect()[0].cnt
        assert cnt == 0, f"{cnt} order lines reference missing suppliers"

    def test_no_orphan_dates(self, spark):
        cnt = spark.sql(f"""
            SELECT COUNT(*) AS cnt FROM {GOLD}.common.vw_orders_all o
            LEFT ANTI JOIN {GOLD}.common.dim_date d ON o.order_date_key = d.date_key
        """).collect()[0].cnt
        assert cnt == 0, f"{cnt} orders reference missing dates"


@pytest.mark.integration
class TestDimDateCompleteness:
    """dim_date must cover the full 1992-1999 range with no gaps."""

    def test_row_count(self, spark):
        assert spark.table(f"{GOLD}.common.dim_date").count() == 2922

    def test_starts_1992(self, spark):
        d = spark.sql(
            f"SELECT MIN(full_date) AS d FROM {GOLD}.common.dim_date"
        ).collect()[0].d
        assert d == date(1992, 1, 1)

    def test_ends_1999(self, spark):
        d = spark.sql(
            f"SELECT MAX(full_date) AS d FROM {GOLD}.common.dim_date"
        ).collect()[0].d
        assert d == date(1999, 12, 31)

    def test_no_gaps(self, spark):
        """datediff(max, min) + 1 must equal row count."""
        r = spark.sql(f"""
            SELECT DATEDIFF(MAX(full_date), MIN(full_date)) + 1 AS expected,
                   COUNT(*) AS actual
            FROM {GOLD}.common.dim_date
        """).collect()[0]
        assert r.actual == r.expected, f"gap detected: {r.expected} expected, {r.actual} actual"


@pytest.mark.integration
class TestNetAmountConsistency:
    """net_amount must equal round(extended_price * (1 - discount_pct), 2) everywhere."""

    @pytest.mark.parametrize("region", REGIONS)
    def test_net_amount_formula(self, spark, region):
        mismatch = spark.sql(f"""
            SELECT COUNT(*) AS cnt
            FROM {GOLD}.{region}.fact_order_lines
            WHERE ROUND(net_amount - ROUND(extended_price * (1 - discount_pct), 2), 2) != 0
        """).collect()[0].cnt
        assert mismatch == 0, f"{region}: {mismatch} rows with wrong net_amount"
