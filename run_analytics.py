"""
Run E-Commerce Sales Analytics Pipeline.

Loads Parquet datasets from data/raw/, executes top customers by revenue,
sales by category, and monthly revenue growth trends using SalesAnalytics,
displays results, measures operation timings, and cleanly stops the Spark session.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Core Project Imports
from src.spark_analytics import SalesAnalytics
from src import config

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunAnalytics")


def run_pipeline(data_dir: Path = config.DATA_RAW_DIR, allow_csv: bool = False) -> int:
    """
    Execute the complete SalesAnalytics workflow with execution telemetry.

    Args:
        data_dir: Directory containing input datasets (default: data/raw/).
        allow_csv: If True, falls back to CSV files if Parquet files are not present.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    total_pipeline_start = time.perf_counter()

    customers_parquet = data_dir / config.CUSTOMERS_PARQUET
    products_parquet = data_dir / config.PRODUCTS_PARQUET
    orders_parquet = data_dir / config.ORDERS_PARQUET

    print("\n" + "=" * 80)
    print("           E-COMMERCE PYSPARK BIG DATA ANALYTICS PIPELINE")
    print("=" * 80)
    print(f"Target Directory: {data_dir}")
    print(f"Datasets: {config.CUSTOMERS_PARQUET}, {config.PRODUCTS_PARQUET}, {config.ORDERS_PARQUET}")
    print("=" * 80 + "\n")

    use_csv = False
    missing_files = [f.name for f in [customers_parquet, products_parquet, orders_parquet] if not f.exists()]

    if missing_files:
        if allow_csv:
            customers_csv = data_dir / config.CUSTOMERS_FILENAME
            products_csv = data_dir / config.PRODUCTS_FILENAME
            orders_csv = data_dir / config.ORDERS_FILENAME
            if customers_csv.exists() and products_csv.exists() and orders_csv.exists():
                logger.warning("Parquet files missing. Falling back to CSV format in %s ...", data_dir)
                use_csv = True
        if not use_csv:
            logger.error(
                "Missing required Parquet file(s) in %s: %s\n"
                "--> Please generate Parquet datasets first by running:\n"
                "    python main.py\n"
                "(Or use --allow-csv if CSV datasets are available).",
                data_dir,
                ", ".join(missing_files)
            )
            return 1

    # -----------------------------------------------------------------------
    # Step 1 & 2: Create Spark Session
    # -----------------------------------------------------------------------
    logger.info("--> [1/5] Initializing PySpark Session (4GB memory, AQE, Kryo)...")
    t0 = time.perf_counter()
    analytics = SalesAnalytics()
    spark = analytics.spark
    t_spark_init = time.perf_counter() - t0
    logger.info("Spark Session created successfully in %.2fs (Spark v%s)", t_spark_init, spark.version)

    try:
        # -------------------------------------------------------------------
        # Step 3: Load Datasets from data/raw/
        # -------------------------------------------------------------------
        logger.info("--> [2/5] Loading datasets from %s ...", data_dir)
        t0 = time.perf_counter()

        if not use_csv:
            customers_df = analytics.load_parquet(customers_parquet)
            products_df = analytics.load_parquet(products_parquet)
            orders_df = analytics.load_parquet(orders_parquet)
        else:
            customers_df = spark.read.csv(str(customers_csv), header=True, inferSchema=True)
            products_df = spark.read.csv(str(products_csv), header=True, inferSchema=True)
            orders_df = spark.read.csv(str(orders_csv), header=True, inferSchema=True)

        num_customers = customers_df.count()
        num_products = products_df.count()
        num_orders = orders_df.count()

        t_load = time.perf_counter() - t0
        logger.info(
            "Loaded datasets in %.2fs: %s customers | %s products | %s orders",
            t_load,
            f"{num_customers:,}",
            f"{num_products:,}",
            f"{num_orders:,}"
        )

        # -------------------------------------------------------------------
        # Step 4: Top Customers by Revenue
        # -------------------------------------------------------------------
        print("\n" + "-" * 80)
        print(" [OPERATION 1] TOP 10 CUSTOMERS BY TOTAL REVENUE")
        print("-" * 80)
        t0 = time.perf_counter()

        top_customers_df = analytics.top_customers_by_revenue(
            orders_df=orders_df,
            products_df=products_df,
            n=10
        )
        top_customers_df.show(truncate=False)

        t_top_cust = time.perf_counter() - t0
        print(f"--> Top Customers computation completed in: {t_top_cust:.3f} seconds\n")

        # -------------------------------------------------------------------
        # Step 5: Sales by Product Category
        # -------------------------------------------------------------------
        print("-" * 80)
        print(" [OPERATION 2] SALES AND UNITS SOLD BY PRODUCT CATEGORY")
        print("-" * 80)
        t0 = time.perf_counter()

        category_sales_df = analytics.sales_by_category(
            orders_df=orders_df,
            products_df=products_df
        )
        category_sales_df.show(truncate=False)

        t_cat_sales = time.perf_counter() - t0
        print(f"--> Sales by Category computation completed in: {t_cat_sales:.3f} seconds\n")

        # -------------------------------------------------------------------
        # Step 6: Monthly Revenue Trends & Month-over-Month Growth (Window Function)
        # -------------------------------------------------------------------
        print("-" * 80)
        print(" [OPERATION 3] MONTHLY REVENUE TRENDS & MONTH-OVER-MONTH (MoM) GROWTH %")
        print("-" * 80)
        t0 = time.perf_counter()

        monthly_trends_df = analytics.monthly_trends(
            orders_df=orders_df,
            products_df=products_df
        )
        monthly_trends_df.show(truncate=False)

        t_trends = time.perf_counter() - t0
        print(f"--> Monthly Trends & Growth computation completed in: {t_trends:.3f} seconds\n")

        # -------------------------------------------------------------------
        # Timing Telemetry Summary
        # -------------------------------------------------------------------
        total_time = time.perf_counter() - total_pipeline_start
        print("=" * 80)
        print("                   ANALYTICS EXECUTION TIME SUMMARY")
        print("=" * 80)
        print(f"{'Operation':<45} | {'Execution Time':<25}")
        print("-" * 80)
        print(f"{'Spark Session Initialization':<45} | {t_spark_init:>20.3f}s")
        print(f"{'Data Ingestion (3 Datasets)':<45} | {t_load:>20.3f}s")
        print(f"{'Top 10 Customers by Revenue':<45} | {t_top_cust:>20.3f}s")
        print(f"{'Sales & Units by Category':<45} | {t_cat_sales:>20.3f}s")
        print(f"{'Monthly Trends & MoM Growth % (Window)':<45} | {t_trends:>20.3f}s")
        print("-" * 80)
        print(f"{'TOTAL PIPELINE ELAPSED TIME':<45} | {total_time:>20.3f}s")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.exception("Error during analytics execution: %s", e)
        return 2

    finally:
        # -------------------------------------------------------------------
        # Step 7: Always Stop Spark Session
        # -------------------------------------------------------------------
        logger.info("Stopping Spark session...")
        analytics.stop()
        logger.info("Spark session cleanly terminated.")


def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(
        description="Execute PySpark SalesAnalytics pipeline on Parquet datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.DATA_RAW_DIR),
        help=f"Directory containing Parquet datasets (default: {config.DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--allow-csv",
        action="store_true",
        help="Fallback to CSV format if Parquet files are missing"
    )

    args = parser.parse_args()
    exit_code = run_pipeline(data_dir=Path(args.data_dir), allow_csv=args.allow_csv)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
