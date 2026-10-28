"""
Performance Benchmark: Pandas vs. PySpark on 1,000,000 Rows.

Compares execution time and memory footprint between Pandas and PySpark:
1. Loads orders.parquet and products.parquet using both engines.
2. Joins on product_id and calculates revenue (quantity * price).
3. Groups by customer_id, sums revenue, and extracts top 10 customers.
4. Formats and prints a detailed side-by-side performance comparison table.
"""

import argparse
import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Core Project Imports
from src.spark_analytics import SalesAnalytics, ensure_java_environment
from src import config

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ComparePerformance")


def generate_temporary_parquet_data(
    orders_path: Path,
    products_path: Path,
    num_orders: int = 1_000_000,
    num_products: int = 10_000,
    num_customers: int = 100_000,
    seed: int = 42
) -> None:
    """
    Generate synthetic Parquet datasets for testing if they do not exist on disk.

    Args:
        orders_path: Destination path for orders.parquet.
        products_path: Destination path for products.parquet.
        num_orders: Total orders to generate (default: 1,000,000).
        num_products: Total products to generate (default: 10,000).
        num_customers: Total customers pool (default: 100,000).
        seed: Random seed for deterministic data.
    """
    logger.info("Generating synthetic 1M dataset for benchmark...")
    np.random.seed(seed)

    orders_path.parent.mkdir(parents=True, exist_ok=True)
    products_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Products
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    products_df = pd.DataFrame({
        "product_id": [f"PROD_{i + 1:05d}" for i in range(num_products)],
        "name": [f"Product Item #{i + 1}" for i in range(num_products)],
        "category": np.random.choice(categories, size=num_products),
        "price": np.random.uniform(10.0, 500.0, size=num_products).round(2),
        "stock": np.random.randint(0, 1000, size=num_products),
        "rating": np.random.uniform(1.0, 5.0, size=num_products).round(1)
    })

    # 2. Orders (Pareto customer distribution)
    alpha = np.log(5.0) / np.log(4.0)
    weights = np.random.pareto(alpha, size=num_customers) + 1.0
    weights /= weights.sum()

    sampled_cust_idx = np.random.choice(num_customers, size=num_orders, p=weights)
    sampled_prod_idx = np.random.choice(num_products, size=num_orders)

    customer_id_pool = np.array([f"CUST_{i + 1:06d}" for i in range(num_customers)])
    product_id_pool = products_df["product_id"].to_numpy()

    orders_df = pd.DataFrame({
        "order_id": [f"ORD_{i + 1:07d}" for i in range(num_orders)],
        "customer_id": customer_id_pool[sampled_cust_idx],
        "product_id": product_id_pool[sampled_prod_idx],
        "quantity": np.random.randint(1, 11, size=num_orders),
        "order_date": pd.date_range("2023-01-01", periods=num_orders, freq="s").strftime("%Y-%m-%d %H:%M:%S")
    })

    # Save to Parquet
    logger.info("Saving temporary Parquet files to: %s", orders_path.parent)
    try:
        orders_df.to_parquet(orders_path, index=False)
        products_df.to_parquet(products_path, index=False)
    except ImportError:
        # Fallback to CSV if pyarrow/fastparquet is not available
        logger.warning("Parquet engine missing; saving as CSV fallback for testing...")
        orders_csv = orders_path.with_suffix(".csv")
        products_csv = products_path.with_suffix(".csv")
        orders_df.to_csv(orders_csv, index=False)
        products_df.to_csv(products_csv, index=False)


def run_pandas_benchmark(
    orders_path: Path,
    products_path: Path
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Execute 1M rows pipeline using Pandas:
    1. Load datasets.
    2. Join on product_id & calculate revenue (quantity * price).
    3. Group by customer_id, sum revenue, and extract top 10.

    Returns:
        Tuple of (timing metrics dictionary, top 10 DataFrame).
    """
    logger.info("--> [PANDAS] Starting benchmark on 1M rows...")
    gc.collect()
    timings: Dict[str, float] = {}

    tracemalloc.start()
    total_start = time.perf_counter()

    # Step 1: Load Datasets
    t0 = time.perf_counter()
    if orders_path.exists() and orders_path.suffix == ".parquet":
        orders_df = pd.read_parquet(orders_path)
        products_df = pd.read_parquet(products_path)
    else:
        csv_orders = orders_path.with_suffix(".csv")
        csv_products = products_path.with_suffix(".csv")
        orders_df = pd.read_csv(csv_orders)
        products_df = pd.read_csv(csv_products)
    timings["load_time"] = time.perf_counter() - t0

    # Step 2: Join on product_id and calculate revenue
    t0 = time.perf_counter()
    merged_df = orders_df.merge(
        products_df[["product_id", "price"]],
        on="product_id",
        how="inner"
    )
    merged_df["revenue"] = merged_df["quantity"] * merged_df["price"]
    timings["join_calc_time"] = time.perf_counter() - t0

    # Step 3: Group by customer_id, sum revenue, get top 10
    t0 = time.perf_counter()
    top_10_df = (
        merged_df.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values(by="total_revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_10_df["total_revenue"] = top_10_df["total_revenue"].round(2)
    timings["groupby_top10_time"] = time.perf_counter() - t0

    timings["total_time"] = time.perf_counter() - total_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings["peak_memory_mb"] = peak_mem / (1024 * 1024)

    logger.info("--> [PANDAS] Finished in %.3fs (Peak Memory: %.2f MB)", timings["total_time"], timings["peak_memory_mb"])
    return timings, top_10_df


def run_pyspark_benchmark(
    orders_path: Path,
    products_path: Path
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Execute 1M rows pipeline using PySpark:
    1. Load datasets into Spark DataFrames.
    2. Join on product_id & calculate revenue (quantity * price).
    3. Group by customer_id, sum revenue, and extract top 10.

    Returns:
        Tuple of (timing metrics dictionary, top 10 DataFrame converted to pandas).
    """
    from pyspark.sql import functions as F

    logger.info("--> [PYSPARK] Initializing SparkSession...")
    gc.collect()
    timings: Dict[str, float] = {}

    t_spark_start = time.perf_counter()
    analytics = SalesAnalytics()
    spark = analytics.spark
    timings["spark_init_time"] = time.perf_counter() - t_spark_start

    tracemalloc.start()
    total_start = time.perf_counter()

    try:
        # Step 1: Load Datasets
        t0 = time.perf_counter()
        if orders_path.exists() and orders_path.suffix == ".parquet":
            orders_spark = analytics.load_parquet(orders_path)
            products_spark = analytics.load_parquet(products_path)
        else:
            orders_spark = spark.read.csv(str(orders_path.with_suffix(".csv")), header=True, inferSchema=True)
            products_spark = spark.read.csv(str(products_path.with_suffix(".csv")), header=True, inferSchema=True)

        # Force materialization to measure real file I/O time
        _ = orders_spark.count()
        _ = products_spark.count()
        timings["load_time"] = time.perf_counter() - t0

        # Step 2: Join on product_id and calculate revenue
        t0 = time.perf_counter()
        joined_spark = orders_spark.join(
            products_spark.select("product_id", F.col("price").cast("double")),
            on="product_id",
            how="inner"
        ).withColumn("revenue", F.round(F.col("quantity") * F.col("price"), 2))

        # Action to measure join execution
        _ = joined_spark.count()
        timings["join_calc_time"] = time.perf_counter() - t0

        # Step 3: Group by customer_id, sum revenue, get top 10
        t0 = time.perf_counter()
        top_10_spark = (
            joined_spark.groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(10)
        )
        # Action to trigger aggregation & collect to driver
        top_10_collected = top_10_spark.toPandas()
        timings["groupby_top10_time"] = time.perf_counter() - t0

        timings["total_time"] = time.perf_counter() - total_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timings["peak_memory_mb"] = peak_mem / (1024 * 1024)

        logger.info("--> [PYSPARK] Finished in %.3fs (Driver Peak Memory: %.2f MB)", timings["total_time"], timings["peak_memory_mb"])
        return timings, top_10_collected

    finally:
        analytics.stop()


def print_performance_comparison(
    pandas_metrics: Dict[str, float],
    spark_metrics: Dict[str, float],
    pandas_top10: pd.DataFrame,
    spark_top10: pd.DataFrame
) -> None:
    """Print top 10 results and formatted side-by-side comparison tables."""

    print("\n" + "=" * 88)
    print("                    TOP 10 CUSTOMERS BY REVENUE (VERIFICATION)")
    print("=" * 88)

    # Format verification side by side
    merged_check = pandas_top10.rename(columns={"total_revenue": "pandas_rev"}).merge(
        spark_top10.rename(columns={"total_revenue": "spark_rev"}),
        on="customer_id",
        how="outer"
    )
    print(f"{'Rank':<5} | {'Customer ID':<15} | {'Pandas Revenue ($)':<22} | {'PySpark Revenue ($)':<22} | {'Match'}")
    print("-" * 88)

    for idx, row in merged_check.iterrows():
        p_val = row.get("pandas_rev", 0.0)
        s_val = row.get("spark_rev", 0.0)
        match = "✓ MATCH" if abs(p_val - s_val) < 0.05 else "≠ DIFF"
        print(f"{idx + 1:<5} | {row['customer_id']:<15} | ${p_val:>18,.2f} | ${s_val:>18,.2f} | {match}")
    print("-" * 88)

    print("\n" + "=" * 88)
    print("             PERFORMANCE COMPARISON TABLE: PANDAS vs. PYSPARK (1,000,000 ROWS)")
    print("=" * 88)
    print(f"{'Operation':<35} | {'Pandas (sec)':<14} | {'PySpark (sec)':<14} | {'Faster Engine':<14} | {'Speedup'}")
    print("-" * 88)

    comparison_steps = [
        ("1. Load Datasets", "load_time"),
        ("2. Join on product_id & Calc Rev", "join_calc_time"),
        ("3. GroupBy customer_id & Top 10", "groupby_top10_time"),
        ("TOTAL COMPUTATION TIME", "total_time"),
    ]

    for label, key in comparison_steps:
        p_time = pandas_metrics[key]
        s_time = spark_metrics[key]

        if p_time < s_time:
            faster = "Pandas"
            ratio = s_time / p_time if p_time > 0 else 1.0
        else:
            faster = "PySpark"
            ratio = p_time / s_time if s_time > 0 else 1.0

        is_total = "TOTAL" in label
        prefix = ">> " if is_total else "   "
        print(
            f"{prefix + label:<35} | "
            f"{p_time:>10.3f} s  | "
            f"{s_time:>10.3f} s  | "
            f"{faster:<14} | "
            f"{ratio:>6.2f}x"
        )

    print("-" * 88)
    print(f"{'Driver / Client Peak Memory (MB)':<35} | {pandas_metrics['peak_memory_mb']:>10.2f} MB | {spark_metrics['peak_memory_mb']:>10.2f} MB | {'-':<14} | {'-'}")
    if "spark_init_time" in spark_metrics:
        print(f"{'PySpark JVM Startup (One-Time)':<35} | {'-':>12} | {spark_metrics['spark_init_time']:>10.3f} s  | {'-':<14} | {'-'}")
    print("=" * 88)

    # Key Architectural Takeaways
    print("\n" + "=" * 88)
    print("                             ARCHITECTURAL SUMMARY")
    print("=" * 88)
    print(
        "• Single-Node Memory (1M rows): Pandas is typically faster for datasets fitting easily\n"
        "  into RAM because it executes in C/Cython without JVM inter-process overhead.\n"
        "• Distributed Scale (>100M+ rows): PySpark provides fault-tolerant out-of-core\n"
        "  distributed memory computation, multi-node clustering, and Adaptive Query Execution\n"
        "  (AQE) when data exceeds single-machine RAM.\n"
    )
    print("=" * 88 + "\n")


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Pandas vs. PySpark performance on 1,000,000 rows."
    )
    parser.add_argument(
        "--orders",
        type=str,
        default=str(config.DATA_RAW_DIR / config.ORDERS_PARQUET),
        help="Path to orders.parquet file"
    )
    parser.add_argument(
        "--products",
        type=str,
        default=str(config.DATA_RAW_DIR / config.PRODUCTS_PARQUET),
        help="Path to products.parquet file"
    )
    parser.add_argument(
        "--generate-if-missing",
        action="store_true",
        default=True,
        help="Auto-generate 1M synthetic rows if files are not found"
    )

    args = parser.parse_args()

    orders_path = Path(args.orders)
    products_path = Path(args.products)

    # Check or auto-generate if files are missing
    if not orders_path.exists() or not products_path.exists():
        if args.generate_if_missing:
            logger.info("Parquet files not found. Auto-generating 1,000,000 sample records...")
            generate_temporary_parquet_data(orders_path, products_path, num_orders=1_000_000)
        else:
            logger.error("Files not found: %s, %s", orders_path, products_path)
            sys.exit(1)

    # 1. Run Pandas Benchmark
    pandas_metrics, pandas_top10 = run_pandas_benchmark(orders_path, products_path)

    # 2. Run PySpark Benchmark
    spark_metrics, spark_top10 = run_pyspark_benchmark(orders_path, products_path)

    # 3. Print Comparison
    print_performance_comparison(
        pandas_metrics=pandas_metrics,
        spark_metrics=spark_metrics,
        pandas_top10=pandas_top10,
        spark_top10=spark_top10
    )


if __name__ == "__main__":
    main()

