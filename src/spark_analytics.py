"""
PySpark Sales Analytics Module.

Provides the SalesAnalytics class for high-performance e-commerce data
analytics using Apache PySpark with adaptive query execution, Kryo serialization,
and windowed time-series growth analysis.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

try:
    from src import config
except ImportError:
    try:
        import config
    except ImportError:
        config = None

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SalesAnalytics")


def ensure_java_environment() -> None:
    """Ensure JAVA_HOME is configured for local PySpark execution on Windows."""
    if not os.environ.get("JAVA_HOME"):
        common_jdk_paths = [
            Path(r"C:\Program Files\Java\jdk-22"),
            Path(r"C:\Program Files\Java\jdk-21"),
            Path(r"C:\Program Files\Java\jdk-17"),
            Path(r"C:\Program Files\Java\jdk-11"),
        ]
        for jdk in common_jdk_paths:
            if jdk.exists():
                os.environ["JAVA_HOME"] = str(jdk)
                bin_path = str(jdk / "bin")
                if bin_path not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
                logger.info("Auto-configured JAVA_HOME: %s", jdk)
                break


class SalesAnalytics:
    """
    SalesAnalytics engine for big data e-commerce business intelligence.

    Implements scalable analytical queries on PySpark DataFrames:
    - Customer lifetime spend and ranking.
    - Product category revenue and volume aggregations.
    - Month-over-month (MoM) revenue growth using SQL Window functions.
    """

    def __init__(self, spark: Optional[SparkSession] = None) -> None:
        """
        Initialize the SalesAnalytics instance.

        Args:
            spark: Existing SparkSession instance. If None, one is initialized on demand.
        """
        self._spark = spark

    @property
    def spark(self) -> SparkSession:
        """Retrieve the active SparkSession, creating one if not already present."""
        if self._spark is None:
            self._spark = self.create_spark_session()
        return self._spark

    @classmethod
    def create_spark_session(cls, app_name: str = "SalesAnalytics") -> SparkSession:
        """
        Configure and initialize a local PySpark session.

        Configurations applied:
        - 4GB Driver & Executor Memory
        - Adaptive Query Execution (AQE) enabled
        - Kryo Serializer enabled for high throughput serialization
        - Local mode multi-threading (local[*])
        - Optimized shuffle partitions and disabled embedded Web UI for local speed

        Args:
            app_name: Name of the Spark application.

        Returns:
            Configured SparkSession instance.
        """
        ensure_java_environment()
        logger.info("Initializing PySpark session with 4GB memory, AQE, and Kryo serialization...")

        builder = (
            SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            # Memory Configuration (4GB)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            # Adaptive Query Execution (AQE)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            # Kryo Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            # Local Execution Performance
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.default.parallelism", "8")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
        )

        spark_session = builder.getOrCreate()
        spark_session.sparkContext.setLogLevel("WARN")
        logger.info("PySpark session created successfully (Version: %s)", spark_session.version)
        return spark_session

    def load_parquet(self, path: Union[str, Path]) -> DataFrame:
        """
        Load a Parquet file or directory into a PySpark DataFrame.

        Args:
            path: String or Path pointing to the Parquet source.

        Returns:
            PySpark DataFrame containing the parquet data.
        """
        file_path_str = str(path)
        logger.info("Loading Parquet data from: %s", file_path_str)
        return self.spark.read.parquet(file_path_str)

    @staticmethod
    def top_customers_by_revenue(
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10
    ) -> DataFrame:
        """
        Join orders with products, calculate total spend per customer, and return top N.

        Calculation:
            order_line_amount = quantity * price
            total_spend = sum(order_line_amount) grouped by customer_id

        Args:
            orders_df: PySpark DataFrame containing orders (customer_id, product_id, quantity).
            products_df: PySpark DataFrame containing products (product_id, price).
            n: Number of top customers to return (default: 10).

        Returns:
            DataFrame containing customer_id, total_spend, total_units_purchased,
            and total_orders, ordered by total_spend descending.
        """
        logger.info("Computing top %d customers by revenue...", n)

        # Select minimal necessary columns from products
        products_subset = products_df.select(
            F.col("product_id"),
            F.col("price").cast("double")
        )

        # Join orders with products on product_id
        joined = orders_df.join(
            products_subset,
            on="product_id",
            how="inner"
        )

        # Aggregate total revenue and units per customer
        top_customers = (
            joined.groupBy("customer_id")
            .agg(
                F.round(F.sum(F.col("quantity") * F.col("price")), 2).alias("total_spend"),
                F.sum("quantity").alias("total_units_purchased"),
                F.countDistinct("order_id").alias("total_orders")
            )
            .orderBy(F.desc("total_spend"))
            .limit(n)
        )
        return top_customers

    @staticmethod
    def sales_by_category(
        orders_df: DataFrame,
        products_df: DataFrame
    ) -> DataFrame:
        """
        Group by product category, sum revenue and units sold.

        Calculation:
            total_revenue = sum(quantity * price)
            total_units_sold = sum(quantity)

        Args:
            orders_df: PySpark DataFrame containing orders (product_id, quantity).
            products_df: PySpark DataFrame containing products (product_id, category, price).

        Returns:
            DataFrame containing category, total_revenue, total_units_sold,
            total_orders, and avg_unit_price, sorted by total_revenue descending.
        """
        logger.info("Computing sales and units sold by product category...")

        products_subset = products_df.select(
            F.col("product_id"),
            F.col("category"),
            F.col("price").cast("double")
        )

        joined = orders_df.join(
            products_subset,
            on="product_id",
            how="inner"
        )

        category_summary = (
            joined.groupBy("category")
            .agg(
                F.round(F.sum(F.col("quantity") * F.col("price")), 2).alias("total_revenue"),
                F.sum("quantity").alias("total_units_sold"),
                F.countDistinct("order_id").alias("total_orders"),
                F.round(F.avg("price"), 2).alias("avg_product_price")
            )
            .orderBy(F.desc("total_revenue"))
        )
        return category_summary

    @staticmethod
    def monthly_trends(
        orders_df: DataFrame,
        products_df: DataFrame
    ) -> DataFrame:
        """
        Calculate month-over-month (MoM) revenue growth percentage using Window functions.

        Calculation:
            month = date_format(to_timestamp(order_date), 'yyyy-MM')
            monthly_revenue = sum(quantity * price)
            prev_month_revenue = lag(monthly_revenue, 1) over (Window.orderBy('year_month'))
            mom_growth_pct = ((monthly_revenue - prev_month_revenue) / prev_month_revenue) * 100

        Args:
            orders_df: PySpark DataFrame containing orders (order_date, product_id, quantity).
            products_df: PySpark DataFrame containing products (product_id, price).

        Returns:
            DataFrame containing year_month, total_revenue, prev_month_revenue,
            and mom_growth_pct, sorted chronologically by year_month.
        """
        logger.info("Computing monthly trends and month-over-month revenue growth percentage...")

        products_subset = products_df.select(
            F.col("product_id"),
            F.col("price").cast("double")
        )

        joined = orders_df.join(
            products_subset,
            on="product_id",
            how="inner"
        )

        # Standardize timestamp and format year-month
        with_month = (
            joined
            .withColumn("order_timestamp", F.to_timestamp("order_date"))
            .withColumn("year_month", F.date_format("order_timestamp", "yyyy-MM"))
        )

        # Monthly aggregation
        monthly_revenue = (
            with_month.groupBy("year_month")
            .agg(
                F.round(F.sum(F.col("quantity") * F.col("price")), 2).alias("total_revenue"),
                F.sum("quantity").alias("total_units_sold"),
                F.countDistinct("order_id").alias("total_orders")
            )
        )

        # Window Specification ordered chronologically
        window_spec = Window.orderBy("year_month")

        # Apply Window lag function to retrieve previous month revenue
        with_lag = monthly_revenue.withColumn(
            "prev_month_revenue",
            F.lag("total_revenue", 1).over(window_spec)
        )

        # Calculate Month-over-Month Growth Percentage
        growth_trends = (
            with_lag.withColumn(
                "mom_growth_pct",
                F.when(
                    F.col("prev_month_revenue").isNull(),
                    F.lit(None)
                ).otherwise(
                    F.round(
                        ((F.col("total_revenue") - F.col("prev_month_revenue")) / F.col("prev_month_revenue")) * 100.0,
                        2
                    )
                )
            )
            .orderBy("year_month")
        )
        return growth_trends

    def stop(self) -> None:
        """Stop the active SparkSession."""
        if self._spark is not None:
            logger.info("Stopping active SparkSession.")
            self._spark.stop()
            self._spark = None


def main():
    """CLI / Demonstration entry point for SalesAnalytics."""
    logger.info("=== Running PySpark SalesAnalytics Demonstration ===")

    # Initialize SalesAnalytics
    analytics = SalesAnalytics()
    spark = analytics.spark

    try:
        # Determine data directory
        base_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
        raw_dir = base_dir / "data" / "raw"
        row_dir = base_dir / "data" / "row"

        orders_parquet = raw_dir / "orders.parquet"
        products_parquet = raw_dir / "products.parquet"

        orders_csv = row_dir / "orders.csv"
        products_csv = row_dir / "products.csv"

        # Load Parquet or fallback to CSV
        if orders_parquet.exists() and products_parquet.exists():
            logger.info("Loading input datasets from Parquet: %s", raw_dir)
            orders_df = analytics.load_parquet(orders_parquet)
            products_df = analytics.load_parquet(products_parquet)
        elif orders_csv.exists() and products_csv.exists():
            logger.info("Parquet files not found; loading fallback CSVs from: %s", row_dir)
            orders_df = spark.read.csv(str(orders_csv), header=True, inferSchema=True)
            products_df = spark.read.csv(str(products_csv), header=True, inferSchema=True)
        else:
            logger.warning("No existing data files found. Generating in-memory test data...")
            orders_data = [
                ("ORD_001", "CUST_001", "PROD_001", 2, "2024-01-15 10:00:00"),
                ("ORD_002", "CUST_002", "PROD_002", 1, "2024-01-20 14:30:00"),
                ("ORD_003", "CUST_001", "PROD_003", 4, "2024-02-10 09:15:00"),
                ("ORD_004", "CUST_003", "PROD_001", 1, "2024-02-18 16:45:00"),
                ("ORD_005", "CUST_002", "PROD_002", 3, "2024-03-05 11:20:00"),
            ]
            products_data = [
                ("PROD_001", "Smart Wireless Headphones", "Electronics", 129.99),
                ("PROD_002", "Classic Cotton T-Shirt", "Clothing", 24.50),
                ("PROD_003", "Stainless Steel Kettle", "Home", 49.00),
            ]
            orders_df = spark.createDataFrame(orders_data, ["order_id", "customer_id", "product_id", "quantity", "order_date"])
            products_df = spark.createDataFrame(products_data, ["product_id", "name", "category", "price"])

        # 1. Top Customers by Revenue
        print("\n" + "=" * 70)
        print("TOP 10 CUSTOMERS BY REVENUE")
        print("=" * 70)
        top_cust = analytics.top_customers_by_revenue(orders_df, products_df, n=10)
        top_cust.show(truncate=False)

        # 2. Sales by Category
        print("\n" + "=" * 70)
        print("SALES AND UNITS SOLD BY PRODUCT CATEGORY")
        print("=" * 70)
        cat_sales = analytics.sales_by_category(orders_df, products_df)
        cat_sales.show(truncate=False)

        # 3. Monthly Trends with MoM Growth
        print("\n" + "=" * 70)
        print("MONTHLY REVENUE TRENDS & MONTH-OVER-MONTH GROWTH")
        print("=" * 70)
        mom_trends = analytics.monthly_trends(orders_df, products_df)
        mom_trends.show(truncate=False)

        print("\n" + "=" * 70)
        print("Analytics completed successfully!")
        print("=" * 70 + "\n")

    finally:
        analytics.stop()


if __name__ == "__main__":
    main()

