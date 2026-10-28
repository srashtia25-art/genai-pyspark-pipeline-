"""
PySpark Analytics Engine for E-Commerce Data Pipeline.

Reads synthetic data from the raw folder, applies transformations,
computes key business metrics and insights using PySpark, and outputs
results into the destination proceed folder.
"""

import logging
from pathlib import Path
from typing import Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

try:
    from src import config
except ImportError:
    import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SparkAnalyzer")


def create_spark_session(app_name: str = config.SPARK_APP_NAME) -> SparkSession:
    """Initialize and configure a local PySpark session with auto-detected JDK."""
    config.ensure_java_environment()
    builder = SparkSession.builder.appName(app_name).master(config.SPARK_MASTER)
    for key, value in config.SPARK_CONFIGS.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_raw_data(spark: SparkSession, data_dir: Path = None) -> Dict[str, DataFrame]:
    """Load raw CSV datasets into PySpark DataFrames matching the synthetic generator schema."""
    if data_dir is None:
        data_dir = config.DATA_ROW_DIR

    logger.info("Loading raw datasets from: %s", data_dir)

    customers_schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True),
        StructField("registration_date", StringType(), True),
    ])

    products_schema = StructType([
        StructField("product_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("stock", IntegerType(), True),
        StructField("rating", DoubleType(), True),
    ])

    orders_schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("order_date", StringType(), True),
    ])

    customers_df = spark.read.csv(
        str(data_dir / config.CUSTOMERS_FILENAME),
        header=True,
        schema=customers_schema
    ).withColumn("registration_timestamp", F.to_timestamp("registration_date"))

    products_df = spark.read.csv(
        str(data_dir / config.PRODUCTS_FILENAME),
        header=True,
        schema=products_schema
    )

    orders_df = spark.read.csv(
        str(data_dir / config.ORDERS_FILENAME),
        header=True,
        schema=orders_schema
    ).withColumn("order_timestamp", F.to_timestamp("order_date"))

    return {
        "customers": customers_df,
        "products": products_df,
        "orders": orders_df,
    }


def analyze_category_performance(
    products_df: DataFrame,
    orders_df: DataFrame
) -> DataFrame:
    """Compute revenue, units sold, and orders by product category."""
    joined = orders_df.join(
        products_df.select("product_id", "category", "price"),
        on="product_id",
        how="inner"
    ).withColumn("order_line_amount", F.round(F.col("quantity") * F.col("price"), 2))

    category_summary = (
        joined.groupBy("category")
        .agg(
            F.countDistinct("order_id").alias("unique_orders"),
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("order_line_amount"), 2).alias("total_revenue"),
            F.round(F.avg("price"), 2).alias("avg_product_price"),
            F.round(F.avg("quantity"), 2).alias("avg_units_per_order")
        )
        .orderBy(F.desc("total_revenue"))
    )
    return category_summary


def analyze_customer_lifetime_value(
    customers_df: DataFrame,
    orders_df: DataFrame,
    products_df: DataFrame
) -> DataFrame:
    """Analyze customer spend frequency, total monetary value, and Pareto distribution concentration."""
    orders_enriched = orders_df.join(
        products_df.select("product_id", "price"),
        on="product_id",
        how="inner"
    ).withColumn("order_line_amount", F.col("quantity") * F.col("price"))

    customer_spend = (
        orders_enriched
        .groupBy("customer_id")
        .agg(
            F.count("order_id").alias("total_orders"),
            F.sum("quantity").alias("total_items_purchased"),
            F.round(F.sum("order_line_amount"), 2).alias("lifetime_spend"),
            F.round(F.avg("order_line_amount"), 2).alias("avg_order_value"),
            F.max("order_date").alias("last_order_date")
        )
    )

    clv_df = (
        customers_df
        .join(customer_spend, on="customer_id", how="left")
        .fillna({"total_orders": 0, "total_items_purchased": 0, "lifetime_spend": 0.0, "avg_order_value": 0.0})
        .select(
            "customer_id",
            "name",
            "age",
            "city",
            "country",
            "total_orders",
            "total_items_purchased",
            "lifetime_spend",
            "avg_order_value",
            "last_order_date"
        )
        .orderBy(F.desc("lifetime_spend"))
    )
    return clv_df


def analyze_monthly_sales_trend(
    orders_df: DataFrame,
    products_df: DataFrame
) -> DataFrame:
    """Analyze monthly sales trends and order volumes over time."""
    orders_enriched = orders_df.join(
        products_df.select("product_id", "price"),
        on="product_id",
        how="inner"
    ).withColumn("order_line_amount", F.col("quantity") * F.col("price"))

    monthly_trend = (
        orders_enriched
        .withColumn("order_month", F.date_format("order_timestamp", "yyyy-MM"))
        .groupBy("order_month")
        .agg(
            F.count("order_id").alias("total_orders"),
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("order_line_amount"), 2).alias("total_revenue"),
            F.round(F.avg("order_line_amount"), 2).alias("avg_order_value")
        )
        .orderBy("order_month")
    )
    return monthly_trend


def analyze_age_group_purchases(
    customers_df: DataFrame,
    orders_df: DataFrame,
    products_df: DataFrame
) -> DataFrame:
    """Analyze purchasing behaviors across customer age demographic brackets."""
    orders_enriched = orders_df.join(
        products_df.select("product_id", "price"),
        on="product_id",
        how="inner"
    ).withColumn("order_line_amount", F.col("quantity") * F.col("price"))

    customers_with_brackets = customers_df.withColumn(
        "age_group",
        F.when(F.col("age") < 25, "18-24")
        .when(F.col("age") < 35, "25-34")
        .when(F.col("age") < 45, "35-44")
        .when(F.col("age") < 55, "45-54")
        .otherwise("55+")
    )

    age_analysis = (
        customers_with_brackets
        .join(orders_enriched, on="customer_id", how="inner")
        .groupBy("age_group")
        .agg(
            F.countDistinct("customer_id").alias("active_customers"),
            F.count("order_id").alias("order_count"),
            F.round(F.sum("order_line_amount"), 2).alias("total_spent"),
            F.round(F.avg("order_line_amount"), 2).alias("avg_order_value")
        )
        .orderBy("age_group")
    )
    return age_analysis


def save_analytics_results(results: Dict[str, DataFrame], output_dir: Path = None) -> None:
    """Save aggregated metrics as CSV and Parquet inside the proceed/processed directory."""
    if output_dir is None:
        output_dir = config.DATA_PROCEED_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing analytical insights to: %s", output_dir)

    for metric_name, df in results.items():
        csv_path = output_dir / f"{metric_name}.csv"
        parquet_path = output_dir / f"{metric_name}.parquet"

        # Export single consolidated CSV for direct inspection
        pdf = df.toPandas()
        pdf.to_csv(csv_path, index=False)
        logger.info("Saved insight CSV: %s", csv_path.name)

        # PySpark Parquet export
        df.write.mode("overwrite").parquet(str(parquet_path))
        logger.info("Saved insight Parquet: %s", parquet_path.name)


def run_pipeline(data_dir: Path = None, output_dir: Path = None) -> Dict[str, DataFrame]:
    """Execute complete PySpark data analytics pipeline."""
    spark = create_spark_session()
    try:
        datasets = load_raw_data(spark, data_dir)

        logger.info("Running business insight analytics on synthetic e-commerce data...")
        category_perf = analyze_category_performance(
            datasets["products"],
            datasets["orders"]
        )
        clv = analyze_customer_lifetime_value(
            datasets["customers"],
            datasets["orders"],
            datasets["products"]
        )
        monthly_trends = analyze_monthly_sales_trend(
            datasets["orders"],
            datasets["products"]
        )
        age_metrics = analyze_age_group_purchases(
            datasets["customers"],
            datasets["orders"],
            datasets["products"]
        )

        insights = {
            "category_performance": category_perf,
            "customer_lifetime_value": clv,
            "monthly_sales_trends": monthly_trends,
            "age_group_metrics": age_metrics,
        }

        print("\n" + "=" * 65)
        print("BUSINESS INSIGHT SUMMARY: TOP CATEGORIES BY REVENUE")
        print("=" * 65)
        category_perf.show(10, truncate=False)

        print("\n" + "=" * 65)
        print("BUSINESS INSIGHT SUMMARY: AGE GROUP DEMOGRAPHIC SPEND")
        print("=" * 65)
        age_metrics.show(10, truncate=False)

        print("\n" + "=" * 65)
        print("BUSINESS INSIGHT SUMMARY: TOP 5 CUSTOMERS BY LIFETIME SPEND")
        print("=" * 65)
        clv.show(5, truncate=False)

        save_analytics_results(insights, output_dir)
        logger.info("PySpark pipeline execution finished successfully!")
        return insights
    finally:
        spark.stop()


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
