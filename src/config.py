"""
Configuration module for the E-Commerce PySpark Data Pipeline.

Provides paths, data generation parameters, and PySpark configuration.
"""

from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

# Raw data directories (supports both data/raw and data/row)
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_ROW_DIR = DATA_DIR / "row"

# Analytical results directories (supports both data/proceed and data/processed)
DATA_PROCEED_DIR = DATA_DIR / "proceed"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Ensure required directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCEED_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File Names
# ---------------------------------------------------------------------------
CUSTOMERS_FILENAME = "customers.csv"
PRODUCTS_FILENAME = "products.csv"
ORDERS_FILENAME = "orders.csv"
ORDER_ITEMS_FILENAME = "order_items.csv"

CUSTOMERS_PARQUET = "customers.parquet"
PRODUCTS_PARQUET = "products.parquet"
ORDERS_PARQUET = "orders.parquet"

CUSTOMERS_FILE = DATA_ROW_DIR / CUSTOMERS_FILENAME
PRODUCTS_FILE = DATA_ROW_DIR / PRODUCTS_FILENAME
ORDERS_FILE = DATA_ROW_DIR / ORDERS_FILENAME
ORDER_ITEMS_FILE = DATA_ROW_DIR / ORDER_ITEMS_FILENAME

# ---------------------------------------------------------------------------
# Data Generation Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

DEFAULT_NUM_CUSTOMERS = 100_000
DEFAULT_NUM_PRODUCTS = 10_000
DEFAULT_NUM_ORDERS = 1_000_000
MAX_ITEMS_PER_ORDER = 5

PRODUCT_CATEGORIES_LIST = [
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Books"
]

CUSTOMER_SEGMENTS = ["Bronze", "Silver", "Gold", "Platinum"]
CUSTOMER_SEGMENT_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

ORDER_STATUSES = ["Delivered", "Shipped", "Processing", "Cancelled", "Returned"]
ORDER_STATUS_WEIGHTS = [0.70, 0.12, 0.08, 0.06, 0.04]

PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "Net Banking",
    "Cash on Delivery",
    "Digital Wallet"
]

PRODUCT_CATEGORIES: Dict[str, Dict[str, float]] = {
    "Electronics": {"min_price": 49.99, "max_price": 1299.99, "margin": 0.25},
    "Apparel": {"min_price": 14.99, "max_price": 149.99, "margin": 0.40},
    "Home & Kitchen": {"min_price": 19.99, "max_price": 399.99, "margin": 0.35},
    "Beauty & Personal Care": {"min_price": 9.99, "max_price": 89.99, "margin": 0.50},
    "Books & Stationery": {"min_price": 4.99, "max_price": 49.99, "margin": 0.30},
    "Sports & Outdoors": {"min_price": 15.99, "max_price": 299.99, "margin": 0.30},
    "Toys & Games": {"min_price": 8.99, "max_price": 119.99, "margin": 0.35},
}

START_DATE = "-2y"  # Faker relative start date for historical activity
END_DATE = "now"

# ---------------------------------------------------------------------------
# PySpark Configuration & Java Setup
# ---------------------------------------------------------------------------
import os

def ensure_java_environment() -> None:
    """Ensure JAVA_HOME is set if JDK is present on standard Windows paths."""
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
                break

ensure_java_environment()

SPARK_APP_NAME = "ECommerceDataPipeline"
SPARK_MASTER = "local[*]"
SPARK_CONFIGS = {
    "spark.driver.memory": "2g",
    "spark.sql.shuffle.partitions": "4",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
    "spark.ui.enabled": "false",
}
