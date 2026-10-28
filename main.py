"""
Main entry point for E-Commerce Synthetic Data Generation Pipeline.

Generates customer, product, and order datasets using SyntheticDataGenerator,
exports each dataset as a Parquet file to data/raw/, and reports detailed
execution times and file sizes with robust error handling.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Core Project Imports
from src.data_generator import SyntheticDataGenerator
from src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("PipelineMain")


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes into a human-readable string (B, KB, MB, GB).

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string representation of size.
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def print_summary_table(
    summary_data: Dict[str, Dict[str, object]],
    total_gen_time: float,
    total_save_time: float,
    output_dir: Path
) -> None:
    """
    Print an ASCII summary table detailing record counts, timings, and file sizes.

    Args:
        summary_data: Dictionary mapping dataset names to metadata.
        total_gen_time: Cumulative data generation time in seconds.
        total_save_time: Cumulative Parquet serialization time in seconds.
        output_dir: Destination directory.
    """
    total_rows = sum(d["rows"] for d in summary_data.values())
    total_bytes = sum(d["size_bytes"] for d in summary_data.values())

    print("\n" + "=" * 94)
    print("                 E-COMMERCE SYNTHETIC DATA GENERATION PIPELINE SUMMARY")
    print("=" * 94)
    print(f"{'Dataset':<12} | {'Rows':<11} | {'Cols':<5} | {'Gen Time':<10} | {'Save Time':<10} | {'File Size':<11} | {'Parquet Path'}")
    print("-" * 94)

    for name, stats in summary_data.items():
        print(
            f"{name:<12} | "
            f"{stats['rows']:>11,} | "
            f"{stats['cols']:>5} | "
            f"{stats['gen_time']:>9.2f}s | "
            f"{stats['save_time']:>9.2f}s | "
            f"{stats['size_str']:>11} | "
            f"{stats['file_path'].name}"
        )

    print("-" * 94)
    print(
        f"TOTALS       | {total_rows:>11,} | {'-':>5} | "
        f"{total_gen_time:>9.2f}s | {total_save_time:>9.2f}s | "
        f"{format_file_size(total_bytes):>11} | Destination: {output_dir}"
    )
    print("=" * 94 + "\n")


def generate_and_save_pipeline(
    num_customers: int = config.DEFAULT_NUM_CUSTOMERS,
    num_products: int = config.DEFAULT_NUM_PRODUCTS,
    num_orders: int = config.DEFAULT_NUM_ORDERS,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = config.RANDOM_SEED,
    fallback_csv: bool = False
) -> int:
    """
    Execute generation and Parquet serialization with error handling and timing metrics.

    Args:
        num_customers: Number of customers to generate (default: 100,000).
        num_products: Number of products to generate (default: 10,000).
        num_orders: Number of orders to generate (default: 1,000,000).
        output_dir: Directory where Parquet files will be stored (default: data/raw/).
        seed: Random seed for reproducibility.

    Returns:
        Exit status code (0 for success, non-zero for failure).
    """
    if output_dir is None:
        target_dir = config.DATA_RAW_DIR
    else:
        target_dir = Path(output_dir)

    logger.info("Initializing pipeline (Target Directory: %s)", target_dir)
    logger.info(
        "Requested Scale: %s Customers | %s Products | %s Orders",
        f"{num_customers:,}",
        f"{num_products:,}",
        f"{num_orders:,}"
    )

    try:
        # Ensure target raw data directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        pipeline_start_time = time.perf_counter()

        # Step 1: Initialize Generator
        generator = SyntheticDataGenerator(
            num_customers=num_customers,
            num_products=num_products,
            num_orders=num_orders,
            seed=seed,
            output_dir=target_dir
        )

        summary: Dict[str, Dict[str, object]] = {}

        # -------------------------------------------------------------------
        # Step 2: Generate Customers
        # -------------------------------------------------------------------
        logger.info("--> [1/3] Generating Customers...")
        t_start = time.perf_counter()
        customers_df = generator.generate_customers(num_customers)
        t_gen_cust = time.perf_counter() - t_start
        logger.info("Generated %s customers in %.2fs", f"{len(customers_df):,}", t_gen_cust)

        # -------------------------------------------------------------------
        # Step 3: Generate Products
        # -------------------------------------------------------------------
        logger.info("--> [2/3] Generating Products...")
        t_start = time.perf_counter()
        products_df = generator.generate_products(num_products)
        t_gen_prod = time.perf_counter() - t_start
        logger.info("Generated %s products in %.2fs", f"{len(products_df):,}", t_gen_prod)

        # -------------------------------------------------------------------
        # Step 4: Generate Orders (Pareto Distribution)
        # -------------------------------------------------------------------
        logger.info("--> [3/3] Generating Orders (Pareto 80/20 customer distribution)...")
        t_start = time.perf_counter()
        orders_df = generator.generate_orders(
            count=num_orders,
            customers_df=customers_df,
            products_df=products_df
        )
        t_gen_ord = time.perf_counter() - t_start
        logger.info("Generated %s orders in %.2fs", f"{len(orders_df):,}", t_gen_ord)

        total_gen_time = t_gen_cust + t_gen_prod + t_gen_ord

        # -------------------------------------------------------------------
        # Step 5: Save Datasets as Parquet Files in data/raw/
        # -------------------------------------------------------------------
        datasets_to_save: Dict[str, Tuple[pd.DataFrame, str, float]] = {
            "Customers": (customers_df, config.CUSTOMERS_PARQUET, t_gen_cust),
            "Products": (products_df, config.PRODUCTS_PARQUET, t_gen_prod),
            "Orders": (orders_df, config.ORDERS_PARQUET, t_gen_ord),
        }

        total_save_time = 0.0

        for name, (df, filename, gen_duration) in datasets_to_save.items():
            parquet_path = target_dir / filename
            logger.info("Saving %s as Parquet to: %s ...", name, parquet_path)

            t_save_start = time.perf_counter()
            actual_saved_path = parquet_path
            try:
                # Save as Parquet file
                df.to_parquet(parquet_path, index=False)
            except ImportError as import_err:
                logger.error(
                    "Missing Parquet serialization engine ('pyarrow' or 'fastparquet'). "
                    "Please install pyarrow via: pip install pyarrow"
                )
                if fallback_csv:
                    csv_path = target_dir / filename.replace(".parquet", ".csv")
                    logger.warning("Saving fallback CSV format to: %s ...", csv_path)
                    df.to_csv(csv_path, index=False)
                    actual_saved_path = csv_path
                else:
                    raise import_err

            save_duration = time.perf_counter() - t_save_start
            total_save_time += save_duration

            file_size_bytes = actual_saved_path.stat().st_size
            summary[name] = {
                "rows": len(df),
                "cols": len(df.columns),
                "gen_time": gen_duration,
                "save_time": save_duration,
                "size_bytes": file_size_bytes,
                "size_str": format_file_size(file_size_bytes),
                "file_path": actual_saved_path,
            }
            logger.info(
                "Successfully saved %s (%s, %s rows) in %.2fs",
                actual_saved_path.name,
                format_file_size(file_size_bytes),
                f"{len(df):,}",
                save_duration
            )

        # Print human-readable summary
        print_summary_table(summary, total_gen_time, total_save_time, target_dir)

        total_elapsed = time.perf_counter() - pipeline_start_time
        logger.info(
            "Pipeline finished successfully in %.2fs (Generation: %.2fs, Serialization: %.2fs)",
            total_elapsed,
            total_gen_time,
            total_save_time
        )
        return 0

    except ImportError as e:
        logger.critical(
            "Dependency error during Parquet export: %s\n"
            "--> Run: pip install pyarrow\n"
            "    or: pip install -r requirements.txt",
            e
        )
        return 1

    except (PermissionError, OSError) as e:
        logger.critical("File system or permission error encountered: %s", e)
        return 2

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user (Ctrl+C). Exiting.")
        return 130

    except Exception as e:
        logger.exception("An unexpected error occurred during pipeline execution: %s", e)
        return 3


def main() -> None:
    """CLI Entry point for main.py."""
    parser = argparse.ArgumentParser(
        description="E-Commerce Synthetic Data Generation & Parquet Exporter Pipeline."
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=config.DEFAULT_NUM_CUSTOMERS,
        help=f"Number of customers to generate (default: {config.DEFAULT_NUM_CUSTOMERS:,})"
    )
    parser.add_argument(
        "--products",
        type=int,
        default=config.DEFAULT_NUM_PRODUCTS,
        help=f"Number of products to generate (default: {config.DEFAULT_NUM_PRODUCTS:,})"
    )
    parser.add_argument(
        "--orders",
        type=int,
        default=config.DEFAULT_NUM_ORDERS,
        help=f"Number of orders to generate (default: {config.DEFAULT_NUM_ORDERS:,})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.DATA_RAW_DIR),
        help=f"Output directory for Parquet files (default: {config.DATA_RAW_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Random seed for deterministic generation (default: {config.RANDOM_SEED})"
    )
    parser.add_argument(
        "--fallback-csv",
        action="store_true",
        help="Fallback to saving CSV if pyarrow/fastparquet engine is not installed"
    )

    args = parser.parse_args()

    exit_code = generate_and_save_pipeline(
        num_customers=args.customers,
        num_products=args.products,
        num_orders=args.orders,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        fallback_csv=args.fallback_csv
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
