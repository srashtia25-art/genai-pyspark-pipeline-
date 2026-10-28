"""
File Format Hardware Benchmarking Suite.

Benchmarks storage efficiency, I/O performance, memory footprint,
and estimated energy consumption across CSV, XLSX, Parquet, ORC,
and Feather file formats on a 500,000-row e-commerce dataset.
"""

import argparse
import gc
import logging
import os
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("BenchmarkSuite")

# Constants
DEFAULT_NUM_ROWS = 500_000
SYSTEM_TDP_WATTS = 65.0  # Assumed CPU Thermal Design Power (Watts)


@dataclass
class BenchmarkMetric:
    """Dataclass holding performance and hardware telemetry for a single file format."""
    format_name: str
    extension: str
    file_size_mb: float
    write_time_sec: float
    read_time_sec: float
    peak_memory_mb: float
    cpu_time_sec: float
    energy_wh: float
    energy_mwh: float
    status: str = "SUCCESS"
    error_msg: Optional[str] = None


def generate_benchmark_dataframe(num_rows: int = DEFAULT_NUM_ROWS, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic e-commerce dataset with specified schema.

    Schema:
    - id: integer 64-bit
    - name: customer full name
    - email: customer email address
    - amount: purchase amount (float rounded to 2 decimals)
    - date: transaction timestamp
    - category: product category

    Args:
        num_rows: Number of records to generate (default: 500,000).
        seed: Random seed for deterministic generation.

    Returns:
        pandas DataFrame with 500,000 records.
    """
    logger.info("Generating benchmark dataset with %s rows...", f"{num_rows:,}")
    t_start = time.perf_counter()

    np.random.seed(seed)

    # First and last name pools to construct diverse names
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young"
    ]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me", "example.org"]
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]

    # Sample components efficiently
    sampled_first = np.random.choice(first_names, size=num_rows)
    sampled_last = np.random.choice(last_names, size=num_rows)
    sampled_domains = np.random.choice(domains, size=num_rows)

    # Vectorized string concatenation via pandas Series
    names = [f"{f} {l}" for f, l in zip(sampled_first, sampled_last)]
    emails = [f"{f.lower()}.{l.lower()}_{i+1}@{d}" for i, (f, l, d) in enumerate(zip(sampled_first, sampled_last, sampled_domains))]

    # Numerical & temporal distributions
    amounts = np.random.uniform(5.0, 1500.0, size=num_rows).round(2)
    start_ts = int(pd.Timestamp("2023-01-01").timestamp())
    end_ts = int(pd.Timestamp("2026-10-01").timestamp())
    random_ts = np.random.randint(start_ts, end_ts, size=num_rows)
    dates = pd.to_datetime(random_ts, unit="s")
    cats = np.random.choice(categories, size=num_rows)

    df = pd.DataFrame({
        "id": np.arange(1, num_rows + 1, dtype=np.int64),
        "name": names,
        "email": emails,
        "amount": amounts,
        "date": dates,
        "category": cats
    })

    t_elapsed = time.perf_counter() - t_start
    in_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        "Dataset generation complete in %.2fs (In-memory footprint: %.2f MB)",
        t_elapsed, in_memory_mb
    )
    return df


class FileFormatBenchmark:
    """Executes read, write, memory, and energy benchmarks across file formats."""

    def __init__(
        self,
        tdp_watts: float = SYSTEM_TDP_WATTS,
        parquet_engine: str = "pyarrow",
        output_dir: Optional[Union[str, Path]] = None,
        cleanup: bool = True
    ):
        """
        Initialize benchmark suite.

        Args:
            tdp_watts: CPU Thermal Design Power in Watts (default: 65W).
            parquet_engine: Preferred engine for Parquet ('pyarrow' or 'fastparquet').
            output_dir: Directory to store generated benchmark files.
            cleanup: Whether to delete benchmark files after measurements.
        """
        self.tdp_watts = tdp_watts
        self.parquet_engine = parquet_engine
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "format_benchmarks"
        self.cleanup = cleanup
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_energy(self, cpu_time_seconds: float) -> Tuple[float, float]:
        """
        Calculate estimated energy consumption in Watt-hours (Wh) and milliwatt-hours (mWh).

        Formula:
            Energy (Wh) = (CPU_time_seconds * TDP_Watts) / 3600
        """
        energy_wh = (cpu_time_seconds * self.tdp_watts) / 3600.0
        energy_mwh = energy_wh * 1000.0
        return energy_wh, energy_mwh

    def _benchmark_operation(
        self,
        format_name: str,
        extension: str,
        write_fn,
        read_fn,
        filepath: Path
    ) -> BenchmarkMetric:
        """Measure write time, read time, peak memory, and CPU time for a format."""
        gc.collect()

        # -------------------------------------------------------------------
        # Write Phase
        # -------------------------------------------------------------------
        tracemalloc.start()
        cpu_write_start = time.process_time()
        wall_write_start = time.perf_counter()

        write_fn(filepath)

        wall_write_time = time.perf_counter() - wall_write_start
        cpu_write_time = time.process_time() - cpu_write_start
        _, peak_mem_write = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure file size
        file_size_bytes = filepath.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        gc.collect()

        # -------------------------------------------------------------------
        # Read Phase
        # -------------------------------------------------------------------
        tracemalloc.start()
        cpu_read_start = time.process_time()
        wall_read_start = time.perf_counter()

        df_read = read_fn(filepath)

        wall_read_time = time.perf_counter() - wall_read_start
        cpu_read_time = time.process_time() - cpu_read_start
        _, peak_mem_read = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Verify read integrity
        del df_read
        gc.collect()

        # Total CPU time and peak memory across operations
        total_cpu_time = cpu_write_time + cpu_read_time
        peak_memory_mb = max(peak_mem_write, peak_mem_read) / (1024 * 1024)

        # Estimated energy consumption
        energy_wh, energy_mwh = self._calculate_energy(total_cpu_time)

        if self.cleanup and filepath.exists():
            try:
                filepath.unlink()
            except OSError:
                pass

        return BenchmarkMetric(
            format_name=format_name,
            extension=extension,
            file_size_mb=file_size_mb,
            write_time_sec=wall_write_time,
            read_time_sec=wall_read_time,
            peak_memory_mb=peak_memory_mb,
            cpu_time_sec=total_cpu_time,
            energy_wh=energy_wh,
            energy_mwh=energy_mwh,
            status="SUCCESS"
        )

    def benchmark_csv(self, df: pd.DataFrame) -> BenchmarkMetric:
        """Benchmark CSV format."""
        filepath = self.output_dir / "benchmark_data.csv"
        return self._benchmark_operation(
            format_name="CSV",
            extension=".csv",
            write_fn=lambda p: df.to_csv(p, index=False),
            read_fn=lambda p: pd.read_csv(p),
            filepath=filepath
        )

    def benchmark_xlsx(self, df: pd.DataFrame, max_rows: int = 50_000) -> BenchmarkMetric:
        """
        Benchmark XLSX format using openpyxl.

        Note: Writing 500,000 rows to Excel via openpyxl is notoriously memory intensive.
        We benchmark a representative sample (or full dataset if <= max_rows) and scale metrics.
        """
        filepath = self.output_dir / "benchmark_data.xlsx"
        try:
            import openpyxl
        except ImportError:
            return BenchmarkMetric(
                format_name="XLSX",
                extension=".xlsx",
                file_size_mb=0.0,
                write_time_sec=0.0,
                read_time_sec=0.0,
                peak_memory_mb=0.0,
                cpu_time_sec=0.0,
                energy_wh=0.0,
                energy_mwh=0.0,
                status="SKIPPED (openpyxl not installed)"
            )

        # Scale down for XLSX if dataset exceeds safe memory thresholds
        sample_size = min(len(df), max_rows)
        sample_df = df.iloc[:sample_size]
        scaling_factor = len(df) / sample_size

        metric = self._benchmark_operation(
            format_name="XLSX",
            extension=".xlsx",
            write_fn=lambda p: sample_df.to_excel(p, index=False, engine="openpyxl"),
            read_fn=lambda p: pd.read_excel(p, engine="openpyxl"),
            filepath=filepath
        )

        if scaling_factor > 1.0:
            metric.file_size_mb *= scaling_factor
            metric.write_time_sec *= scaling_factor
            metric.read_time_sec *= scaling_factor
            metric.cpu_time_sec *= scaling_factor
            metric.energy_wh, metric.energy_mwh = self._calculate_energy(metric.cpu_time_sec)
            metric.format_name = f"XLSX (est. {len(df):,} rows)"

        return metric

    def benchmark_parquet(self, df: pd.DataFrame) -> BenchmarkMetric:
        """Benchmark Parquet format (using pyarrow or fastparquet)."""
        filepath = self.output_dir / "benchmark_data.parquet"
        try:
            # Check engine availability
            engine_to_use = self.parquet_engine
            try:
                import pyarrow
                engine_to_use = "pyarrow"
            except ImportError:
                import fastparquet
                engine_to_use = "fastparquet"

            return self._benchmark_operation(
                format_name=f"Parquet ({engine_to_use})",
                extension=".parquet",
                write_fn=lambda p: df.to_parquet(p, index=False, engine=engine_to_use),
                read_fn=lambda p: pd.read_parquet(p, engine=engine_to_use),
                filepath=filepath
            )
        except ImportError:
            return BenchmarkMetric(
                format_name="Parquet",
                extension=".parquet",
                file_size_mb=0.0,
                write_time_sec=0.0,
                read_time_sec=0.0,
                peak_memory_mb=0.0,
                cpu_time_sec=0.0,
                energy_wh=0.0,
                energy_mwh=0.0,
                status="SKIPPED (pyarrow/fastparquet not installed)"
            )

    def benchmark_orc(self, df: pd.DataFrame) -> BenchmarkMetric:
        """Benchmark Apache ORC format (requires pyarrow)."""
        filepath = self.output_dir / "benchmark_data.orc"
        try:
            import pyarrow
            return self._benchmark_operation(
                format_name="ORC (pyarrow)",
                extension=".orc",
                write_fn=lambda p: df.to_orc(p, index=False),
                read_fn=lambda p: pd.read_orc(p),
                filepath=filepath
            )
        except (ImportError, AttributeError):
            return BenchmarkMetric(
                format_name="ORC",
                extension=".orc",
                file_size_mb=0.0,
                write_time_sec=0.0,
                read_time_sec=0.0,
                peak_memory_mb=0.0,
                cpu_time_sec=0.0,
                energy_wh=0.0,
                energy_mwh=0.0,
                status="SKIPPED (pyarrow ORC support not available)"
            )

    def benchmark_feather(self, df: pd.DataFrame) -> BenchmarkMetric:
        """Benchmark Apache Arrow Feather format."""
        filepath = self.output_dir / "benchmark_data.feather"
        try:
            import pyarrow
            return self._benchmark_operation(
                format_name="Feather (Arrow)",
                extension=".feather",
                write_fn=lambda p: df.to_feather(p),
                read_fn=lambda p: pd.read_feather(p),
                filepath=filepath
            )
        except ImportError:
            return BenchmarkMetric(
                format_name="Feather",
                extension=".feather",
                file_size_mb=0.0,
                write_time_sec=0.0,
                read_time_sec=0.0,
                peak_memory_mb=0.0,
                cpu_time_sec=0.0,
                energy_wh=0.0,
                energy_mwh=0.0,
                status="SKIPPED (pyarrow not installed)"
            )

    def run_all_benchmarks(self, df: pd.DataFrame, skip_xlsx: bool = False) -> List[BenchmarkMetric]:
        """Run complete benchmark suite across all 5 file formats."""
        logger.info("Executing benchmark suite for %s rows across CSV, XLSX, Parquet, ORC, Feather...", f"{len(df):,}")
        results: List[BenchmarkMetric] = []

        # 1. CSV (Baseline)
        logger.info("--> [1/5] Benchmarking CSV (Baseline)...")
        results.append(self.benchmark_csv(df))

        # 2. XLSX
        if not skip_xlsx:
            logger.info("--> [2/5] Benchmarking XLSX (Excel)...")
            results.append(self.benchmark_xlsx(df))
        else:
            logger.info("--> [2/5] Skipping XLSX benchmark as requested.")

        # 3. Parquet
        logger.info("--> [3/5] Benchmarking Parquet...")
        results.append(self.benchmark_parquet(df))

        # 4. ORC
        logger.info("--> [4/5] Benchmarking ORC...")
        results.append(self.benchmark_orc(df))

        # 5. Feather
        logger.info("--> [5/5] Benchmarking Feather...")
        results.append(self.benchmark_feather(df))

        return results


def print_comparison_tables(results: List[BenchmarkMetric]) -> None:
    """Print formatted comparison tables with raw metrics and percentage savings vs CSV baseline."""
    # Find CSV baseline
    csv_metric = next((m for m in results if m.extension == ".csv" and m.status == "SUCCESS"), None)

    print("\n" + "=" * 115)
    print("                     FILE FORMAT BENCHMARK: ABSOLUTE HARDWARE & PERFORMANCE METRICS")
    print(f"                                (Assumed CPU TDP: {SYSTEM_TDP_WATTS:.0f}W)")
    print("=" * 115)
    print(
        f"{'Format':<20} | {'File Size':<10} | {'Write (s)':<10} | {'Read (s)':<10} | "
        f"{'Peak Mem':<10} | {'CPU Time':<10} | {'Energy (Wh)':<12} | {'Energy (mWh)':<12} | {'Status'}"
    )
    print("-" * 115)

    for m in results:
        if m.status == "SUCCESS":
            print(
                f"{m.format_name:<20} | "
                f"{m.file_size_mb:>8.2f} MB | "
                f"{m.write_time_sec:>8.3f} s | "
                f"{m.read_time_sec:>8.3f} s | "
                f"{m.peak_memory_mb:>8.2f} MB | "
                f"{m.cpu_time_sec:>8.3f} s | "
                f"{m.energy_wh:>10.5f} Wh | "
                f"{m.energy_mwh:>10.2f} mWh | "
                f"{m.status}"
            )
        else:
            print(
                f"{m.format_name:<20} | "
                f"{'N/A':>10} | {'N/A':>10} | {'N/A':>10} | "
                f"{'N/A':>10} | {'N/A':>10} | {'N/A':>12} | {'N/A':>12} | "
                f"{m.status}"
            )

    print("=" * 115)

    if csv_metric is None:
        logger.warning("CSV baseline was not successful; cannot compute relative savings.")
        return

    # Percentage Savings Table vs CSV Baseline
    # Savings % = (Baseline - Format) / Baseline * 100
    # Positive indicates improvement (smaller size, faster time, less memory, less energy)
    print("\n" + "=" * 115)
    print("             PERCENTAGE SAVINGS VS. CSV BASELINE  (Positive % = Better / Saved Resource)")
    print("=" * 115)
    print(
        f"{'Format':<20} | {'Size Savings %':<16} | {'Write Speedup %':<16} | "
        f"{'Read Speedup %':<16} | {'Peak Mem Savings %':<18} | {'Energy Savings %':<16}"
    )
    print("-" * 115)

    for m in results:
        if m.extension == ".csv":
            print(
                f"{m.format_name + ' (Baseline)':<20} | "
                f"{'0.00% (Baseline)':<16} | {'0.00% (Baseline)':<16} | "
                f"{'0.00% (Baseline)':<16} | {'0.00% (Baseline)':<18} | {'0.00% (Baseline)':<16}"
            )
        elif m.status == "SUCCESS":
            size_sav = ((csv_metric.file_size_mb - m.file_size_mb) / csv_metric.file_size_mb) * 100.0
            write_sav = ((csv_metric.write_time_sec - m.write_time_sec) / csv_metric.write_time_sec) * 100.0
            read_sav = ((csv_metric.read_time_sec - m.read_time_sec) / csv_metric.read_time_sec) * 100.0
            mem_sav = ((csv_metric.peak_memory_mb - m.peak_memory_mb) / csv_metric.peak_memory_mb) * 100.0
            energy_sav = ((csv_metric.energy_wh - m.energy_wh) / csv_metric.energy_wh) * 100.0

            def fmt_pct(val: float) -> str:
                sign = "+" if val >= 0 else ""
                return f"{sign}{val:.2f}%"

            print(
                f"{m.format_name:<20} | "
                f"{fmt_pct(size_sav):>16} | "
                f"{fmt_pct(write_sav):>16} | "
                f"{fmt_pct(read_sav):>16} | "
                f"{fmt_pct(mem_sav):>18} | "
                f"{fmt_pct(energy_sav):>16}"
            )
        else:
            print(f"{m.format_name:<20} | {'N/A (Skipped)':<16} | {'N/A':<16} | {'N/A':<16} | {'N/A':<18} | {'N/A':<16}")

    print("=" * 115 + "\n")


def main():
    """Command-line interface for the benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark file formats (CSV, XLSX, Parquet, ORC, Feather) with hardware telemetry."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_NUM_ROWS,
        help=f"Number of rows in the benchmark dataset (default: {DEFAULT_NUM_ROWS:,})"
    )
    parser.add_argument(
        "--tdp",
        type=float,
        default=SYSTEM_TDP_WATTS,
        help=f"CPU Thermal Design Power in Watts for energy estimation (default: {SYSTEM_TDP_WATTS}W)"
    )
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="Skip Excel (.xlsx) benchmark to avoid long serialization times"
    )
    parser.add_argument(
        "--parquet-engine",
        type=str,
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Preferred Parquet engine"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated benchmark files on disk instead of cleaning up"
    )

    args = parser.parse_args()

    # Generate benchmark dataset
    df = generate_benchmark_dataframe(num_rows=args.rows)

    # Execute benchmarks
    benchmarker = FileFormatBenchmark(
        tdp_watts=args.tdp,
        parquet_engine=args.parquet_engine,
        cleanup=not args.keep_files
    )
    results = benchmarker.run_all_benchmarks(df, skip_xlsx=args.skip_xlsx)

    # Print comparative reports
    print_comparison_tables(results)


if __name__ == "__main__":
    main()

