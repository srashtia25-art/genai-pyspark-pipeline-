"""Unit tests for the file format benchmark suite."""

import unittest
from pathlib import Path

import pandas as pd

from src.benchmark import (
    DEFAULT_NUM_ROWS,
    SYSTEM_TDP_WATTS,
    FileFormatBenchmark,
    generate_benchmark_dataframe,
)


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite for file format benchmarking."""

    def test_dataframe_generation(self):
        """Verify generated benchmark DataFrame schema and record counts."""
        test_rows = 200
        df = generate_benchmark_dataframe(num_rows=test_rows)
        self.assertEqual(len(df), test_rows)

        expected_columns = {"id", "name", "email", "amount", "date", "category"}
        self.assertTrue(expected_columns.issubset(set(df.columns)))
        self.assertTrue((df["amount"] > 0).all())
        self.assertEqual(df["id"].nunique(), test_rows)

    def test_energy_calculation(self):
        """Verify energy calculation formula: (CPU_time * TDP) / 3600."""
        benchmarker = FileFormatBenchmark(tdp_watts=65.0)
        cpu_time = 3600.0  # 1 hour of CPU time at 65W should equal 65 Wh
        wh, mwh = benchmarker._calculate_energy(cpu_time)
        self.assertAlmostEqual(wh, 65.0)
        self.assertAlmostEqual(mwh, 65000.0)

    def test_csv_benchmarking(self):
        """Verify CSV benchmark produces valid telemetry."""
        df = generate_benchmark_dataframe(num_rows=500)
        benchmarker = FileFormatBenchmark()
        metric = benchmarker.benchmark_csv(df)

        self.assertEqual(metric.format_name, "CSV")
        self.assertEqual(metric.status, "SUCCESS")
        self.assertGreater(metric.file_size_mb, 0.0)
        self.assertGreater(metric.write_time_sec, 0.0)
        self.assertGreater(metric.read_time_sec, 0.0)
        self.assertGreater(metric.cpu_time_sec, 0.0)
        self.assertGreater(metric.energy_wh, 0.0)


if __name__ == "__main__":
    unittest.main()

