"""Unit tests for compare_performance module."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compare_performance import (
    generate_temporary_parquet_data,
    run_pandas_benchmark,
)


class TestComparePerformance(unittest.TestCase):
    """Test suite for Pandas vs. PySpark benchmark pipeline."""

    def test_pandas_benchmark_operations(self):
        """Verify Pandas join, calculation, and top 10 aggregation logic."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            orders_path = tmp_path / "orders.csv"
            products_path = tmp_path / "products.csv"

            # Create small mock datasets
            orders_df = pd.DataFrame({
                "order_id": ["ORD_1", "ORD_2", "ORD_3", "ORD_4"],
                "customer_id": ["C1", "C2", "C1", "C3"],
                "product_id": ["P1", "P2", "P1", "P3"],
                "quantity": [2, 1, 3, 5],
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            })
            products_df = pd.DataFrame({
                "product_id": ["P1", "P2", "P3"],
                "name": ["A", "B", "C"],
                "category": ["Cat1", "Cat2", "Cat3"],
                "price": [10.0, 20.0, 30.0],
                "stock": [100, 100, 100],
                "rating": [4.0, 4.5, 5.0]
            })

            orders_df.to_csv(orders_path, index=False)
            products_df.to_csv(products_path, index=False)

            timings, top_10 = run_pandas_benchmark(orders_path, products_path)

            self.assertIn("load_time", timings)
            self.assertIn("join_calc_time", timings)
            self.assertIn("groupby_top10_time", timings)
            self.assertIn("total_time", timings)

            self.assertEqual(len(top_10), 3)
            # C3: 5 * 30 = 150.0 (Rank 1)
            # C1: (2*10) + (3*10) = 50.0 (Rank 2)
            # C2: 1 * 20 = 20.0 (Rank 3)
            self.assertEqual(top_10.iloc[0]["customer_id"], "C3")
            self.assertEqual(top_10.iloc[0]["total_revenue"], 150.0)
            self.assertEqual(top_10.iloc[1]["customer_id"], "C1")
            self.assertEqual(top_10.iloc[1]["total_revenue"], 50.0)


if __name__ == "__main__":
    unittest.main()

