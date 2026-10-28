"""Unit tests for SalesAnalytics class."""

import inspect
import unittest

from src.spark_analytics import SalesAnalytics


class TestSalesAnalyticsClass(unittest.TestCase):
    """Test suite for SalesAnalytics class interface and signature validation."""

    def test_class_methods_exist(self):
        """Verify all required methods exist on SalesAnalytics with expected signatures."""
        required_methods = [
            "create_spark_session",
            "load_parquet",
            "top_customers_by_revenue",
            "sales_by_category",
            "monthly_trends",
        ]
        for method_name in required_methods:
            self.assertTrue(
                hasattr(SalesAnalytics, method_name),
                f"Missing required method: {method_name}"
            )
            method = getattr(SalesAnalytics, method_name)
            self.assertTrue(
                callable(method),
                f"Attribute {method_name} is not callable"
            )

    def test_top_customers_signature(self):
        """Verify top_customers_by_revenue accepts orders_df, products_df, and n=10."""
        sig = inspect.signature(SalesAnalytics.top_customers_by_revenue)
        params = list(sig.parameters.keys())
        self.assertIn("orders_df", params)
        self.assertIn("products_df", params)
        self.assertIn("n", params)
        self.assertEqual(sig.parameters["n"].default, 10)

    def test_sales_by_category_signature(self):
        """Verify sales_by_category accepts orders_df and products_df."""
        sig = inspect.signature(SalesAnalytics.sales_by_category)
        params = list(sig.parameters.keys())
        self.assertIn("orders_df", params)
        self.assertIn("products_df", params)

    def test_monthly_trends_signature(self):
        """Verify monthly_trends accepts orders_df and products_df."""
        sig = inspect.signature(SalesAnalytics.monthly_trends)
        params = list(sig.parameters.keys())
        self.assertIn("orders_df", params)
        self.assertIn("products_df", params)

    def test_load_parquet_signature(self):
        """Verify load_parquet accepts path."""
        sig = inspect.signature(SalesAnalytics.load_parquet)
        params = list(sig.parameters.keys())
        self.assertIn("path", params)


if __name__ == "__main__":
    unittest.main()

