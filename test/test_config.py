"""Unit tests for configuration module."""

import unittest
from pathlib import Path

from src import config


class TestConfig(unittest.TestCase):
    """Test configuration paths and parameter values."""

    def test_base_directories_exist(self):
        """Verify project directories are defined as valid Path instances."""
        self.assertIsInstance(config.BASE_DIR, Path)
        self.assertTrue(config.DATA_DIR.exists())
        self.assertTrue(config.DATA_ROW_DIR.exists())
        self.assertTrue(config.DATA_PROCEED_DIR.exists())

    def test_generation_parameters(self):
        """Verify default generation parameters are positive."""
        self.assertGreater(config.DEFAULT_NUM_CUSTOMERS, 0)
        self.assertGreater(config.DEFAULT_NUM_PRODUCTS, 0)
        self.assertGreater(config.DEFAULT_NUM_ORDERS, 0)
        self.assertGreater(config.MAX_ITEMS_PER_ORDER, 0)

    def test_categories_and_statuses(self):
        """Verify categories and statuses are properly populated."""
        self.assertIn("Electronics", config.PRODUCT_CATEGORIES)
        self.assertIn("Delivered", config.ORDER_STATUSES)
        self.assertIn("Gold", config.CUSTOMER_SEGMENTS)


if __name__ == "__main__":
    unittest.main()

