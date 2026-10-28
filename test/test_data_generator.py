"""
Unit tests for the SyntheticDataGenerator class.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test suite for SyntheticDataGenerator."""

    @classmethod
    def setUpClass(cls):
        """Set up generator with a small deterministic test scale."""
        cls.test_num_customers = 500
        cls.test_num_products = 100
        cls.test_num_orders = 2500
        cls.seed = 42

        cls.generator = SyntheticDataGenerator(
            num_customers=cls.test_num_customers,
            num_products=cls.test_num_products,
            num_orders=cls.test_num_orders,
            seed=cls.seed
        )

        cls.customers_df = cls.generator.generate_customers(cls.test_num_customers)
        cls.products_df = cls.generator.generate_products(cls.test_num_products)
        cls.orders_df = cls.generator.generate_orders(
            count=cls.test_num_orders,
            customers_df=cls.customers_df,
            products_df=cls.products_df
        )

    def test_default_initialization(self):
        """Verify default parameters are 100K customers, 10K products, 1M orders."""
        default_gen = SyntheticDataGenerator()
        self.assertEqual(default_gen.num_customers, 100_000)
        self.assertEqual(default_gen.num_products, 10_000)
        self.assertEqual(default_gen.num_orders, 1_000_000)

    def test_customer_schema_and_distribution(self):
        """Verify customer DataFrame schema and normal distribution around 35."""
        df = self.customers_df
        self.assertEqual(len(df), self.test_num_customers)

        # Check required columns
        expected_cols = {
            "customer_id", "name", "email", "age",
            "city", "country", "registration_date"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

        # Unique customer_ids
        self.assertEqual(df["customer_id"].nunique(), self.test_num_customers)

        # Age distribution around 35
        mean_age = df["age"].mean()
        self.assertAlmostEqual(mean_age, 35.0, delta=2.5)
        self.assertTrue((df["age"] >= 18).all())
        self.assertTrue((df["age"] <= 80).all())

        # Email and name integrity
        self.assertTrue(df["email"].str.contains("@").all())
        self.assertTrue(df["name"].str.len().gt(2).all())

    def test_product_schema_and_constraints(self):
        """Verify product catalog schema, categories, prices, and ratings."""
        df = self.products_df
        self.assertEqual(len(df), self.test_num_products)

        # Check required columns
        expected_cols = {
            "product_id", "name", "category", "price", "stock", "rating"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

        # Categories: Electronics, Clothing, Home, Sports, Books
        valid_categories = {"Electronics", "Clothing", "Home", "Sports", "Books"}
        self.assertTrue(set(df["category"].unique()).issubset(valid_categories))

        # Price range: 10 - 500
        self.assertTrue((df["price"] >= 10.0).all())
        self.assertTrue((df["price"] <= 500.0).all())

        # Rating range: 1 - 5
        self.assertTrue((df["rating"] >= 1.0).all())
        self.assertTrue((df["rating"] <= 5.0).all())

        # Stock range: non-negative integer
        self.assertTrue((df["stock"] >= 0).all())

    def test_orders_schema_and_pareto_distribution(self):
        """Verify order schema, quantities [1-10], and Pareto 80/20 distribution."""
        df = self.orders_df
        self.assertEqual(len(df), self.test_num_orders)

        # Check required columns
        expected_cols = {
            "order_id", "customer_id", "product_id", "quantity", "order_date"
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

        # Quantities in [1, 10]
        self.assertTrue((df["quantity"] >= 1).all())
        self.assertTrue((df["quantity"] <= 10).all())

        # Referential integrity
        valid_cust_ids = set(self.customers_df["customer_id"])
        self.assertTrue(df["customer_id"].isin(valid_cust_ids).all())

        valid_prod_ids = set(self.products_df["product_id"])
        self.assertTrue(df["product_id"].isin(valid_prod_ids).all())

        # Pareto 80/20 verification:
        # 20% of customers should account for the vast majority of orders
        order_counts = df["customer_id"].value_counts().values
        top_20_count = int(0.20 * self.test_num_customers)
        top_20_orders = order_counts[:top_20_count].sum()
        top_20_share = (top_20_orders / len(df)) * 100.0

        # Empirical share should be ~70% to 85% for Pareto distribution
        self.assertGreater(top_20_share, 60.0)

    def test_save_data_and_generate_all(self):
        """Verify generate_all and save_data write valid CSV files to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            generator = SyntheticDataGenerator(
                num_customers=50,
                num_products=20,
                num_orders=100,
                output_dir=temp_path
            )
            datasets = generator.generate_all(save_to_disk=True)

            self.assertIn("customers", datasets)
            self.assertIn("products", datasets)
            self.assertIn("orders", datasets)

            for fname in ["customers.csv", "products.csv", "orders.csv"]:
                fpath = temp_path / fname
                self.assertTrue(fpath.exists())
                read_df = pd.read_csv(fpath)
                self.assertGreater(len(read_df), 0)


if __name__ == "__main__":
    unittest.main()
