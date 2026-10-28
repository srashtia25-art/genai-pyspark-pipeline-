"""
Synthetic E-Commerce Data Generator.

Provides the SyntheticDataGenerator class to produce realistic, high-volume
synthetic e-commerce data (customers, products, and orders) using Faker,
NumPy statistical distributions, and tqdm progress visualization.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

try:
    from src import config
except ImportError:
    import config

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SyntheticDataGenerator")


class SyntheticDataGenerator:
    """
    Synthetic E-Commerce Data Generator.

    Generates relational datasets for customers, products, and orders adhering to:
    - Normal age distribution for customers (mean ~ 35).
    - Specified product categories with price ranges [10, 500] and ratings [1, 5].
    - Pareto 80/20 distribution for order distribution among customers.
    - Visual progress tracking via tqdm and structured logging.
    """

    # Specified categories for products
    PRODUCT_CATEGORIES: List[str] = [
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Books"
    ]

    # Category product vocabulary for realistic product names
    PRODUCT_CATALOG_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "Electronics": {
            "prefixes": ["Smart", "Wireless", "Ultra HD", "Pro Gaming", "Noise-Canceling", "Portable", "Bluetooth", "Compact"],
            "nouns": ["Headphones", "Speaker", "Monitor", "Keyboard", "Mouse", "Earbuds", "Webcam", "Power Bank", "Smartwatch", "Charger"]
        },
        "Clothing": {
            "prefixes": ["Classic Cotton", "Slim-Fit", "Waterproof", "Merino Wool", "Breathable", "Vintage Denim", "Performance", "Casual"],
            "nouns": ["T-Shirt", "Jeans", "Jacket", "Hoodie", "Sneakers", "Sweater", "Running Shorts", "Oxford Shirt", "Polo", "Track Pants"]
        },
        "Home": {
            "prefixes": ["Stainless Steel", "Ergonomic", "Memory Foam", "Ceramic", "Non-Stick", "Bamboo", "Minimalist", "Smart"],
            "nouns": ["Cookware Set", "Office Chair", "Pillow", "Coffee Maker", "Air Purifier", "Vacuum Cleaner", "Table Lamp", "Dinnerware Set"]
        },
        "Sports": {
            "prefixes": ["Non-Slip", "Adjustable", "Insulated", "Heavy-Duty", "Lightweight", "Waterproof", "Pro Series", "Aerodynamic"],
            "nouns": ["Yoga Mat", "Dumbbell Set", "Water Bottle", "Backpack", "Resistance Bands", "Cycling Helmet", "Fitness Tracker", "Trekking Poles"]
        },
        "Books": {
            "prefixes": ["The Art of", "Modern", "Advanced", "The Essential Guide to", "Mastering", "Principles of", "Handbook of", "Inside"],
            "nouns": ["Data Engineering", "Machine Learning", "Cloud Architecture", "Python Programming", "Productivity Habits", "Global Economics"]
        }
    }

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: Optional[int] = 42,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the generator with scale parameters and seed.

        Args:
            num_customers: Number of customers to generate by default (100,000).
            num_products: Number of products to generate by default (10,000).
            num_orders: Number of orders to generate by default (1,000,000).
            seed: Random seed for reproducibility.
            output_dir: Destination directory for exported CSV files.
        """
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.seed = seed

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = config.DATA_ROW_DIR

        # Initialize Faker and Random Seed
        self.fake = Faker()
        if self.seed is not None:
            Faker.seed(self.seed)
            np.random.seed(self.seed)

        logger.info(
            "SyntheticDataGenerator initialized (Defaults: %s customers, %s products, %s orders, seed=%s)",
            f"{self.num_customers:,}",
            f"{self.num_products:,}",
            f"{self.num_orders:,}",
            self.seed
        )

    def generate_customers(self, count: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic customer records.

        Fields: customer_id, name, email, age, city, country, registration_date
        - Age is sampled from a normal distribution centered around 35.

        Args:
            count: Number of customers to generate (defaults to self.num_customers).

        Returns:
            pandas DataFrame containing customer records.
        """
        total = count if count is not None else self.num_customers
        logger.info("Starting generation of %s customers...", f"{total:,}")

        # Normal distribution for age around 35 (mean=35, std=10, clipped between 18 and 80)
        raw_ages = np.random.normal(loc=35.0, scale=10.0, size=total).round().astype(int)
        ages = np.clip(raw_ages, 18, 80)

        # Generate timestamps across a 3-year historical window
        start_ts = int(pd.Timestamp("2023-01-01").timestamp())
        end_ts = int(pd.Timestamp("2026-10-01").timestamp())
        random_timestamps = np.random.randint(start_ts, end_ts, size=total)
        registration_dates = pd.to_datetime(random_timestamps, unit="s").strftime("%Y-%m-%d %H:%M:%S")

        # Batched generation for Faker strings to maintain high throughput with tqdm updates
        batch_size = 5000 if total >= 5000 else total
        names: List[str] = []
        emails: List[str] = []
        cities: List[str] = []
        countries: List[str] = []

        with tqdm(total=total, desc="Customers", unit="rec") as pbar:
            for i in range(0, total, batch_size):
                curr_batch = min(batch_size, total - i)
                batch_names = [self.fake.name() for _ in range(curr_batch)]
                batch_cities = [self.fake.city() for _ in range(curr_batch)]
                batch_countries = [self.fake.country() for _ in range(curr_batch)]
                
                # Derive realistic unique emails from name and index
                batch_emails = []
                for idx, n in enumerate(batch_names):
                    clean_name = n.lower().replace(" ", ".").replace("'", "")
                    domain = self.fake.free_email_domain()
                    batch_emails.append(f"{clean_name}_{i + idx + 1}@{domain}")

                names.extend(batch_names)
                emails.extend(batch_emails)
                cities.extend(batch_cities)
                countries.extend(batch_countries)
                pbar.update(curr_batch)

        customer_ids = [f"CUST_{i + 1:06d}" for i in range(total)]

        customers_df = pd.DataFrame({
            "customer_id": customer_ids,
            "name": names,
            "email": emails,
            "age": ages,
            "city": cities,
            "country": countries,
            "registration_date": registration_dates
        })

        logger.info(
            "Customers generated: %s rows (Age: mean=%.1f, min=%d, max=%d)",
            f"{len(customers_df):,}",
            float(ages.mean()),
            int(ages.min()),
            int(ages.max())
        )
        return customers_df

    def generate_products(self, count: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic product catalog.

        Fields: product_id, name, category, price, stock, rating
        - Categories: Electronics, Clothing, Home, Sports, Books
        - Price: Float in range [10.0, 500.0]
        - Stock: Integer in range [0, 1000]
        - Rating: Float in range [1.0, 5.0]

        Args:
            count: Number of products to generate (defaults to self.num_products).

        Returns:
            pandas DataFrame containing product catalog.
        """
        total = count if count is not None else self.num_products
        logger.info("Starting generation of %s products...", f"{total:,}")

        # Vectorized generation of numerical attributes
        categories = np.random.choice(self.PRODUCT_CATEGORIES, size=total)
        prices = np.random.uniform(10.0, 500.0, size=total).round(2)
        stocks = np.random.randint(0, 1001, size=total)
        ratings = np.random.uniform(1.0, 5.0, size=total).round(1)

        product_ids = [f"PROD_{i + 1:05d}" for i in range(total)]

        # Generate realistic product names using category vocabulary
        product_names: List[str] = []
        with tqdm(total=total, desc="Products ", unit="rec") as pbar:
            for i in range(total):
                cat = categories[i]
                vocab = self.PRODUCT_CATALOG_TEMPLATES[cat]
                prefix = np.random.choice(vocab["prefixes"])
                noun = np.random.choice(vocab["nouns"])
                model_num = (i % 900) + 100
                product_names.append(f"{prefix} {noun} #{model_num}")
                pbar.update(1)

        products_df = pd.DataFrame({
            "product_id": product_ids,
            "name": product_names,
            "category": categories,
            "price": prices,
            "stock": stocks,
            "rating": ratings
        })

        logger.info(
            "Products generated: %s rows (Avg Price: $%.2f, Avg Rating: %.1f)",
            f"{len(products_df):,}",
            float(prices.mean()),
            float(ratings.mean())
        )
        return products_df

    def generate_orders(
        self,
        count: Optional[int] = None,
        customers_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic order transactions with Pareto customer distribution.

        Fields: order_id, customer_id, product_id, quantity, order_date
        - Quantity: Integer in range [1, 10]
        - Customer distribution: Pareto distribution (alpha ~ 1.16) ensuring
          that 20% of the customer base accounts for approximately 80% of orders.

        Args:
            count: Number of orders to generate (defaults to self.num_orders).
            customers_df: Optional DataFrame of customers (generated if None).
            products_df: Optional DataFrame of products (generated if None).

        Returns:
            pandas DataFrame containing order records.
        """
        total = count if count is not None else self.num_orders

        if customers_df is None:
            logger.info("No customers DataFrame supplied; generating %s customers...", f"{self.num_customers:,}")
            customers_df = self.generate_customers(self.num_customers)

        if products_df is None:
            logger.info("No products DataFrame supplied; generating %s products...", f"{self.num_products:,}")
            products_df = self.generate_products(self.num_products)

        logger.info("Starting generation of %s orders with Pareto customer distribution...", f"{total:,}")

        num_cust = len(customers_df)
        num_prod = len(products_df)

        # -------------------------------------------------------------------
        # Pareto Distribution for Customer Orders (80/20 Rule)
        # alpha = ln(5) / ln(4) ~= 1.160964 creates an exact 80/20 Lorenz curve
        # -------------------------------------------------------------------
        alpha = np.log(5.0) / np.log(4.0)
        raw_pareto = np.random.pareto(alpha, size=num_cust) + 1.0
        customer_weights = raw_pareto / raw_pareto.sum()

        # Log Pareto distribution statistics
        sorted_weights = np.sort(customer_weights)[::-1]
        top_20_count = max(1, int(0.20 * num_cust))
        top_20_order_share = sorted_weights[:top_20_count].sum() * 100.0
        logger.info(
            "Pareto distribution configured: Top 20%% customers hold %.2f%% expected order probability weight",
            top_20_order_share
        )

        # Vectorized sampling of customer indices and product indices
        sampled_cust_indices = np.random.choice(num_cust, size=total, p=customer_weights)
        sampled_prod_indices = np.random.choice(num_prod, size=total)

        customer_id_lookup = customers_df["customer_id"].to_numpy()
        product_id_lookup = products_df["product_id"].to_numpy()

        order_customer_ids = customer_id_lookup[sampled_cust_indices]
        order_product_ids = product_id_lookup[sampled_prod_indices]

        # Quantities in [1, 10]
        quantities = np.random.randint(1, 11, size=total)

        # Dates within historical timeframe
        start_ts = int(pd.Timestamp("2023-01-01").timestamp())
        end_ts = int(pd.Timestamp("2026-10-01").timestamp())
        random_timestamps = np.random.randint(start_ts, end_ts, size=total)
        order_dates = pd.to_datetime(random_timestamps, unit="s").strftime("%Y-%m-%d %H:%M:%S")

        # Chunked generation for order IDs and progress tracking
        chunk_size = 200_000 if total >= 200_000 else total
        order_ids: List[str] = []

        with tqdm(total=total, desc="Orders   ", unit="rec") as pbar:
            for i in range(0, total, chunk_size):
                curr_size = min(chunk_size, total - i)
                order_ids.extend([f"ORD_{j + 1:07d}" for j in range(i, i + curr_size)])
                pbar.update(curr_size)

        orders_df = pd.DataFrame({
            "order_id": order_ids,
            "customer_id": order_customer_ids,
            "product_id": order_product_ids,
            "quantity": quantities,
            "order_date": order_dates
        })

        # Calculate actual empirical Pareto share in generated orders
        order_counts_per_cust = orders_df["customer_id"].value_counts().values
        emp_top_20_count = max(1, int(0.20 * num_cust))
        emp_top_20_orders = order_counts_per_cust[:emp_top_20_count].sum()
        emp_top_20_pct = (emp_top_20_orders / total) * 100.0

        logger.info(
            "Orders generated: %s rows. Empirical verification: Top 20%% customers generated %.2f%% of all orders",
            f"{len(orders_df):,}",
            emp_top_20_pct
        )
        return orders_df

    def save_data(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save generated datasets to designated directory in CSV format.

        Args:
            datasets: Dictionary containing DataFrames ('customers', 'products', 'orders').
            output_dir: Destination folder (defaults to self.output_dir).
        """
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving datasets to disk: %s", target_dir)

        file_mappings = {
            "customers": config.CUSTOMERS_FILENAME,
            "products": config.PRODUCTS_FILENAME,
            "orders": config.ORDERS_FILENAME,
        }

        for key, df in datasets.items():
            if key in file_mappings:
                file_path = target_dir / file_mappings[key]
                logger.info("Writing %s (%s rows) to %s ...", key, f"{len(df):,}", file_path.name)
                df.to_csv(file_path, index=False)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info("Successfully wrote %s (%.2f MB)", file_path.name, size_mb)

    def generate_all(
        self,
        save_to_disk: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute full end-to-end generation for customers, products, and orders.

        Args:
            save_to_disk: If True, writes resulting CSV files into output_dir.
            output_dir: Optional directory override.

        Returns:
            Dictionary containing 'customers', 'products', and 'orders' DataFrames.
        """
        logger.info(
            "=== Starting Full E-Commerce Synthetic Data Generation (%s CUST, %s PROD, %s ORD) ===",
            f"{self.num_customers:,}",
            f"{self.num_products:,}",
            f"{self.num_orders:,}"
        )

        customers_df = self.generate_customers()
        products_df = self.generate_products()
        orders_df = self.generate_orders(
            customers_df=customers_df,
            products_df=products_df
        )

        datasets = {
            "customers": customers_df,
            "products": products_df,
            "orders": orders_df,
        }

        if save_to_disk:
            self.save_data(datasets, output_dir=output_dir)

        logger.info("=== Full Synthetic Data Generation Completed Successfully! ===")
        return datasets


# ---------------------------------------------------------------------------
# Convenience Module Functions & CLI
# ---------------------------------------------------------------------------

def generate_customers(count: int = 100_000) -> pd.DataFrame:
    """Convenience helper to generate customers."""
    generator = SyntheticDataGenerator(num_customers=count)
    return generator.generate_customers(count)


def generate_products(count: int = 10_000) -> pd.DataFrame:
    """Convenience helper to generate products."""
    generator = SyntheticDataGenerator(num_products=count)
    return generator.generate_products(count)


def generate_orders(
    count: int = 1_000_000,
    customers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Convenience helper to generate orders."""
    generator = SyntheticDataGenerator(num_orders=count)
    return generator.generate_orders(count=count, customers_df=customers_df, products_df=products_df)


def run_generation(
    num_customers: int = 100_000,
    num_products: int = 10_000,
    num_orders: int = 1_000_000,
    output_dir: Optional[Union[str, Path]] = None,
    save_to_disk: bool = True
) -> Dict[str, pd.DataFrame]:
    """Execute complete generation process."""
    generator = SyntheticDataGenerator(
        num_customers=num_customers,
        num_products=num_products,
        num_orders=num_orders,
        output_dir=output_dir
    )
    return generator.generate_all(save_to_disk=save_to_disk)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic E-Commerce Data Generator using Faker and NumPy Distributions."
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=100_000,
        help="Number of customers to generate (default: 100,000)"
    )
    parser.add_argument(
        "--products",
        type=int,
        default=10_000,
        help="Number of products to generate (default: 10,000)"
    )
    parser.add_argument(
        "--orders",
        type=int,
        default=1_000_000,
        help="Number of orders to generate (default: 1,000,000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory path for CSV files"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Generate in-memory DataFrames without saving to disk"
    )

    args = parser.parse_args()

    run_generation(
        num_customers=args.customers,
        num_products=args.products,
        num_orders=args.orders,
        output_dir=args.output_dir,
        save_to_disk=not args.no_save
    )


if __name__ == "__main__":
    main()
