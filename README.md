# E-Commerce PySpark Data Pipeline 🚀

A modular, production-ready Python and Apache PySpark data pipeline project designed to generate large-scale synthetic e-commerce transactional data, process high-volume datasets, and uncover actionable business insights.

---

## 📌 Project Overview

This project simulates an enterprise e-commerce data ecosystem:
1. **Synthetic Data Generation (`SyntheticDataGenerator`)**:
   - **Customers** (100,000 default): `customer_id`, `name`, `email`, `age` (normal distribution around 35), `city`, `country`, `registration_date`.
   - **Products** (10,000 default): `product_id`, `name`, `category` (`Electronics`, `Clothing`, `Home`, `Sports`, `Books`), `price` ($10-$500), `stock` (0-1000), `rating` (1.0-5.0).
   - **Orders** (1,000,000 default): `order_id`, `customer_id`, `product_id`, `quantity` (1-10), `order_date`.
   - **Pareto Principle**: 20% of customers make approximately 80% of orders.
   - **Performance**: Vectorized NumPy distributions, Faker batching, and `tqdm` progress tracking.
2. **PySpark Big Data Analytics**: Ingests raw CSV datasets into typed Spark DataFrames and computes key e-commerce KPIs and business intelligence aggregations.
3. **Data Lake Storage**: Stores generated inputs in `data/row/` (or `data/raw/`) and analytical outputs in `data/proceed/` (or `data/processed/`) in both **Parquet** and **CSV** formats.
4. **Automated Testing & Exploration**: Includes unit test suites for data integrity and an interactive Jupyter Notebook for visual exploration.

---

## 🗂️ Folder Structure

```text
genai-pyspark-pipeline-/
├── data/
│   ├── row/                 # Generated raw CSV data (customers.csv, products.csv, orders.csv)
│   ├── proceed/             # PySpark analytical outputs (Parquet & CSV)
│   ├── raw/                 # Alias raw data directory
│   └── processed/           # Alias processed results directory
├── notebook/
│   └── ecommerce_analytics_exploration.ipynb  # Interactive EDA and PySpark notebook
├── src/
│   ├── __init__.py          # Standard Python package marker
│   ├── _init_.py            # Alias init file
│   ├── config.py            # Central configuration (paths, defaults, Spark settings, JDK auto-detect)
│   ├── data_generator.py    # SyntheticDataGenerator class with Faker, NumPy, and tqdm
│   ├── benchmark.py         # Hardware telemetry file format benchmark (CSV, XLSX, Parquet, ORC, Feather)
│   ├── spark_analyzer.py    # PySpark transformation and business insight analytics engine
│   └── spark_analytics.py   # SalesAnalytics class (4GB memory, AQE, Kryo, Window functions)
├── test/
│   ├── __init__.py          # Test suite package marker
│   ├── test_config.py       # Configuration unit tests
│   ├── test_data_generator.py # SyntheticDataGenerator unit tests (schema, distributions, Pareto)
│   ├── test_benchmark.py    # Benchmark suite unit tests (telemetry, energy calculation)
│   └── test_sales_analytics.py # SalesAnalytics signature and interface unit tests
├── tests/                   # Additional test discovery directory
├── benchmark.py             # Root benchmark shortcut script
├── main.py                  # Primary pipeline CLI script (Parquet export & metrics)
├── spark_analytics.py       # Root SalesAnalytics shortcut script
├── run_analytics.py         # End-to-end PySpark analytics runner (loads Parquet & times operations)
├── compare_performance.py   # Benchmark comparing Pandas vs. PySpark performance on 1M rows
├── requirements.txt         # Project dependencies
├── .gitignore               # Python, Spark, and data lake git ignore rules
├── LICENSE                  # Project license
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites & Setup

### 1. Python Environment
Python 3.10+ is recommended. Activate your virtual environment and install the dependencies:

```powershell
# Activate virtual environment (.venv)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Java Runtime (For PySpark)
PySpark requires a Java JDK (JDK 11, 17, 21, or 22).
`src/config.py` includes automatic detection for installed Windows JDKs (such as `C:\Program Files\Java\jdk-22`).

---

## 🚀 How to Run

### Step 1: Generate Synthetic Data
Run the generator directly from the command line:

```powershell
# Default scale: 100K customers, 10K products, 1M orders
python src/data_generator.py

# Or customize record counts for faster development runs:
python src/data_generator.py --customers 5000 --products 500 --orders 25000
```

Alternatively, use the `SyntheticDataGenerator` class in Python:

```python
from src.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    num_customers=100_000,
    num_products=10_000,
    num_orders=1_000_000
)

# Returns pandas DataFrames and saves CSV files to data/row/
datasets = generator.generate_all(save_to_disk=True)

customers_df = datasets["customers"]
products_df = datasets["products"]
orders_df = datasets["orders"]
```

---

### Step 2: Run PySpark Analytics Pipeline
Execute the PySpark pipeline to ingest raw data, execute analytical queries, and export results:

```powershell
# Run the complete SalesAnalytics runner on data/raw/ Parquet datasets:
python run_analytics.py

# Or run the batch analytics exporter (Parquet & CSV exports to data/proceed/):
python src/spark_analyzer.py
```

Generated outputs will be saved in `data/proceed/`:
- `category_performance.csv` / `.parquet`
- `customer_lifetime_value.csv` / `.parquet`
- `monthly_sales_trends.csv` / `.parquet`
- `age_group_metrics.csv` / `.parquet`

---

### Step 3: Run Unit Tests
Execute the automated test suite:

```powershell
python -m unittest discover -s test -v
```

---

### Step 4: Interactive Analysis in Jupyter Notebook
Launch JupyterLab:

```powershell
jupyter lab
```
Open [`notebook/ecommerce_analytics_exploration.ipynb`](file:///c:/Users/tusht/OneDrive/Desktop/srashtipython/genai-pyspark-pipeline-/notebook/ecommerce_analytics_exploration.ipynb).

---

### Step 5: Benchmark File Formats (Hardware Telemetry)
Compare I/O speed, file size, peak memory (`tracemalloc`), and estimated energy consumption ($Wh$) across **CSV, XLSX, Parquet, ORC, and Feather**:

```powershell
# Run benchmark on 500,000 rows
python benchmark.py

# Fast benchmark run (skipping Excel)
python benchmark.py --rows 100000 --skip-xlsx
```

---

### Step 6: Compare Pandas vs. PySpark Performance on 1M Rows
Benchmark and compare exact execution times, memory footprint, and top 10 customer revenue calculations between **Pandas** and **PySpark**:

```powershell
python compare_performance.py
```

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for details.
