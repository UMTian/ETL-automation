"""
Setup script for Universal ETL Pipeline
"""

from setuptools import setup, find_packages

setup(
    name="universal-etl-pipeline",
    version="1.0.0",
    description="A local-first, modular ETL pipeline that follows industry best practices",
    author="ETL Pipeline Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "pyyaml", 
        "python-dotenv",
        "psycopg2-binary",
        "boto3",
        "pyarrow",
        "fastparquet",
        "sqlalchemy",
        "requests",
        "pydantic",
        "watchdog",
        "colorama",
        "rich",
        "streamlit"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "jupyter",
            "ipykernel"
        ],
        "ml": [
            "scikit-learn",
            "scipy",
            "numpy",
            "matplotlib",
            "seaborn",
            "plotly",
            "joblib",
            "mlflow",
            "optuna",
            "xgboost",
            "lightgbm",
            "catboost",
            "tensorflow",
            "torch",
            "transformers",
            "datasets",
            "tqdm"
        ]
    }
)
