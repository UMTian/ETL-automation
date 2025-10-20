#!/usr/bin/env python3
"""
Universal ETL + ML Pipeline Dashboard
A comprehensive Streamlit dashboard for monitoring and controlling the entire pipeline.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, Any, List
import subprocess
import io

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load various types of API keys
    API_KEY = os.getenv('API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # Check which API keys are available
    available_keys = []
    if API_KEY:
        available_keys.append("Generic API")
    if GROQ_API_KEY:
        available_keys.append("Groq AI")
    if OPENAI_API_KEY:
        available_keys.append("OpenAI")
    if ANTHROPIC_API_KEY:
        available_keys.append("Anthropic")
    
    if available_keys:
        st.sidebar.success(f"🔑 API Keys loaded: {', '.join(available_keys)}")
        # Store the primary API key for use in the application
        PRIMARY_API_KEY = GROQ_API_KEY or API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY
    else:
        st.sidebar.warning("⚠️ No API keys found in .env file")
        PRIMARY_API_KEY = None
        
except ImportError:
    st.sidebar.info("📝 Install python-dotenv to load .env files")
    API_KEY = None
    GROQ_API_KEY = None
    PRIMARY_API_KEY = None
except Exception as e:
    st.sidebar.error(f"❌ Error loading .env: {e}")
    API_KEY = None
    GROQ_API_KEY = None
    PRIMARY_API_KEY = None

# Add src to path
sys.path.append('src')
sys.path.append('.')

# Display API key status in sidebar
def show_api_key_status():
    """Display API key status in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 API Configuration")
    
    if PRIMARY_API_KEY:
        st.sidebar.success("✅ API Keys Available")
        
        # Show which keys are loaded
        if GROQ_API_KEY:
            st.sidebar.info("🤖 Groq AI API: Available")
        if OPENAI_API_KEY:
            st.sidebar.info("🧠 OpenAI API: Available")
        if ANTHROPIC_API_KEY:
            st.sidebar.info("🤖 Anthropic API: Available")
        if API_KEY:
            st.sidebar.info("🔧 Generic API: Available")
            
        st.sidebar.markdown("💡 API keys are loaded from `.env` file")
    else:
        st.sidebar.warning("⚠️ No API Keys Found")
        st.sidebar.markdown("""
        **To add API keys:**
        1. Create a `.env` file in your project root
        2. Add your keys:
        ```
        GROQ_API_KEY=your_key_here
        OPENAI_API_KEY=your_key_here
        API_KEY=your_key_here
        ```
        3. Restart the dashboard
        """)
    
    st.sidebar.markdown("---")

# Call the function to display API status
show_api_key_status()

# Add API key management utility
def get_api_key_for_service(service_name: str) -> str:
    """Get the appropriate API key for a specific service."""
    service_keys = {
        'groq': GROQ_API_KEY,
        'openai': OPENAI_API_KEY,
        'anthropic': ANTHROPIC_API_KEY,
        'generic': API_KEY,
        'default': PRIMARY_API_KEY
    }
    
    # Try exact match first
    if service_name.lower() in service_keys and service_keys[service_name.lower()]:
        return service_keys[service_name.lower()]
    
    # Fall back to primary key
    return PRIMARY_API_KEY or ""

# Groq AI Functions
def analyze_data_with_groq(data: pd.DataFrame, analysis_type: str = "general") -> str:
    """Use Groq AI to analyze data and provide insights."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return "❌ Groq API not available. Please install 'groq' package and set GROQ_API_KEY."
    
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Prepare data summary for AI analysis
        data_summary = f"""
        Dataset Summary:
        - Shape: {data.shape}
        - Columns: {list(data.columns)}
        - Data Types: {dict(data.dtypes)}
        - Missing Values: {data.isnull().sum().sum()}
        - Numeric Columns: {list(data.select_dtypes(include=[np.number]).columns)}
        - Categorical Columns: {list(data.select_dtypes(include=['object', 'category']).columns)}
        """
        
        # Sample data for context
        sample_data = data.head(5).to_string()
        
        # Create analysis prompt based on type
        if analysis_type == "quality":
            prompt = f"""
            Analyze this dataset for data quality issues and provide actionable recommendations:
            
            {data_summary}
            
            Sample Data:
            {sample_data}
            
            Please provide:
            1. Data quality assessment
            2. Specific issues found
            3. Recommended cleaning steps
            4. Business impact of issues
            """
        elif analysis_type == "insights":
            prompt = f"""
            Analyze this dataset and provide business insights and recommendations:
            
            {data_summary}
            
            Sample Data:
            {sample_data}
            
            Please provide:
            1. Key patterns and trends
            2. Business opportunities
            3. Risk factors
            4. Recommended actions
            """
        else:  # general
            prompt = f"""
            Provide a comprehensive analysis of this dataset:
            
            {data_summary}
            
            Sample Data:
            {sample_data}
            
            Please provide:
            1. Dataset overview
            2. Key characteristics
            3. Potential use cases
            4. Recommendations
            """
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a data science expert specializing in ETL pipelines and business intelligence. Provide clear, actionable insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",  # Fast and reliable production model
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error calling Groq API: {str(e)}"

def generate_etl_recommendations_with_groq(data: pd.DataFrame) -> str:
    """Use Groq AI to generate ETL pipeline recommendations."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return "❌ Groq API not available. Please install 'groq' package and set GROQ_API_KEY."
    
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Data characteristics for ETL recommendations
        data_info = f"""
        Data Characteristics:
        - Shape: {data.shape}
        - Columns: {list(data.columns)}
        - Data Types: {dict(data.dtypes)}
        - Missing Values: {data.isnull().sum().sum()}
        - Duplicates: {data.duplicated().sum()}
        - Numeric Columns: {list(data.select_dtypes(include=[np.number]).columns)}
        - Categorical Columns: {list(data.select_dtypes(include=['object', 'category']).columns)}
        - Date Columns: {[col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]}
        """
        
        prompt = f"""
        As an ETL pipeline expert, analyze this dataset and provide specific ETL recommendations:
        
        {data_info}
        
        Please provide:
        1. Recommended data cleaning steps
        2. Data transformation suggestions
        3. Quality checks to implement
        4. Pipeline configuration recommendations
        5. Performance optimization tips
        6. Monitoring and alerting suggestions
        
        Be specific and actionable.
        """
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an ETL pipeline expert with deep knowledge of data engineering best practices."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",  # Fast and reliable production model
            temperature=0.2,
            max_tokens=1200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error calling Groq API: {str(e)}"

def get_ai_data_summary_with_groq(data: pd.DataFrame) -> str:
    """Use Groq AI to generate a natural language data summary."""
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return "❌ Groq API not available. Please install 'groq' package and set GROQ_API_KEY."
    
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Basic statistics
        numeric_stats = data.describe().to_string() if len(data.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"
        
        prompt = f"""
        Create a natural language summary of this dataset for business stakeholders:
        
        Dataset Info:
        - Shape: {data.shape}
        - Columns: {list(data.columns)}
        
        Numeric Statistics:
        {numeric_stats}
        
        Please provide:
        1. Executive summary
        2. Key metrics and insights
        3. Data quality assessment
        4. Business implications
        5. Next steps
        
        Use business-friendly language.
        """
        
        # Call Groq API
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst who translates data insights into business value."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",  # Fast and reliable production model
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error calling Groq API: {str(e)}"

# Example usage in functions that need API keys
def test_api_connection(service: str = 'default'):
    """Test API connection for a specific service."""
    api_key = get_api_key_for_service(service)
    if api_key:
        return f"✅ API key available for {service}"
    else:
        return f"❌ No API key available for {service}"

# Disable cross-drive file watcher issues on Windows
try:
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    from streamlit import config as _st_config
    _st_config.set_option('server.fileWatcherType', 'none')
except Exception:
    pass

# Set page config
st.set_page_config(
    page_title="ETL + ML Pipeline Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left-color: #28a745;
    }
    .warning-card {
        border-left-color: #ffc107;
    }
    .error-card {
        border-left-color: #dc3545;
    }
    .pipeline-status {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    /* Ensure proper scrolling */
    .stDataFrame {
        max-height: 70vh;
        overflow-y: auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    /* Make sure content is scrollable */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        text-align: center;
    }
    .status-running {
        background-color: #d4edda;
        color: #155724;
    }
    .status-completed {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    /* Force page containers to allow scrolling */
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        height: auto !important;
        overflow: auto !important;
    }
    [data-testid="stSidebar"] {
        overflow: auto !important;
    }
    /* Ensure vertical blocks do not clip children */
    [data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# Import our pipeline components directly (only what we need for the UI)
try:
    from src.ml.model_trainer import ModelTrainer
    from src.models import create_model_config
    from src.ml.feature_engineering import FeatureEngineer
    from src.pipeline import ETLPipeline
    from src.core.config import PipelineConfig
    from utils.dual_logger import DualLogger
except ImportError:
    # If imports fail, we'll handle it gracefully in the UI
    ModelTrainer = None
    create_model_config = None
    FeatureEngineer = None
    ETLPipeline = None
    PipelineConfig = None
    DualLogger = None

# Groq API functionality
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.sidebar.warning("⚠️ Install 'groq' package for AI features: `pip install groq`")

# --- Data Loading and Generation Functions ---
def load_sample_data():
    """Load sample data for dashboard visualization when no ETL pipeline has been run."""
    try:
        # Try to load existing ETL output files
        output_dir = "data/output"
        if os.path.exists(output_dir):
            # Look for the most recent parquet file
            parquet_files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
            if parquet_files:
                # Sort by modification time and get the most recent
                parquet_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                latest_file = os.path.join(output_dir, parquet_files[0])
                try:
                    df = pd.read_parquet(latest_file)
                    return df, f"Loaded from: {latest_file}"
                except Exception as e:
                    st.warning(f"Could not load {latest_file}: {e}")
        
        # Try to load from input directory
        input_dir = "data/input"
        if os.path.exists(input_dir):
            # Look for any CSV files
            csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
            if csv_files:
                # Use the first CSV file found
                sample_file = os.path.join(input_dir, csv_files[0])
                try:
                    df = pd.read_csv(sample_file)
                    return df, f"Loaded from: {sample_file}"
                except Exception as e:
                    st.warning(f"Could not load {sample_file}: {e}")
        
        # Fallback: create minimal sample data
        np.random.seed(42)
        n_records = 100
        df = pd.DataFrame({
            'id': range(1, n_records + 1),
            'value': np.random.normal(100, 30, n_records),
            'category': np.random.choice(['A', 'B', 'C'], n_records),
            'date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
            'status': np.random.choice(['active', 'inactive'], n_records)
        })
        return df, "Generated minimal sample data"
        
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        # Return empty DataFrame as last resort
        return pd.DataFrame(), "No data available"

# --- Helpers to make DataFrames Arrow/Streamlit-safe ---
def _fix_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    try:
        fixed = df.copy()
        # Reset index to avoid exotic index types confusing Arrow
        if not fixed.index.equals(pd.RangeIndex(start=0, stop=len(fixed), step=1)):
            fixed = fixed.reset_index(drop=True)
        # Ensure all column names are strings (Arrow can choke on None or mixed)
        cols = [str(c) if c is not None else "Unnamed" for c in fixed.columns]
        # Ensure unique column names
        seen = {}
        unique_cols = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                unique_cols.append(c)
            else:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
        fixed.columns = unique_cols
        for col in fixed.columns:
            series = fixed[col]
            # Convert datetime64[ns, tz] to naive or ISO strings to be safe
            if isinstance(series.dtype, pd.DatetimeTZDtype):
                fixed[col] = series.dt.tz_convert(None)
                series = fixed[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                # Keep as datetime where possible; Arrow supports it. If object-mixed, convert to ISO string below
                pass
            # Replace inf/-inf with NaN, then fill numerics to avoid Arrow errors
            if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
                series = series.replace([np.inf, -np.inf], np.nan)
                # Do not change dtype unnecessarily; fill NaN with median if possible
                if series.isnull().any():
                    try:
                        # Check if we have any non-NaN values before calculating median
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0:
                            med = np.nanmedian(non_null_series.astype(float))
                            series = series.fillna(med)
                        else:
                            # If all values are NaN, fill with 0
                            series = series.fillna(0)
                    except Exception:
                        series = series.fillna(0)
                fixed[col] = series
                continue
            # Convert category to string to avoid mixed/unknown Arrow type
            if isinstance(series.dtype, pd.CategoricalDtype):
                fixed[col] = series.astype(str)
                continue
            # For object columns, coerce to string (mixed objects are unsafe for Arrow)
            if pd.api.types.is_object_dtype(series):
                # If looks numeric, try numeric coercion; else stringify
                coerced = pd.to_numeric(series, errors='coerce')
                # Decide based on how many became numbers
                if coerced.notna().sum() >= max(5, int(0.5 * len(series))):
                    coerced = coerced.replace([np.inf, -np.inf], np.nan)
                    if coerced.isnull().any():
                        try:
                            # Check if we have any non-NaN values before calculating median
                            non_null_coerced = coerced.dropna()
                            if len(non_null_coerced) > 0:
                                med = np.nanmedian(non_null_coerced.astype(float))
                                coerced = coerced.fillna(med)
                            else:
                                # If all values are NaN, fill with 0
                                coerced = coerced.fillna(0)
                        except Exception:
                            coerced = coerced.fillna(0)
                    fixed[col] = coerced
                else:
                    # Convert to string, replacing NaN with empty string to avoid object issues
                    fixed[col] = series.astype(str).replace({"nan": "", "None": ""})
        return fixed
    except Exception:
        # On any unexpected issue, return original to avoid hard failure
        return df

def _safe_dataframe(df: pd.DataFrame, rows: int = None):
    try:
        display_df = _fix_dataframe_for_streamlit(df)
        if rows is not None:
            try:
                # Ensure rows is an integer and non-negative
                safe_rows = int(rows)
                if safe_rows < 0:
                    safe_rows = 0
            except Exception:
                # Fallback: ignore invalid rows input
                safe_rows = None
            if safe_rows is not None:
                display_df = display_df.head(safe_rows)
        st.dataframe(display_df, width='stretch')
    except Exception as e:
        st.error(f"Data display error: {e}")
        try:
            # Mirror the same safety when building JSON fallback
            safe_rows_json = None
            if rows is not None:
                try:
                    safe_rows_json = int(rows)
                    if safe_rows_json < 0:
                        safe_rows_json = 0
                except Exception:
                    safe_rows_json = None
            st.json(
                display_df.head(safe_rows_json).to_dict(orient='records')
                if safe_rows_json is not None else display_df.to_dict(orient='records')
            )
        except Exception:
            st.write("Unable to render preview.")

class PipelineDashboard:
    def __init__(self):
        self.pipeline_status = "idle"
        self.pipeline_results = {}
        self.ml_results = {}
        self.data_cache = {}
        # Dual logger: UI + terminal
        try:
            self.logger = DualLogger(ui_enabled=True)
        except Exception:
            self.logger = None
        
    def load_data_from_file(self, file_path: str):
        """Load data from a file path."""
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == '.csv':
                return pd.read_csv(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                st.error(f"Unsupported file type: {file_ext}")
                return None
        except Exception as e:
            st.error(f"Error loading file {file_path}: {e}")
            return None
    
    def run_etl_pipeline(self, config_path: str):
        """Run the ETL pipeline."""
        try:
            config = PipelineConfig.from_yaml(config_path)
            pipeline = ETLPipeline(config)
            result = pipeline.run()
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_ml_pipeline(self, data_type: str, task: str, models: List[str]):
        """Run the ML pipeline."""
        try:
            if self.logger:
                self.logger.info(f"START ML pipeline: data={data_type}, task={task}, models={models}")
            
            # Use uploaded data from cache
            if not self.data_cache:
                return {"success": False, "error": "No data available. Please upload data first."}
            
            # Get the first available dataset
            data_name = list(self.data_cache.keys())[0]
            data = self.data_cache[data_name]
            st.info(f"Using uploaded data: {data_name} with {len(data)} rows")
            
            if data is None or len(data) == 0:
                return {"success": False, "error": "No valid data available for ML pipeline"}
            
            # Prepare features based on data type
            if data_type == "ml_training":
                # ML training data has predefined targets
                if task == "classification":
                    target_col = "loan_approval"
                    X = data.drop(columns=['loan_approval', 'loan_amount', 'customer_segment'])
                    y = data['loan_approval']
                elif task == "regression":
                    target_col = "loan_amount"
                    X = data.drop(columns=['loan_approval', 'loan_amount', 'customer_segment'])
                    y = data['loan_amount']
                else:
                    return {"success": False, "error": "Unsupported task"}
                    
            elif data_type == "ecommerce":
                # Create target columns for e-commerce data
                if task == "classification":
                    # Create binary classification target: high value transactions
                    if 'total_amount' in data.columns:
                        median_amount = data['total_amount'].median()
                        data['high_value_transaction'] = (data['total_amount'] > median_amount).astype(int)
                        target_col = "high_value_transaction"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'total_amount' column found in e-commerce data"}
                elif task == "regression":
                    if 'total_amount' in data.columns:
                        target_col = "total_amount"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'total_amount' column found in e-commerce data"}
                        
            elif data_type == "financial":
                # Create target columns for financial data
                if task == "classification":
                    # Create binary classification target: fraudulent transactions
                    if 'amount' in data.columns:
                        # Simulate fraud detection based on amount and other features
                        data['is_fraudulent'] = ((data['amount'] > data['amount'].quantile(0.95)) | 
                                               (data['amount'] < data['amount'].quantile(0.05))).astype(int)
                        target_col = "is_fraudulent"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'amount' column found in financial data"}
                elif task == "regression":
                    if 'amount' in data.columns:
                        target_col = "amount"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'amount' column found in financial data"}
                        
            elif data_type == "healthcare":
                # Create target columns for healthcare data
                if task == "classification":
                    # Create binary classification target: has diagnosis
                    if 'diagnosis' in data.columns:
                        data['has_diagnosis'] = (data['diagnosis'].notna() & (data['diagnosis'] != '')).astype(int)
                        target_col = "has_diagnosis"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'diagnosis' column found in healthcare data"}
                elif task == "regression":
                    if 'cost' in data.columns:
                        target_col = "cost"
                        X = data.drop(columns=[target_col, 'data_source'])
                        y = data[target_col]
                    else:
                        return {"success": False, "error": "No 'cost' column found in healthcare data"}
            else:
                return {"success": False, "error": f"Unsupported data type: {data_type}"}
            
            # Remove non-numeric columns from features
            X = X.select_dtypes(include=[np.number])
            
            if X.empty:
                return {"success": False, "error": "No numerical features found after preprocessing"}
            
            # Handle NaN values in target column
            if y.isnull().any():
                st.warning(f"Found {y.isnull().sum()} NaN values in target column '{target_col}'. Removing these rows.")
                # Remove rows where target is NaN
                valid_indices = y.notnull()
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) == 0:
                    return {"success": False, "error": "No valid data remaining after removing NaN target values"}
            
            # Feature engineering
            feature_config = {
                'handle_missing_values': True,
                'encode_categorical': True,
                'scale_features': True,
                'feature_selection': {'enabled': True, 'n_features': min(10, X.shape[1])}
            }
            
            feature_engineer = FeatureEngineer(feature_config)
            with self.logger.step("Feature Engineering", f"{X.shape[1]}-> select/scale/encode") if self.logger else _nullcontext():
                X_processed = feature_engineer.fit_transform(X, target_col)
            
            # Model configuration
            ml_config = {
                'feature_engineering': feature_config,
                'data_splitting': {'test_size': 0.2, 'random_state': 42},
                'models': [create_model_config(task, model) for model in models],
                'hyperparameter_optimization': {'enabled': True, 'n_trials': 5},
                'cross_validation': {'enabled': True, 'cv_folds': 3}
            }
            
            # Train models
            trainer = ModelTrainer(ml_config)
            with self.logger.step("Train/Test Split") if self.logger else _nullcontext():
                X_train, X_test, y_train, y_test = trainer.split_data(X_processed, y)
            
            # Check for NaN values in training data
            if X_train.isnull().any().any():
                st.warning("Found NaN values in training features. Filling with median values.")
                X_train = X_train.fillna(X_train.median())
                X_test = X_test.fillna(X_test.median())
            
            if y_train.isnull().any():
                st.warning("Found NaN values in training target. Removing these rows.")
                valid_train = y_train.notnull()
                X_train = X_train[valid_train]
                y_train = y_train[valid_train]
                
                if len(X_train) == 0:
                    return {"success": False, "error": "No valid training data remaining after removing NaN target values"}
            
            with self.logger.step("Model Training", ", ".join(models)) if self.logger else _nullcontext():
                results = trainer.train_multiple_models(X_train, y_train, X_test, y_test)
                if self.logger and results:
                    self.logger.info(f"Best model: {trainer.best_model.config.get('model_type')} score={trainer.best_score:.4f}")
            
            return {
                "success": True,
                "results": results,
                "best_model": trainer.best_model.config.get('model_type') if trainer.best_model else None,
                "best_score": trainer.best_score if trainer.best_score else 0,
                "data_shape": X_processed.shape
            }
            
        except Exception as e:
            st.error(f"ML Pipeline Error: {str(e)}")
            return {"success": False, "error": str(e)}


# Fallback nullcontext for when logger is not available
class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def main():
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = PipelineDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Header with improved styling
    st.markdown('<h1 class="main-header">🚀 Universal ETL + ML Pipeline Dashboard</h1>', unsafe_allow_html=True)
    
    # Welcome message
    if not dashboard.data_cache:
        st.info("👋 **Welcome!** Start by loading unified datasets or uploading new files from the sidebar.")
        st.markdown("""
        ### 🎯 Quick Start Guide:
        1. **📁 Load Unified Data**: Use the "Unified Data" tab to load comprehensive datasets
        2. **📂 Load Original Files**: Use "Original Files" tab for individual data files
        3. **📤 Upload Custom Files**: Use "Upload Files" tab for new data
        4. **📊 Analyze**: Explore your data with the visualization tabs below
        5. **🔄 Process**: Run ETL pipelines to clean and transform your data
        6. **🤖 ML**: Train machine learning models on your processed data
        """)
        
        # Show available unified datasets
        st.markdown("### 🎯 Available Unified Datasets:")
        unified_info = [
            ("💰 Financial Data", "3,533 records", "Banking, stocks, fraud detection"),
            ("🛒 E-commerce Data", "6,575 records", "Transactions, customers, products"),
            ("🏥 Healthcare Data", "2,800 records", "Patients, visits, medical billing"),
            ("🤖 ML Analytics", "2,366 records", "Loan data, time series analytics"),
            ("📚 Reference Data", "JSON files", "Product catalogs, user profiles")
        ]
        
        for category, records, description in unified_info:
            st.markdown(f"**{category}**: {records} - {description}")
    
    # Sidebar
    st.sidebar.title("🎛️ Pipeline Controls")
    
    # Verbosity control
    st.sidebar.subheader("🔊 Logging")
    log_level = st.sidebar.selectbox("Log level", ["DEBUG", "INFO", "WARN", "ERROR"], index=1)
    try:
        dashboard.logger.set_level(log_level)
    except Exception:
        pass

    # Pipeline selection
    pipeline_type = st.sidebar.selectbox(
        "Select Pipeline Type",
        ["Data Upload", "ETL Pipeline", "ML Pipeline", "Complete ETL + ML Pipeline"]
    )
    
    if pipeline_type == "Data Upload":
        show_data_upload(dashboard)
    elif pipeline_type == "ETL Pipeline":
        show_etl_pipeline(dashboard)
    elif pipeline_type == "ML Pipeline":
        show_ml_pipeline(dashboard)
    else:
        show_complete_pipeline(dashboard)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_pipeline_monitoring(dashboard)
    
    with col2:
        show_quick_stats(dashboard)
    
    # Data visualization section - Full width
    st.markdown("---")  # Add separator
    show_data_visualization(dashboard)
    
    # Model results section
    if dashboard.ml_results:
        st.markdown("---")  # Add separator
        show_model_results(dashboard)

def show_data_upload(dashboard):
    """Show enhanced data upload controls with unified data support."""
    st.sidebar.subheader("📤 Data Upload")
    
    # Create tabs for different data sources
    upload_tab1, upload_tab2, upload_tab3 = st.sidebar.tabs(["📁 Unified Data", "📂 Original Files", "📤 Upload Files"])
    
    with upload_tab1:
        st.sidebar.write("**🎯 Recommended: Unified Datasets**")
        
        # Show unified data files
        unified_dirs = [
            ("💰 Financial", "data/unified/financial"),
            ("🛒 E-commerce", "data/unified/ecommerce"), 
            ("🏥 Healthcare", "data/unified/healthcare"),
            ("🤖 ML Analytics", "data/unified/ml_analytics"),
            ("📚 Reference", "data/unified/reference_data")
        ]
        
        unified_files = []
        for category, dir_path in unified_dirs:
            if Path(dir_path).exists():
                for pattern in ["*.csv", "*.json"]:
                    files = list(Path(dir_path).glob(pattern))
                    for file in files:
                        unified_files.append((f"{category} - {file.name}", str(file)))
        
        if unified_files:
            # File selection
            selected_unified = st.sidebar.selectbox(
                "Select unified dataset",
                [f[0] for f in unified_files],
                index=None,
                placeholder="Choose unified dataset..."
            )
            
            if selected_unified and st.sidebar.button("📥 Load Unified Data", key="load_unified"):
                file_path = next(f[1] for f in unified_files if f[0] == selected_unified)
                try:
                    # Load based on file extension
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext == '.csv':
                        data = pd.read_csv(file_path)
                    elif file_ext == '.json':
                        data = pd.read_json(file_path)
                    else:
                        st.error(f"Unsupported file type: {file_ext}")
                        return
                    
                    if not data.empty:
                        # Add data source identifier
                        data['data_source'] = Path(file_path).stem
                        
                        # Cache the data (sanitized for Streamlit)
                        dashboard.data_cache[selected_unified] = _fix_dataframe_for_streamlit(data)
                        st.success(f"✅ Loaded {len(data)} rows from '{selected_unified}'")
                    else:
                        st.warning(f"File '{selected_unified}' is empty.")
                except Exception as e:
                    st.error(f"Error loading file '{selected_unified}': {e}")
        else:
            st.sidebar.info("No unified data found. Run data combination first.")
    
    with upload_tab2:
        st.sidebar.write("**📂 Original Input Files**")
        
        # Show available input files
        input_dir = Path("data/input")
        if input_dir.exists():
            # Get all supported files
            all_files = []
            for pattern in ["*.csv", "*.json", "*.parquet", "*.xlsx", "*.xls"]:
                all_files.extend(list(input_dir.glob(pattern)))
            
            if all_files:
                # Show total file count
                st.sidebar.write(f"**Available Files:** {len(all_files)}")
                
                # File selection
            selected_file = st.sidebar.selectbox(
                    "Select original file",
                    [f.name for f in all_files],
                index=None,
                    placeholder="Choose original file..."
            )
            
                if selected_file and st.sidebar.button("📥 Load Original File", key="load_original"):
                    file_path = input_dir / selected_file
                try:
                        # Load based on file extension
                        file_ext = file_path.suffix.lower()
                        if file_ext == '.csv':
                    data = pd.read_csv(file_path)
                        elif file_ext == '.json':
                            data = pd.read_json(file_path)
                        elif file_ext == '.parquet':
                            data = pd.read_parquet(file_path)
                        elif file_ext in ['.xlsx', '.xls']:
                            data = pd.read_excel(file_path)
                        else:
                            st.error(f"Unsupported file type: {file_ext}")
                            return
                        
                    if not data.empty:
                        # Add data source identifier
                        data['data_source'] = selected_file.split('.')[0]
                        
                        # Cache the data (sanitized for Streamlit)
                        dashboard.data_cache[selected_file] = _fix_dataframe_for_streamlit(data)
                        st.success(f"✅ Loaded {len(data)} rows from '{selected_file}'")
                    else:
                        st.warning(f"File '{selected_file}' is empty.")
                except Exception as e:
                    st.error(f"Error loading file '{selected_file}': {e}")
            else:
                st.sidebar.info("No files found in data/input/")
        else:
            st.sidebar.warning("data/input/ directory not found")
    
    with upload_tab3:
        st.sidebar.write("**📤 Upload Custom Files**")
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload a file", 
            type=["csv", "json", "parquet", "xlsx", "xls"],
            help="Supported: CSV, JSON, Parquet, Excel"
        )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_name = uploaded_file.name
        
            try:
                # Load based on file type
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension == "json":
            data = pd.read_json(uploaded_file)
        elif file_extension == "parquet":
            data = pd.read_parquet(uploaded_file)
                elif file_extension in ["xlsx", "xls"]:
                    data = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return
        
        if data.empty:
            st.warning(f"Uploaded file '{file_name}' is empty.")
            return
        
        # Add a data source identifier
                data['data_source'] = file_name.split('.')[0]
        
        # Cache the data (sanitized for Streamlit)
        dashboard.data_cache[file_name] = _fix_dataframe_for_streamlit(data)
                st.success(f"✅ Uploaded '{file_name}' ({len(data)} rows)")
                
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
    
    # Data management section
    st.sidebar.subheader("🗂️ Data Management")
    
    if dashboard.data_cache:
        st.sidebar.write(f"**Loaded:** {len(dashboard.data_cache)} dataset(s)")
        
        # Show loaded datasets
        for dataset_name, dataset in dashboard.data_cache.items():
            if hasattr(dataset, 'shape'):
                st.sidebar.write(f"• {dataset_name}: {dataset.shape[0]} rows × {dataset.shape[1]} cols")
        
        # Clear all data button
        if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
            dashboard.data_cache.clear()
            st.success("✅ All data cleared")
            st.rerun()
    else:
        st.sidebar.info("No data loaded yet")

def show_etl_pipeline(dashboard):
    """Show ETL pipeline controls."""
    st.sidebar.subheader("📊 ETL Pipeline Configuration")
    
    # Check if we have data loaded
    if not dashboard.data_cache:
        st.sidebar.warning("⚠️ No data loaded. Please load data first from the 'Data Upload' section.")
        return
    
    # Select data to process
    data_name = st.sidebar.selectbox(
        "Select Data to Process",
        list(dashboard.data_cache.keys())
    )
    
    if data_name:
        data = dashboard.data_cache[data_name]
        
        # Show data info
        st.sidebar.write(f"**Selected Data:** {data_name}")
        st.sidebar.write(f"Shape: {data.shape}")
    
    # Output format
    output_format = st.sidebar.selectbox("Output Format", ["CSV", "JSON", "Parquet"])
    
    # Run button
    if st.sidebar.button("🚀 Run ETL Pipeline", type="primary"):
        with st.spinner("Running ETL Pipeline..."):
                try:
                    # Save processed data
                    output_path = f"data/output/{data_name}_processed.{output_format.lower()}"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if output_format == "CSV":
                    data.to_csv(output_path, index=False)
                elif output_format == "JSON":
                    data.to_json(output_path, orient='records', indent=2)
                elif output_format == "Parquet":
                    data.to_parquet(output_path, index=False)
                
                    st.success(f"✅ Data processed and saved to {output_path}")
                dashboard.pipeline_status = "completed"
                dashboard.pipeline_results = {
                    "success": True,
                        "data_name": data_name,
                    "sample_size": len(data),
                    "output_path": output_path
                }
                except Exception as e:
                    st.error(f"Error processing data: {e}")
                    dashboard.pipeline_status = "error"

def show_ml_pipeline(dashboard):
    """Show ML pipeline controls."""
    st.sidebar.subheader("🤖 ML Pipeline Configuration")
    
    # Data type
    data_type = st.sidebar.selectbox(
        "Training Data",
        ["ml_training", "ecommerce", "financial", "healthcare"]
    )
    
    # ML task
    task = st.sidebar.selectbox("ML Task", ["classification", "regression"])
    
    # Model selection
    available_models = {
        "classification": ["random_forest", "logistic_regression", "gradient_boosting", "svm"],
        "regression": ["linear_regression", "random_forest", "gradient_boosting", "svr"]
    }
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models[task],
        default=available_models[task][:3]
    )
    
    # Hyperparameter optimization
    enable_optimization = st.sidebar.checkbox("Enable Hyperparameter Optimization", value=True)
    
    # Run button
    if st.sidebar.button("🤖 Train ML Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model")
            return
            
        with st.spinner("Training ML Models..."):
            result = dashboard.run_ml_pipeline(data_type, task, selected_models)
            
            if result["success"]:
                dashboard.ml_results = result
                dashboard.pipeline_status = "completed"
                st.success("✅ ML Pipeline completed successfully!")
            else:
                st.error(f"❌ ML Pipeline failed: {result['error']}")

def show_complete_pipeline(dashboard):
    """Show complete ETL + ML pipeline controls."""
    st.sidebar.subheader("🔄 Complete Pipeline Configuration")
    
    # Data type
    data_type = st.sidebar.selectbox(
        "Data Type",
        ["ecommerce", "financial", "healthcare", "ml_training"]
    )
    
    # ML task
    task = st.sidebar.selectbox("ML Task", ["classification", "regression"])
    
    # Models
    available_models = {
        "classification": ["random_forest", "logistic_regression", "gradient_boosting"],
        "regression": ["linear_regression", "random_forest", "gradient_boosting"]
    }
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models[task],
        default=available_models[task][:2]
    )
    
    # Run complete pipeline
    if st.sidebar.button("🔄 Run Complete Pipeline", type="primary"):
        if not selected_models:
            st.error("Please select at least one model")
            return
            
        with st.spinner("Running Complete ETL + ML Pipeline..."):
            # Step 1: Use uploaded data
            if not dashboard.data_cache:
                st.error("No data available. Please upload data first.")
                return
            
            data_name = list(dashboard.data_cache.keys())[0]
            data = dashboard.data_cache[data_name]
            st.success(f"✅ Using uploaded data: {data_name} with {len(data)} rows")
                
                # Step 2: Run ML pipeline
                result = dashboard.run_ml_pipeline(data_type, task, selected_models)
                
                if result["success"]:
                    dashboard.ml_results = result
                    dashboard.pipeline_status = "completed"
                    st.success("✅ Complete Pipeline finished successfully!")
                else:
                    st.error(f"❌ Pipeline failed: {result['error']}")

def show_pipeline_monitoring(dashboard):
    """Show pipeline monitoring dashboard."""
    st.subheader("📊 Pipeline Monitoring")
    
    # Status indicator
    status_colors = {
        "idle": "gray",
        "running": "blue",
        "completed": "green",
        "error": "red"
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Pipeline Status",
            dashboard.pipeline_status.upper(),
            delta=None
        )
    
    with col2:
        if dashboard.pipeline_results:
            st.metric(
                "Records Processed",
                dashboard.pipeline_results.get("sample_size", 0)
            )
    
    with col3:
        if dashboard.ml_results:
            st.metric(
                "Best Model Score",
                f"{dashboard.ml_results.get('best_score', 0):.4f}"
            )
    
    # Real-time progress (simulated)
    if dashboard.pipeline_status == "running":
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        dashboard.pipeline_status = "completed"

def show_quick_stats(dashboard):
    """Show quick statistics."""
    st.subheader("📈 Quick Statistics")
    
    if dashboard.data_cache:
        for data_type, data in dashboard.data_cache.items():
            with st.expander(f"{data_type.title()} Data Stats"):
                try:
                    # Ensure data is a DataFrame
                    if hasattr(data, 'shape') and hasattr(data, 'columns'):
                        st.write(f"**Shape:** {data.shape}")
                        st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
                        st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
                        
                        # Show data source distribution if available
                        if 'data_source' in data.columns:
                            st.write("**Data Sources:**")
                            source_counts = data['data_source'].value_counts()
                            for source, count in source_counts.items():
                                st.write(f"  - {source}: {count} rows")
                        
                        # Show sample data
                        st.write("**Sample Data:**")
                        _safe_dataframe(data.head(50))
                    else:
                        st.error(f"Invalid data format for {data_type}. Expected DataFrame, got {type(data)}")
                        st.write(f"Data type: {type(data)}")
                        if hasattr(data, '__len__'):
                            st.write(f"Length: {len(data)}")
                except Exception as e:
                    st.error(f"Error processing {data_type} data: {e}")
                    st.write(f"Data type: {type(data)}")
                    if hasattr(data, '__len__'):
                        st.write(f"Length: {len(data)}")

def show_data_visualization(dashboard):
    """Show data visualization."""
    st.subheader("📊 Data Visualization")
    
    if not dashboard.data_cache:
        st.info("No data available for visualization. Run a pipeline first.")
        return
    
    # Select data for visualization
    data_type = st.selectbox("Select Data for Visualization", list(dashboard.data_cache.keys()))
    data = dashboard.data_cache[data_type]
    
    # Ensure data is a DataFrame
    if not hasattr(data, 'shape') or not hasattr(data, 'columns'):
        st.error(f"Invalid data format for visualization. Expected DataFrame, got {type(data)}")
        return
    
    # Add data source filter if available
    if 'data_source' in data.columns:
        available_sources = data['data_source'].unique()
        selected_source = st.selectbox("Filter by Data Source", ["All"] + list(available_sources))
        
        if selected_source != "All":
            data = data[data['data_source'] == selected_source]
            st.info(f"Showing data from: {selected_source}")
    
    # Add preprocessing visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Data Distribution", 
        "🔍 Data Quality", 
        "🧹 Preprocessing Steps", 
        "📊 Correlation Analysis",
        "📋 Sample Data",
        "💼 Business Insights",
        "🔄 ETL Results",
        "📈 Advanced Analytics",
        "🎯 Actionable Insights",
        "🤖 AI Insights"
    ])
    
    with tab1:
        show_distribution_plots(data)
    
    with tab2:
        show_data_quality_analysis(data)
    
    with tab3:
        show_preprocessing_steps(data)
    
    with tab4:
        show_correlation_analysis(data)
    
    with tab5:
        show_sample_data(data)
    
    with tab6:
        show_business_insights(dashboard)
    
    with tab7:
        show_etl_results(dashboard)
    
    with tab8:
        show_advanced_analytics(dashboard)
    
    with tab9:
        show_actionable_insights(dashboard)
    
    with tab10:
        show_ai_insights(dashboard)

def show_distribution_plots(data):
    """Show distribution plots for numerical columns."""
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        selected_col = st.selectbox("Select Column for Distribution", numerical_cols)
        
        # Create distribution plot
        fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, width='stretch', key="data_distribution_histogram")
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{data[selected_col].mean():.2f}")
        with col2:
            st.metric("Median", f"{data[selected_col].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{data[selected_col].std():.2f}")
        with col4:
            st.metric("Count", f"{len(data[selected_col].dropna())}")
    else:
        st.warning("No numerical columns found for distribution plots.")

def show_data_quality_analysis(data):
    """Show comprehensive data quality analysis."""
    st.subheader("🔍 Data Quality Analysis")
    
    # Data shape and memory
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(data))
    with col2:
        st.metric("Total Columns", len(data.columns))
    with col3:
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    
    # Missing values analysis
    st.subheader("📊 Missing Values Analysis")
    missing_data = data.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': (missing_data.values / len(data)) * 100
    })
    
    if missing_data.sum() > 0:
        fig = px.bar(
            missing_df[missing_df['Missing Count'] > 0],
            x='Column',
            y='Missing Count',
            title="Missing Values by Column",
            color='Missing Percentage',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, width='stretch', key="missing_values_chart")
        
        # Show missing values table
        _safe_dataframe(missing_df[missing_df['Missing Count'] > 0])
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data types analysis
    st.subheader("🔧 Data Types Analysis")
    dtype_df = pd.DataFrame({
        'Column': data.columns,
        'Data Type': data.dtypes.astype(str),
        'Unique Values': [data[col].nunique() for col in data.columns],
        'Sample Values': [str(data[col].dropna().head(3).tolist()) for col in data.columns]
    })
    _safe_dataframe(dtype_df)

def show_preprocessing_steps(data):
    """Show preprocessing steps and options."""
    st.subheader("🧹 Data Preprocessing Pipeline")
    
    # Preprocessing configuration
    st.write("**Configure Preprocessing Steps:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
        scale_features = st.checkbox("Scale Numerical Features", value=True)
    
    with col2:
        feature_selection = st.checkbox("Feature Selection", value=True)
        outlier_detection = st.checkbox("Outlier Detection", value=True)
        create_features = st.checkbox("Create New Features", value=True)
        dimensionality_reduction = st.checkbox("Dimensionality Reduction", value=False)
    
    # Missing value handling options
    if handle_missing:
        st.write("**Missing Value Strategy:**")
        missing_strategy = st.selectbox(
            "Select Strategy",
            ["mean", "median", "most_frequent", "constant", "drop"],
            help="Choose how to handle missing values"
        )
        
        if missing_strategy == "constant":
            fill_value = st.number_input("Fill Value", value=0.0)
    
    # Feature selection options
    if feature_selection:
        n_features = st.slider(
            "Number of Features to Select",
            min_value=1,
            max_value=min(20, len(data.select_dtypes(include=[np.number]).columns)),
            value=min(10, len(data.select_dtypes(include=[np.number]).columns))
        )
    
    # Outlier detection options
    if outlier_detection:
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["zscore", "iqr", "isolation_forest"],
            help="Choose outlier detection method"
        )
        
        if outlier_method == "zscore":
            z_threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        elif outlier_method == "iqr":
            iqr_multiplier = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        elif outlier_method == "isolation_forest":
            contamination = st.slider("Contamination Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01, 
                                   help="Expected proportion of outliers in the dataset (0.1 = 10%)")
    
    # Run preprocessing button
    if st.button("🚀 Run Preprocessing Pipeline", type="primary"):
        with st.spinner("Running preprocessing pipeline..."):
            # Simulate preprocessing steps
            config_dict = locals()
            if 'contamination' in config_dict:
                config_dict['contamination'] = contamination
            preprocessing_results = run_preprocessing_simulation(data, config_dict)
            show_preprocessing_results(preprocessing_results)

def run_preprocessing_simulation(data, config):
    """Simulate preprocessing pipeline execution."""
    results = {
        'steps': [],
        'data_shape': [],
        'missing_values': [],
        'duplicates_removed': 0,
        'outliers_removed': 0,
        'features_created': 0,
        'execution_time': 0
    }
    
    df = data.copy()
    step = 1
    
    # Step 1: Data shape
    results['steps'].append(f"Step {step}: Initial Data Shape")
    results['data_shape'].append(df.shape)
    results['missing_values'].append(df.isnull().sum().sum())
    step += 1
    
    # Step 2: Handle missing values
    if config.get('handle_missing', False):
        results['steps'].append(f"Step {step}: Handle Missing Values")
        initial_missing = df.isnull().sum().sum()
        
        # Simulate missing value handling (type-aware and safe)
        strategy = config.get('missing_strategy')
        if strategy == 'drop':
            df = df.dropna()
        elif strategy in ['mean', 'median', 'most_frequent']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.columns.difference(numeric_cols)
            if strategy == 'mean':
                if len(numeric_cols) > 0:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == 'median':
                if len(numeric_cols) > 0:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            # For non-numeric (and for most_frequent on all), use per-column mode
            mode_fill_cols = non_numeric_cols if strategy in ['mean', 'median'] else df.columns
            for col in mode_fill_cols:
                try:
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                except Exception:
                    # If mode fails, skip the column
                    pass
        elif strategy == 'constant':
            df = df.fillna(config.get('fill_value', 0))
        
        results['data_shape'].append(df.shape)
        results['missing_values'].append(df.isnull().sum().sum())
        step += 1
    
    # Step 3: Remove duplicates
    if config.get('remove_duplicates', False):
        results['steps'].append(f"Step {step}: Remove Duplicates")
        initial_count = len(df)
        df = df.drop_duplicates()
        results['duplicates_removed'] = initial_count - len(df)
        results['data_shape'].append(df.shape)
        results['missing_values'].append(df.isnull().sum().sum())
        step += 1
    
    # Step 4: Outlier detection
    if config.get('outlier_detection', False):
        results['steps'].append(f"Step {step}: Outlier Detection")
        initial_count = len(df)
        
        # Simulate outlier removal
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if config.get('outlier_method') == 'zscore':
                z_scores = np.abs((df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std())
                df = df[(z_scores < config.get('z_threshold', 3.0)).all(axis=1)]
            elif config.get('outlier_method') == 'iqr':
                Q1 = df[numerical_cols].quantile(0.25)
                Q3 = df[numerical_cols].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[numerical_cols] < (Q1 - config.get('iqr_multiplier', 1.5) * IQR)) | 
                          (df[numerical_cols] > (Q3 + config.get('iqr_multiplier', 1.5) * IQR))).any(axis=1)]
            elif config.get('outlier_method') == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    # Prepare data for Isolation Forest (handle infinite values)
                    clean_data = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
                    clean_data = clean_data.fillna(clean_data.median())
                    
                    # Fit Isolation Forest
                    iso_forest = IsolationForest(
                        contamination=config.get('contamination', 0.1),  # Default 10% outliers
                        random_state=42,
                        n_estimators=100
                    )
                    outlier_labels = iso_forest.fit_predict(clean_data)
                    
                    # Remove outliers (label -1 indicates outlier)
                    df = df[outlier_labels == 1]
                except ImportError:
                    st.warning("scikit-learn not available for Isolation Forest")
                except Exception as e:
                    st.warning(f"Isolation Forest failed: {e}")
        
        results['outliers_removed'] = initial_count - len(df)
        results['data_shape'].append(df.shape)
        results['missing_values'].append(df.isnull().sum().sum())
        step += 1
    
    # Step 5: Feature engineering
    if config.get('create_features', False):
        results['steps'].append(f"Step {step}: Feature Engineering")
        # Simulate feature creation
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            numerical_cols = df.select_dtypes(include=[np.number]).columns[:2]
            df[f'{numerical_cols[0]}_plus_{numerical_cols[1]}'] = df[numerical_cols[0]] + df[numerical_cols[1]]
            df[f'{numerical_cols[0]}_times_{numerical_cols[1]}'] = df[numerical_cols[0]] * df[numerical_cols[1]]
            results['features_created'] = 2
        
        results['data_shape'].append(df.shape)
        results['missing_values'].append(df.isnull().sum().sum())
        step += 1
    
    # Final step
    results['steps'].append(f"Step {step}: Final Data Shape")
    results['data_shape'].append(df.shape)
    results['missing_values'].append(df.isnull().sum().sum())
    
    return results

def show_preprocessing_results(results):
    """Show preprocessing pipeline results."""
    st.subheader("📊 Preprocessing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Rows", results['data_shape'][0][0])
    with col2:
        st.metric("Final Rows", results['data_shape'][-1][0])
    with col3:
        st.metric("Duplicates Removed", results['duplicates_removed'])
    with col4:
        st.metric("Outliers Removed", results['outliers_removed'])
    
    # Step-by-step visualization
    st.subheader("📈 Preprocessing Pipeline Progress")
    
    # Create progress visualization
    steps_data = []
    for i, (step, shape, missing) in enumerate(zip(results['steps'], results['data_shape'], results['missing_values'])):
        steps_data.append({
            'Step': step,
            'Rows': shape[0],
            'Columns': shape[1],
            'Missing Values': missing,
            'Step Number': i + 1
        })
    
    steps_df = pd.DataFrame(steps_data)
    
    # Show data shape progression
    fig = px.line(steps_df, x='Step Number', y='Rows', 
                  title="Data Shape Progression Through Pipeline",
                  markers=True)
    fig.update_layout(xaxis_title="Pipeline Step", yaxis_title="Number of Rows")
    st.plotly_chart(fig, width='stretch', key="data_shape_progression")
    
    # Show missing values progression
    if any(missing > 0 for missing in results['missing_values']):
        fig2 = px.line(steps_df, x='Step Number', y='Missing Values',
                      title="Missing Values Progression Through Pipeline",
                      markers=True)
        fig2.update_layout(xaxis_title="Pipeline Step", yaxis_title="Missing Values")
        st.plotly_chart(fig2, width='stretch', key="missing_values_progression")
    
    # Show detailed results table
    st.subheader("📋 Detailed Pipeline Results")
    _safe_dataframe(steps_df)
    
    # Success message
    st.success(f"✅ Preprocessing pipeline completed! Final dataset: {results['data_shape'][-1][0]} rows × {results['data_shape'][-1][1]} columns")

def show_correlation_analysis(data):
    """Show correlation analysis for numerical data."""
    numerical_data = data.select_dtypes(include=[np.number])
    
    if len(numerical_data.columns) > 1:
        # Correlation matrix
        corr_matrix = numerical_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, width='stretch', key="correlation_matrix")
        
        # Show correlation values
        st.write("**Correlation Values:**")
        _safe_dataframe(corr_matrix.round(3))
        
        # Find highly correlated features with different thresholds
        high_corr_strong = []  # > 0.7
        high_corr_moderate = []  # > 0.5
        high_corr_weak = []  # > 0.3
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                feature_pair = {
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                }
                
                if abs(corr_val) > 0.7:
                    high_corr_strong.append(feature_pair)
                elif abs(corr_val) > 0.5:
                    high_corr_moderate.append(feature_pair)
                elif abs(corr_val) > 0.3:
                    high_corr_weak.append(feature_pair)
        
        # Show correlation analysis results
        st.subheader("🔍 Correlation Analysis Results")
        
        if high_corr_strong:
            st.warning("⚠️ **Strongly Correlated Features (>0.8):**")
            strong_corr_df = pd.DataFrame(high_corr_strong)
            _safe_dataframe(strong_corr_df)
        
        if high_corr_moderate:
            st.info("📊 **Moderately Correlated Features (0.6-0.8):**")
            moderate_corr_df = pd.DataFrame(high_corr_moderate)
            _safe_dataframe(moderate_corr_df)
        
        if high_corr_weak:
            st.info("📈 **Weakly Correlated Features (0.4-0.6):**")
            weak_corr_df = pd.DataFrame(high_corr_weak)
            _safe_dataframe(weak_corr_df)
        
        if not any([high_corr_strong, high_corr_moderate, high_corr_weak]):
            st.success("✅ No significant correlations detected (all < 0.3).")
        
        # Show correlation statistics
        st.subheader("📊 Correlation Statistics")
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Correlation", f"{np.mean(corr_values):.3f}")
        with col2:
            st.metric("Max Correlation", f"{np.max(corr_values):.3f}")
        with col3:
            st.metric("Min Correlation", f"{np.min(corr_values):.3f}")
        with col4:
            st.metric("Std Correlation", f"{np.std(corr_values):.3f}")
    else:
        st.warning("Need at least 2 numerical columns for correlation analysis.")

def show_sample_data(data):
    """Show sample data with options."""
    st.subheader("📋 Sample Data")
    
    # Show data info
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_size = st.slider("Sample Size", 5, min(1000, len(data)), 100)
    with col2:
        show_all = st.checkbox("Show All Data", value=False)
    with col3:
        show_info = st.checkbox("Show Data Info", value=True)
    
    # Show sample data
    if show_all:
        _safe_dataframe(data)
    else:
    _safe_dataframe(data.head(sample_size))
    
    if show_info:
        st.subheader("📊 Data Information")
        
        # Basic info
        buffer = io.StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        
        st.text(info_str)
        
        # Show data types
        st.write("**Data Types:**")
        dtype_counts = data.dtypes.astype(str).value_counts().rename_axis('dtype').reset_index(name='count')
        _safe_dataframe(dtype_counts)

def show_model_results(dashboard):
    """Show ML model results."""
    st.subheader("🤖 Model Results")
    
    if not dashboard.ml_results or not dashboard.ml_results.get("success"):
        st.info("No model results available. Train models first.")
        return
    
    results = dashboard.ml_results.get("results", [])
    
    if results:
        # Determine if this is classification or regression based on available metrics
        first_result = results[0]
        metrics = first_result.get("evaluation_metrics", {})
        
        # Check if we have classification or regression metrics
        is_classification = any(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1_score'])
        is_regression = any(key in metrics for key in ['mse', 'rmse', 'mae', 'r2'])
        
        if is_classification:
            # Create results dataframe for classification
            results_data = []
            for result in results:
                metrics = result.get("evaluation_metrics", {})
                results_data.append({
                    "Model": result.get("model_name", "Unknown"),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1-Score": metrics.get("f1_score", 0)
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Display results table safely
            _safe_dataframe(results_df)
            
            # Create comparison chart for classification
            fig = go.Figure()
            
            for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df["Model"],
                    y=results_df[metric]
                ))
            
            fig.update_layout(
                title="Model Performance Comparison (Classification)",
                barmode='group',
                xaxis_title="Models",
                yaxis_title="Score"
            )
            
            st.plotly_chart(fig, width='stretch')
            
        elif is_regression:
            # Create results dataframe for regression
            results_data = []
            for result in results:
                metrics = result.get("evaluation_metrics", {})
                results_data.append({
                    "Model": result.get("model_name", "Unknown"),
                    "R² Score": metrics.get("r2", 0),
                    "RMSE": metrics.get("rmse", 0),
                    "MAE": metrics.get("mae", 0),
                    "MSE": metrics.get("mse", 0)
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Display results table safely
            _safe_dataframe(results_df)
            
            # Create comparison chart for regression
            fig = go.Figure()
            
            for metric in ["R² Score", "RMSE", "MAE", "MSE"]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df["Model"],
                    y=results_df[metric]
                ))
            
            fig.update_layout(
                title="Model Performance Comparison (Regression)",
                barmode='group',
                xaxis_title="Models",
                yaxis_title="Score"
            )
            
            st.plotly_chart(fig, width='stretch', key="classification_performance")
        
        # Best model info
        if dashboard.ml_results.get("best_model"):
            st.success(f"🏆 Best Model: {dashboard.ml_results['best_model']}")
            st.success(f"📈 Best Score: {dashboard.ml_results.get('best_score', 0):.4f}")
            
            # Show additional info about the task type
            if is_classification:
                st.info("📊 Task Type: Classification")
            elif is_regression:
                st.info("📊 Task Type: Regression")

def show_actionable_insights(dashboard):
    """Show actionable business recommendations and insights."""
    st.subheader("🎯 Actionable Business Insights")
    
    if not dashboard.data_cache:
        st.info("No data available for insights. Please load data first.")
        return
    
    # Get the actual uploaded data
    data_name = list(dashboard.data_cache.keys())[0]
    data = dashboard.data_cache[data_name]
    
    st.success(f"✅ Analyzing uploaded data: {data_name}")
    
    # Check if this is healthcare data
    is_healthcare = any(keyword in data_name.lower() for keyword in ['health', 'medical', 'patient', 'billing'])
    is_financial = any(keyword in data_name.lower() for keyword in ['financial', 'bank', 'transaction', 'payment'])
    is_ecommerce = any(keyword in data_name.lower() for keyword in ['ecommerce', 'customer', 'product', 'order'])
    
    # Data Overview
    st.write("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(data.columns):,}")
    
    with col3:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", f"{numeric_cols:,}")
    
    with col4:
        missing_values = data.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    # Show data type distribution
    st.write("### 🔧 Data Types")
    dtype_counts = data.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Data Types Distribution"
    )
    st.plotly_chart(fig, width='stretch', key="actionable_insights_data_types")
    
    # Show sample data
    st.write("### 📋 Sample Data")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_all_data = st.checkbox("Show All Data", value=False, key="business_insights_show_all")
    with col2:
        sample_size = st.slider("Sample Size", 10, min(500, len(data)), 50, key="business_insights_sample")
    
    if show_all_data:
        _safe_dataframe(data)
    else:
        _safe_dataframe(data.head(sample_size))
    
    # Data-specific insights based on type
    if is_healthcare:
        st.write("### 🏥 Healthcare Data Insights")
        st.info("""
        **Healthcare Data Detected!** This appears to be medical/patient data.
        
        **Key Areas to Analyze:**
        - Patient demographics and patterns
        - Medical billing and revenue cycles
        - Treatment outcomes and effectiveness
        - Resource utilization and efficiency
        - Compliance and regulatory requirements
        """)
        
        # Show healthcare-specific columns if available
        health_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['patient', 'medical', 'diagnosis', 'treatment', 'billing', 'insurance'])]
        if health_cols:
            st.write("**Healthcare-related columns found:**")
            for col in health_cols:
                st.write(f"• {col}")
    
    elif is_financial:
        st.write("### 💰 Financial Data Insights")
        st.info("""
        **Financial Data Detected!** This appears to be banking/financial data.
        
        **Key Areas to Analyze:**
        - Transaction patterns and fraud detection
        - Customer behavior and segmentation
        - Risk assessment and compliance
        - Revenue optimization and growth
        - Operational efficiency metrics
        """)
    
    elif is_ecommerce:
        st.write("### 🛒 E-commerce Data Insights")
        st.info("""
        **E-commerce Data Detected!** This appears to be customer/product data.
        
        **Key Areas to Analyze:**
        - Customer lifetime value and retention
        - Product performance and inventory
        - Sales trends and seasonality
        - Marketing effectiveness and ROI
        - Customer satisfaction and feedback
        """)
    
    else:
        st.write("### 📊 General Data Insights")
        st.info("""
        **General Dataset Detected!** Analyzing for common patterns and insights.
        
        **Key Areas to Analyze:**
        - Data quality and completeness
        - Statistical patterns and distributions
        - Outliers and anomalies
        - Correlation and relationships
        - Business value and applications
        """)
    
    # Data Analysis and Insights
    st.write("### 🔍 Data Analysis and Insights")
    
    # Show key statistics for numerical columns
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.write("**Numerical Column Statistics:**")
        numeric_stats = data.describe()
        _safe_dataframe(numeric_stats)
        
        # Show correlation matrix for numerical columns
        if len(data.select_dtypes(include=[np.number]).columns) > 1:
            st.write("**Correlation Matrix (Top 10):**")
            # Filter only numerical columns for correlation
            numeric_data = data.select_dtypes(include=[np.number])
            try:
                corr_matrix = numeric_data.corr()
                # Get top correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Show moderate correlations
                            corr_pairs.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_val:.3f}"
                            })
                
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=lambda x: abs(pd.to_numeric(x)), ascending=False)
                    st.write(f"**All {len(corr_df)} correlations:**")
                    _safe_dataframe(corr_df)
                else:
                    st.info("No significant correlations found between numerical features.")
            except Exception as e:
                st.warning(f"⚠️ Could not calculate correlations: {str(e)}")
                st.info("This usually happens when there are non-numeric values in columns that appear numeric.")
    
    # Show categorical column insights
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.write("**Categorical Column Insights:**")
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if data[col].nunique() < 50:  # Show more unique values
                value_counts = data[col].value_counts()
                st.write(f"**{col}** (All {len(value_counts)} values):")
                _safe_dataframe(pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values}))
    
    # Recommendations based on data type
    st.write("### 🎯 Data-Specific Recommendations")
    
    if is_healthcare:
        recommendations = [
            "🏥 **Patient Data Analysis**: Analyze patient demographics and treatment patterns",
            "💰 **Billing Optimization**: Review revenue cycles and payment patterns",
            "📊 **Resource Planning**: Optimize resource allocation based on patient volumes",
            "🔒 **Compliance Check**: Ensure HIPAA compliance and data security",
            "📈 **Quality Metrics**: Track treatment outcomes and patient satisfaction"
        ]
    elif is_financial:
        recommendations = [
            "💰 **Risk Assessment**: Analyze transaction patterns for fraud detection",
            "👥 **Customer Segmentation**: Identify high-value customer segments",
            "📊 **Performance Metrics**: Track key financial KPIs and trends",
            "🔒 **Compliance Monitoring**: Ensure regulatory compliance",
            "📈 **Growth Opportunities**: Identify revenue optimization strategies"
        ]
    elif is_ecommerce:
        recommendations = [
            "🛒 **Customer Behavior**: Analyze purchase patterns and preferences",
            "📊 **Product Performance**: Identify top-performing products and categories",
            "💰 **Revenue Optimization**: Optimize pricing and inventory management",
            "📈 **Marketing ROI**: Track campaign effectiveness and customer acquisition",
            "👥 **Retention Strategy**: Develop customer loyalty programs"
        ]
    else:
        recommendations = [
            "📊 **Data Quality**: Assess completeness and accuracy of your data",
            "🔍 **Pattern Analysis**: Identify trends and anomalies in your data",
            "📈 **Business Value**: Determine how this data can drive business decisions",
            "🔄 **ETL Optimization**: Optimize data processing pipelines",
            "🤖 **AI Integration**: Consider AI-powered insights and automation"
        ]
    
    for rec in recommendations:
        st.info(rec)
    
    # Next steps
    st.write("### 🚀 Next Steps")
    st.markdown("""
    1. **Use AI Insights Tab**: Get AI-powered analysis of your specific data
    2. **Run ETL Pipeline**: Clean and transform your data for better analysis
    3. **Explore Visualizations**: Use other tabs to understand your data better
    4. **Export Results**: Save insights and recommendations for stakeholders
    """)

def show_ai_insights(dashboard):
    """Show AI-powered insights using Groq API."""
    st.subheader("🤖 AI-Powered Insights")
    
    if not dashboard.data_cache:
        st.warning("No data loaded. Please load data first.")
        return
    
    # Check if Groq is available
    if not GROQ_AVAILABLE:
        st.error("❌ Groq package not installed. Install with: `pip install groq`")
        return
    
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
        st.info("💡 Get your API key from: https://console.groq.com/")
        return
    
    # Get the first dataset for analysis
    data_name = list(dashboard.data_cache.keys())[0]
    data = dashboard.data_cache[data_name]
    
    st.info(f"🤖 AI analyzing dataset: {data_name}")
    
    # AI Analysis Options
    st.subheader("🎯 Choose AI Analysis Type")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Data Quality Analysis", type="primary", width='stretch'):
            with st.spinner("🤖 AI analyzing data quality..."):
                analysis = analyze_data_with_groq(data, "quality")
                st.subheader("🔍 AI Data Quality Analysis")
                st.markdown(analysis)
    
    with col2:
        if st.button("💡 Business Insights", type="primary", width='stretch'):
            with st.spinner("🤖 AI generating business insights..."):
                analysis = analyze_data_with_groq(data, "insights")
                st.subheader("💡 AI Business Insights")
                st.markdown(analysis)
    
    with col3:
        if st.button("📊 General Analysis", type="primary", width='stretch'):
            with st.spinner("🤖 AI performing general analysis..."):
                analysis = analyze_data_with_groq(data, "general")
                st.subheader("📊 AI General Analysis")
                st.markdown(analysis)
    
    # ETL Recommendations
    st.subheader("🔄 AI ETL Pipeline Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Get ETL Recommendations", type="secondary", width='stretch'):
            with st.spinner("🤖 AI generating ETL recommendations..."):
                recommendations = generate_etl_recommendations_with_groq(data)
                st.subheader("🚀 AI ETL Recommendations")
                st.markdown(recommendations)
    
    with col2:
        if st.button("📋 Executive Summary", type="secondary", width='stretch'):
            with st.spinner("🤖 AI creating executive summary..."):
                summary = get_ai_data_summary_with_groq(data)
                st.subheader("📋 AI Executive Summary")
                st.markdown(summary)
    
    # AI Status and Configuration
    st.subheader("⚙️ AI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"🤖 **Model**: llama-3.1-8b-instant")
        st.info(f"🌡️ **Temperature**: 0.2-0.3 (Focused)")
        st.info(f"📝 **Max Tokens**: 800-1200")
    
    with col2:
        st.success(f"✅ **Groq API**: Available")
        st.success(f"🔑 **API Key**: Configured")
        st.success(f"📊 **Dataset**: {data.shape[0]:,} rows × {data.shape[1]} columns")
    
    # Usage Tips
    st.subheader("💡 AI Usage Tips")
    st.markdown("""
    - **Data Quality**: Best for identifying data issues and cleaning recommendations
    - **Business Insights**: Great for discovering patterns and opportunities
    - **ETL Recommendations**: Perfect for pipeline optimization suggestions
    - **Executive Summary**: Ideal for stakeholder presentations
    
    **Note**: Each analysis costs API credits. Use strategically for important insights.
    """)

def show_business_insights(dashboard):
    """Show comprehensive business intelligence dashboard."""
    st.subheader("💼 Business Intelligence Dashboard")
    
    if not dashboard.data_cache:
        st.info("No data available for business insights. Please load data first.")
        return
    
    # Get the actual uploaded data
    data_name = list(dashboard.data_cache.keys())[0]
    data = dashboard.data_cache[data_name]
    
    st.success(f"✅ Analyzing uploaded data: {data_name}")
    
    # Check if this is healthcare data
    is_healthcare = any(keyword in data_name.lower() for keyword in ['health', 'medical', 'patient', 'billing'])
    is_financial = any(keyword in data_name.lower() for keyword in ['financial', 'bank', 'transaction', 'payment'])
    is_ecommerce = any(keyword in data_name.lower() for keyword in ['ecommerce', 'customer', 'product', 'order'])
    
    # Data Overview and KPIs
    st.write("### 📊 Data Overview and Key Metrics")
    
    # Basic data metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(data.columns):,}")
    
    with col3:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", f"{numeric_cols:,}")
    
    with col4:
        missing_values = data.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    # Data Type Analysis
    st.write("### 🔧 Data Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data types distribution
        dtype_counts = data.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, width='stretch', key="data_types_distribution")
    
    with col2:
        # Missing values by column
        missing_by_col = data.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0]
        if len(missing_by_col) > 0:
        fig = px.bar(
                x=missing_by_col.index,
                y=missing_by_col.values,
                title="Missing Values by Column",
                labels={'x': 'Column', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, width='stretch', key="business_insights_missing_values")
        else:
            st.success("✅ No missing values found!")
    
    # Numerical Data Analysis
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.write("### 📈 Numerical Data Analysis")
        
        # Show descriptive statistics
        numeric_stats = data.describe()
        st.write("**Descriptive Statistics:**")
        _safe_dataframe(numeric_stats)
        
        # Show correlation heatmap for numerical columns
        if len(data.select_dtypes(include=[np.number]).columns) > 1:
            try:
                st.write("**Correlation Heatmap:**")
                # Filter only numerical columns for correlation
                numeric_data = data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto"
                )
                st.plotly_chart(fig, width='stretch', key="business_insights_correlation")
                
                # Find strong correlations
                strong_correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_correlations.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_val:.3f}"
                            })
                
                if strong_correlations:
                    st.write("**Strong Correlations (>0.7):**")
                    _safe_dataframe(pd.DataFrame(strong_correlations))
                else:
                    st.info("No significant correlations found between numerical features.")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not calculate correlations: {str(e)}")
                st.info("This usually happens when there are non-numeric values in columns that appear numeric.")
        else:
            st.info("📊 Need at least 2 numerical columns to calculate correlations.")
    
    # Categorical Data Analysis
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.write("### 📝 Categorical Data Analysis")
        
        # Show value counts for categorical columns (limit to first 5 to avoid overwhelming)
        for col in categorical_cols[:5]:
            if data[col].nunique() < 50:  # Show more unique values
                value_counts = data[col].value_counts()
                st.write(f"**{col}** (All {len(value_counts)} values):")
                _safe_dataframe(pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values}))
                
                # Show pie chart for categorical data
                if len(value_counts) <= 10:
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {col}"
                    )
                    st.plotly_chart(fig, width='stretch', key=f"business_insights_categorical_{col}")
    
    # Data Quality Assessment
    st.write("### 🔍 Data Quality Assessment")
    
    quality_metrics = {
        "Completeness": f"{((len(data) * len(data.columns) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%",
        "Duplicates": f"{data.duplicated().sum():,}",
        "Unique Records": f"{len(data.drop_duplicates()):,}",
        "Memory Usage": f"{data.memory_usage(deep=True).sum() / (1024*1024):.1f} MB"
    }
    
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(quality_metrics.items()):
        with col1 if i < 2 else col2:
            st.metric(metric, value)
    
    # Data-Specific Insights
    st.write("### 🎯 Data-Specific Insights")
    
    if is_healthcare:
        st.info("""
        **🏥 Healthcare Data Detected!**
        
        **Key Analysis Areas:**
        - Patient demographics and patterns
        - Medical billing and revenue cycles
        - Treatment outcomes and effectiveness
        - Resource utilization and efficiency
        - Compliance and regulatory requirements
        
        **Recommended Actions:**
        - Analyze patient visit patterns
        - Review billing accuracy and compliance
        - Assess resource allocation efficiency
        - Monitor quality metrics and outcomes
        """)
        
        # Show healthcare-specific columns if available
        health_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['patient', 'medical', 'diagnosis', 'treatment', 'billing', 'insurance', 'visit', 'doctor'])]
        if health_cols:
            st.write("**🏥 Healthcare-related columns found:**")
            for col in health_cols:
                st.write(f"• {col}")
    
    elif is_financial:
        st.info("""
        **💰 Financial Data Detected!**
        
        **Key Analysis Areas:**
        - Transaction patterns and fraud detection
        - Customer behavior and segmentation
        - Risk assessment and compliance
        - Revenue optimization and growth
        - Operational efficiency metrics
        
        **Recommended Actions:**
        - Monitor transaction patterns for anomalies
        - Analyze customer spending behaviors
        - Assess risk and compliance metrics
        - Optimize revenue and growth strategies
        """)
    
    elif is_ecommerce:
        st.info("""
        **🛒 E-commerce Data Detected!**
        
        **Key Analysis Areas:**
        - Customer lifetime value and retention
        - Product performance and inventory
        - Sales trends and seasonality
        - Marketing effectiveness and ROI
        - Customer satisfaction and feedback
        
        **Recommended Actions:**
        - Analyze customer purchase patterns
        - Optimize product inventory and pricing
        - Track marketing campaign performance
        - Improve customer retention strategies
        """)
    
    else:
        st.info("""
        **📊 General Dataset Detected!**
        
        **Key Analysis Areas:**
        - Data quality and completeness
        - Statistical patterns and distributions
        - Outliers and anomalies
        - Correlation and relationships
        - Business value and applications
        
        **Recommended Actions:**
        - Assess data quality and completeness
        - Identify patterns and trends
        - Detect outliers and anomalies
        - Explore correlations and relationships
        """)
    
    # Next Steps
    st.write("### 🚀 Next Steps")
    st.markdown("""
    1. **Use AI Insights Tab**: Get AI-powered analysis of your specific data
    2. **Run ETL Pipeline**: Clean and transform your data for better analysis
    3. **Explore Visualizations**: Use other tabs to understand your data better
    4. **Export Results**: Save insights and recommendations for stakeholders
    """)

def show_etl_results(dashboard):
    """Show comprehensive ETL pipeline results and data transformation summary."""
    st.subheader("🔄 ETL Pipeline Results & Data Transformation")
    
    if not dashboard.data_cache:
        st.info("No ETL pipeline results available. Please load data first.")
        return
    
    # Get the actual uploaded data
    data_name = list(dashboard.data_cache.keys())[0]
    data = dashboard.data_cache[data_name]
    
    st.success(f"✅ Analyzing ETL pipeline data: {data_name}")
    
    # Data Pipeline Summary
    st.write("### 📊 Data Pipeline Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        st.metric("Total Columns", len(data.columns))
    
    with col3:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        missing_values = data.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    
    # Data Quality Metrics
    st.write("### 🔍 Data Quality Metrics")
    
    # Missing values analysis by column
    missing_by_col = data.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]
    
    if len(missing_by_col) > 0:
        st.write("**Missing Values by Column:**")
        missing_df = pd.DataFrame({
            'Column': missing_by_col.index,
            'Missing Count': missing_by_col.values,
            'Missing Percentage': (missing_by_col.values / len(data)) * 100
    }).round(2)
        _safe_dataframe(missing_df)
        
        # Missing values chart
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Count',
            title="Missing Values by Column",
            color='Missing Percentage',
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, width='stretch', key="etl_results_missing_values")
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Data transformation summary
    st.write("### 🔄 Data Transformation Summary")
    
    # Data quality metrics
    quality_metrics = {
        "Completeness": f"{((len(data) * len(data.columns) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%",
        "Duplicates": f"{data.duplicated().sum():,}",
        "Unique Records": f"{len(data.drop_duplicates()):,}",
        "Memory Usage": f"{data.memory_usage(deep=True).sum() / (1024*1024):.1f} MB"
    }
    
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(quality_metrics.items()):
        with col1 if i < 2 else col2:
            st.metric(metric, value)
    
    # Data type analysis
    st.write("### 🔧 Data Type Analysis")
    
    dtype_summary = pd.DataFrame({
        'Data Type': data.dtypes.value_counts().index.astype(str),
        'Column Count': data.dtypes.value_counts().values
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types Summary:**")
        _safe_dataframe(dtype_summary)
    
    with col2:
        # Data types pie chart
        fig = px.pie(
            dtype_summary,
            values='Column Count',
            names='Data Type',
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, width='stretch', key="etl_results_data_types")
    
    # Sample data
    st.write("### 📋 Sample Data")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_all_data = st.checkbox("Show All Data", value=False, key="etl_results_show_all")
    with col2:
        sample_size = st.slider("Sample Size", 10, min(500, len(data)), 50, key="etl_results_sample")
    
    if show_all_data:
        _safe_dataframe(data)
    else:
        _safe_dataframe(data.head(sample_size))
    
    # Next steps
    st.write("### 🚀 Next Steps")
    st.markdown("""
    1. **Run Preprocessing**: Use the preprocessing tools to clean your data
    2. **Feature Engineering**: Create new features for better analysis
    3. **AI Insights**: Get AI-powered analysis of your data
    4. **Export Results**: Save processed data for further use
    """)

def show_advanced_analytics(dashboard):
    """Show advanced analytics including predictive insights and complex patterns."""
    st.subheader("📈 Advanced Analytics & Predictive Insights")
    
    if not dashboard.data_cache:
        st.info("No data available for advanced analytics. Please load data first.")
        return
    
    # Get the actual uploaded data
    data_name = list(dashboard.data_cache.keys())[0]
    data = dashboard.data_cache[data_name]
    
    st.success(f"✅ Analyzing data for advanced analytics: {data_name}")
    
    # Data Overview
    st.write("### 📊 Data Overview for Advanced Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        st.metric("Total Columns", len(data.columns))
    
    with col3:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        categorical_cols = len(data.select_dtypes(include=['object', 'category']).columns)
        st.metric("Categorical Columns", categorical_cols)
    
    # Advanced Statistical Analysis
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.write("### 🔬 Advanced Statistical Analysis")
        
        # Descriptive statistics
        numeric_data = data.select_dtypes(include=[np.number])
        st.write("**Comprehensive Statistics:**")
        _safe_dataframe(numeric_data.describe())
        
        # Correlation analysis
        if len(numeric_data.columns) > 1:
            st.write("**Correlation Analysis:**")
            corr_matrix = numeric_data.corr()
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, width='stretch', key="advanced_analytics_correlation")
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })
            
            if strong_correlations:
                st.write("**Strong Correlations (>0.7):**")
                _safe_dataframe(pd.DataFrame(strong_correlations))
    
    # Pattern Detection
    st.write("### 🔍 Pattern Detection & Insights")
    
    # Check for outliers in numerical columns
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.write("**Outlier Detection:**")
        
        outlier_summary = []
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': outlier_count,
                    'Outlier Percentage': f"{(outlier_count / len(data)) * 100:.1f}%"
                })
        
        if outlier_summary:
            st.write("**Outliers Found:**")
            _safe_dataframe(pd.DataFrame(outlier_summary))
        else:
            st.success("✅ No significant outliers detected!")
    
    # Categorical pattern analysis
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.write("**Categorical Pattern Analysis:**")
        
        for col in categorical_cols[:3]:  # Limit to first 3 to avoid overwhelming
            if data[col].nunique() < 20:
                value_counts = data[col].value_counts()
                st.write(f"**{col} Distribution:**")
    
    col1, col2 = st.columns(2)
    with col1:
                    _safe_dataframe(pd.DataFrame({
                        'Value': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': (value_counts.values / len(data)) * 100
                    }).round(2))
    
    with col2:
        fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Distribution of {col}"
                    )
                    st.plotly_chart(fig, width='stretch', key=f"advanced_analytics_categorical_{col}")
    
    # Data Quality Insights
    st.write("### 🔍 Data Quality Insights")
    
    quality_insights = {
        "Completeness": f"{((len(data) * len(data.columns) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100):.1f}%",
        "Duplicates": f"{data.duplicated().sum():,}",
        "Memory Efficiency": f"{data.memory_usage(deep=True).sum() / (1024*1024):.1f} MB",
        "Column Types": f"{len(data.dtypes.unique())} unique types"
    }
    
    col1, col2 = st.columns(2)
    for i, (metric, value) in enumerate(quality_insights.items()):
        with col1 if i < 2 else col2:
            st.metric(metric, value)
    
    # Recommendations
    st.write("### 💡 Advanced Analytics Recommendations")
    
    recommendations = [
        "🔬 **Statistical Modeling**: Consider regression or classification models for numerical data",
        "📊 **Time Series Analysis**: If date columns exist, analyze temporal patterns",
        "🎯 **Clustering**: Use K-means or hierarchical clustering for customer segmentation",
        "📈 **Predictive Modeling**: Train ML models for forecasting and predictions",
        "🔍 **Anomaly Detection**: Implement automated outlier detection systems"
    ]
    
    for rec in recommendations:
        st.info(rec)
    
    # Next steps
    st.write("### 🚀 Next Steps")
    st.markdown("""
    1. **Use AI Insights**: Get AI-powered analysis and recommendations
    2. **Run ML Models**: Train predictive models on your data
    3. **Export Insights**: Save analysis results for stakeholders
    4. **Monitor Trends**: Set up automated monitoring and alerts
    """)
    
    # Data-Driven Insights
    st.write("### 🔮 Data-Driven Insights")
    
    # Show insights based on data characteristics
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.info("""
        **📊 Numerical Data Detected!**
        
        **Available Analysis:**
        - Statistical summaries and distributions
        - Correlation analysis between features
        - Outlier detection and analysis
        - Trend analysis and patterns
        """)
    
    if len(data.select_dtypes(include=['object', 'category']).columns) > 0:
        st.info("""
        **📝 Categorical Data Detected!**
        
        **Available Analysis:**
        - Value distribution analysis
        - Pattern recognition
        - Category-based insights
        - Frequency analysis
        """)
    
    # Time series detection
    date_columns = [col for col in data.columns if data[col].dtype == 'object' and 
                   data[col].str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any()]
    
    if date_columns:
        st.info(f"""
        **📅 Time Series Data Detected!**
        
        **Date Columns Found:** {', '.join(date_columns)}
        
        **Available Analysis:**
        - Temporal pattern analysis
        - Seasonal trend detection
        - Time-based forecasting
        - Chronological insights
        """)
    


if __name__ == "__main__":
    main()
