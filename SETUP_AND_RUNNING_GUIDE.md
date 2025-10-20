# ETL + ML Pipeline: Setup & Running Guide

## 📁 Project Folder Structure

```
ETL Pipelines/
├── 📁 src/                          # Core application source code
│   ├── 📁 core/                     # Core pipeline components
│   │   ├── config.py                # Configuration management
│   │   ├── pipeline.py              # Main pipeline orchestration
│   │   ├── logger.py                # Logging functionality
│   │   └── exceptions.py            # Custom exception classes
│   ├── 📁 extractors/               # Data extraction modules
│   │   ├── base.py                  # Base extractor class
│   │   ├── csv_extractor.py         # CSV file extraction
│   │   └── json_extractor.py        # JSON file extraction
│   ├── 📁 transformers/             # Data transformation modules
│   │   ├── base.py                  # Base transformer class
│   │   ├── data_cleaner.py          # Data cleaning operations
│   │   └── data_validator.py        # Data validation logic
│   ├── 📁 loaders/                  # Data loading modules
│   │   ├── base.py                  # Base loader class
│   │   ├── csv_loader.py            # CSV file output
│   │   └── json_loader.py           # JSON file output
│   └── __init__.py                  # Package initialization
├── 📁 config/                       # Pipeline configuration files
│   ├── pipeline_config.yaml         # Main pipeline configuration
│   ├── ecommerce_pipeline.yaml      # E-commerce specific config
│   ├── financial_timeseries_pipeline.yaml  # Financial data config
│   └── customer_cleaning_pipeline.yaml     # Customer data config
├── 📁 data/                         # Data storage directories
│   ├── 📁 input/                    # Input data files
│   ├── 📁 output/                   # Processed data output
│   ├── 📁 unified/                  # Combined unified datasets
│   └── 📁 temp/                     # Temporary processing files
├── 📁 .venv/                        # Python virtual environment
├── 🐍 main.py                       # Command-line entry point
├── 🐍 dashboard.py                  # Streamlit web dashboard
├── 📋 requirements.txt              # Python dependencies
├── 📋 .env                          # Environment variables (create this)
├── 📋 README.md                     # Project overview
├── 📋 BUSINESS_PROBLEM_AND_SOLUTION.md  # Business case document
└── 📋 SETUP_AND_RUNNING_GUIDE.md   # This setup guide
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+** installed on your system
- **Git** for version control (optional)
- **Windows 10/11** (tested on Windows 10.0.26100)

### Step 1: Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd "D:\Salman AI\ETL Pipelines"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

#### Option B: Using Conda (Alternative)
```bash
# Create conda environment
conda create -n etl_ml_pipeline python=3.9

# Activate environment
conda activate etl_ml_pipeline
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or install individually if you prefer:
pip install pandas pyyaml python-dotenv streamlit plotly scikit-learn groq
```

### Step 3: Environment Configuration
```bash
# Create .env file for API keys (optional but recommended for AI features)
# Create a new file called .env in the project root

# Add your Groq API key for AI insights:
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🎯 Running the Application

### Option 1: Web Dashboard (Recommended for Beginners)

#### Start the Streamlit Dashboard
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate

# Launch the dashboard
streamlit run dashboard.py
```

#### Dashboard Features
- **Data Upload & Management**: Upload CSV, JSON, Excel, Parquet files
- **Unified Data Access**: Load from pre-combined datasets in `data/unified/`
- **Data Quality Assessment**: Automated outlier detection and data profiling
- **Interactive Visualizations**: Plotly charts for data exploration
- **AI-Powered Insights**: Groq AI integration for intelligent data summaries
- **Business Intelligence**: Domain-specific insights for healthcare, financial, ecommerce data
- **ETL Pipeline Execution**: Run configurable data processing workflows

#### Dashboard Access
- **Local URL**: http://localhost:8501 (or 8502, 8503 if port is busy)
- **Network URL**: http://your_ip:8501 (for remote access)

### Option 2: Command Line Interface

#### Run Basic ETL Pipeline
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate

# Run the main pipeline
python main.py
```

#### Run with Custom Configuration
```bash
# Run specific pipeline configuration
python main.py --config config/ecommerce_pipeline.yaml

# Run with custom parameters
python main.py --source data/input/ --output data/output/
```

#### Available Pipeline Configurations
- `config/pipeline_config.yaml` - Main configuration
- `config/ecommerce_pipeline.yaml` - E-commerce data processing
- `config/financial_timeseries_pipeline.yaml` - Financial data processing
- `config/customer_cleaning_pipeline.yaml` - Customer data cleaning

---

## ⚙️ Configuration Management

### Pipeline Configuration Files

#### Main Configuration (`config/pipeline_config.yaml`)
```yaml
pipeline:
  name: "Universal ETL Pipeline"
  version: "1.0.0"
  
extract:
  source_type: "file"
  source_path: "data/input/"
  file_pattern: "*.csv"
  
transform:
  cleaning:
    remove_duplicates: true
    handle_missing: "drop"
  validation:
    schema_validation: true
    
load:
  output_type: "parquet"
  output_path: "data/output/"
  compression: "snappy"
```

#### E-commerce Pipeline (`config/ecommerce_pipeline.yaml`)
```yaml
pipeline:
  name: "E-commerce Data Pipeline"
  
extract:
  source_type: "file"
  source_path: "data/input/"
  file_pattern: "*.csv"
  
transform:
  cleaning:
    remove_duplicates: true
    handle_missing: "drop"
  validation:
    schema_validation: true
    
load:
  output_type: "parquet"
  output_path: "data/output/"
  compression: "snappy"
```

### Environment Variables (`.env`)
```bash
# API Keys (Optional but recommended for AI features)
GROQ_API_KEY=your_groq_api_key_here

# Logging Configuration (Optional)
LOG_LEVEL=INFO
LOG_FILE=logs/pipeline.log
```

---

## 🔧 Advanced Configuration

### Data Processing Workflow

#### File-Based Processing
The current implementation supports file-based data processing with the following workflow:

1. **Extract**: Load data from CSV, JSON, Excel, or Parquet files
2. **Transform**: Clean, validate, and preprocess data
3. **Load**: Save processed data to output directory

#### Supported File Formats
- **CSV**: Comma-separated values
- **JSON**: JavaScript Object Notation
- **Excel**: .xlsx and .xls files
- **Parquet**: Columnar storage format

#### Data Quality Features
- **Outlier Detection**: Z-score, IQR, and Isolation Forest algorithms
- **Missing Value Handling**: Drop, fill, or interpolate missing data
- **Data Validation**: Schema validation and type checking
- **Duplicate Removal**: Automatic duplicate detection and removal

---

## 📊 Data Processing Examples

### Example 1: CSV Data Processing
```bash
# 1. Place your CSV file in data/input/
# 2. Run the pipeline with default configuration
python main.py

# 3. Or use a specific configuration
python main.py --config config/ecommerce_pipeline.yaml
```

### Example 2: Using the Dashboard
```bash
# 1. Start the dashboard
streamlit run dashboard.py

# 2. Open browser to http://localhost:8501
# 3. Upload your data file using the sidebar
# 4. Explore data with interactive visualizations
# 5. Run ETL pipeline through the dashboard
```

### Example 3: Working with Unified Data
```bash
# 1. Access pre-combined datasets in data/unified/
# 2. Use the dashboard "Unified Data" tab
# 3. Select from available combined datasets:
#    - comprehensive_healthcare_data.csv
#    - comprehensive_financial_data.csv
#    - comprehensive_ecommerce_data.csv
#    - comprehensive_products_data.csv
```

---

## 🤖 AI Integration & Data Analysis

### Groq AI Integration

#### AI-Powered Insights
The dashboard includes AI integration using Groq API for intelligent data analysis:

1. **Data Summaries**: Get AI-generated summaries of your datasets
2. **Business Insights**: Receive domain-specific recommendations
3. **ETL Recommendations**: AI suggestions for data processing steps
4. **Actionable Insights**: Get next steps for data analysis

#### Setup AI Features
```bash
# 1. Create .env file in project root
# 2. Add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

# 3. Restart the dashboard to enable AI features
streamlit run dashboard.py
```

#### Using AI Features in Dashboard
1. **Upload or Load Data**: Use the data upload or unified data tabs
2. **Navigate to AI Insights**: Click on the "🤖 AI Insights" tab
3. **Get AI Analysis**: Click "Generate AI Insights" button
4. **Review Recommendations**: Read AI-generated insights and suggestions

### Data Quality Analysis

#### Outlier Detection
The dashboard provides multiple outlier detection methods:

- **Z-Score Method**: Statistical outlier detection
- **IQR Method**: Interquartile range-based detection
- **Isolation Forest**: Machine learning-based anomaly detection

#### Data Profiling
- **Statistical Summary**: Mean, median, standard deviation
- **Data Types**: Automatic detection of numerical and categorical columns
- **Missing Values**: Identification and handling of null values
- **Data Distribution**: Histograms and distribution analysis

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Issue 1: Import Errors
```bash
# Problem: ModuleNotFoundError or ImportError
# Solution: Ensure virtual environment is activated
.venv\Scripts\activate

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall requirements if needed
pip install -r requirements.txt --force-reinstall
```

#### Issue 2: Missing Dependencies
```bash
# Problem: Package not found
# Solution: Install missing packages
pip install package_name

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### Issue 3: Port Already in Use
```bash
# Problem: Port 8501 already in use
# Solution: Use different port
streamlit run dashboard.py --server.port 8502

# Or kill existing process (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

#### Issue 4: Groq API Errors
```bash
# Problem: Groq API not working
# Solution: Check your API key in .env file
# Make sure GROQ_API_KEY=your_actual_key_here

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key loaded:', bool(os.getenv('GROQ_API_KEY')))"
```

#### Issue 5: Data Loading Issues
```bash
# Problem: Can't load data files
# Solution: Check file paths and permissions
# Ensure data files are in data/input/ or data/unified/ directories
```

### Debug Mode
```bash
# Enable debug logging
set LOG_LEVEL=DEBUG

# Run with verbose output
python main.py --verbose --debug

# Check logs (if logs directory exists)
# tail -f logs/pipeline.log
```

---

## 📈 Performance Tips

### Data Processing Optimization

#### 1. **File Size Management**
- **Large Files**: Process files in chunks if memory is limited
- **File Formats**: Use Parquet for better compression and performance
- **Data Types**: Optimize data types to reduce memory usage

#### 2. **Dashboard Performance**
- **Data Caching**: Dashboard caches loaded data for faster navigation
- **Visualization Limits**: Large datasets may show sample data for performance
- **Memory Usage**: Monitor memory usage with very large datasets

#### 3. **AI API Usage**
- **Rate Limits**: Groq API has rate limits, use sparingly for large datasets
- **Data Sampling**: AI insights work best with representative data samples
- **API Key Management**: Keep your Groq API key secure and rotate regularly

---

## 🔒 Security Considerations

### API Key Management
- **Never commit API keys** to version control
- **Use .env file** for sensitive data (already in .gitignore)
- **Keep keys secure** and don't share them
- **Rotate keys regularly** for production use

### Data Privacy
- **Local Processing**: All data processing happens locally on your machine
- **No Data Transmission**: Your data files are not sent to external servers (except for AI insights)
- **AI Data**: Only data summaries are sent to Groq API for insights
- **File Permissions**: Ensure proper file permissions for sensitive data

### Network Security
- **Local Access Only**: Dashboard runs locally by default
- **Network Access**: Use network URL only in trusted environments
- **HTTPS**: Consider using HTTPS for production deployments

---

## 📚 Additional Resources

### Documentation
- **Source Code**: Check Python docstrings for detailed API information
- **Configuration Examples**: See config/ directory for pipeline examples
- **Business Case**: Read BUSINESS_PROBLEM_AND_SOLUTION.md for use cases

### Getting Help
- **Troubleshooting**: Review the troubleshooting section above
- **Error Messages**: Check console output for detailed error information
- **Data Issues**: Verify file formats and data structure

---

## 🎯 Next Steps

1. **Start with Dashboard**: Launch the Streamlit interface for easiest experience
2. **Try Unified Data**: Explore pre-combined datasets in the "Unified Data" tab
3. **Upload Your Data**: Use the file upload feature to analyze your own data
4. **Explore AI Insights**: Set up Groq API key and try AI-powered analysis
5. **Run ETL Pipelines**: Use command line or dashboard to process data
6. **Customize Configurations**: Modify YAML configs for your specific needs

---

## 📞 Support & Contact

- **Documentation**: This guide and README files
- **Issues**: Check console output and error messages
- **Data Questions**: Verify file formats and data structure
- **API Issues**: Check Groq API key and rate limits

---

**Happy Data Processing! 🚀📊**

*This guide covers the essential setup and running instructions for the current implementation. The platform is designed to be user-friendly and accessible for both technical and non-technical users.*
