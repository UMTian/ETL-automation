#!/usr/bin/env python3
"""
Main entry point for the Universal ETL Pipeline.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from config import PipelineConfig
from pipeline import ETLPipeline
from logger import PipelineLogger


def main():
    """Main function to run the ETL pipeline."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        config_path = "config/pipeline_config.yaml"
        config = PipelineConfig.from_yaml(config_path)
        
        # Setup logging
        logging_config = config.get_logging_config()
        logger = PipelineLogger(
            name="main",
            config=logging_config
        )
        
        logger.info("Starting Universal ETL Pipeline")
        logger.info(f"Configuration loaded from: {config_path}")
        
        # Create and run pipeline
        pipeline = ETLPipeline(config)
        
        # Execute pipeline
        start_time = datetime.now()
        result = pipeline.run()
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log results
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info(f"Records processed: {result.get('record_count', 0)}")
        
        # Print summary
        print("\n" + "="*60)
        print("ETL PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Pipeline: {result.get('pipeline_name', 'Unknown')}")
        print(f"Version: {result.get('pipeline_version', 'Unknown')}")
        print(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Records Processed: {result.get('record_count', 0)}")
        
        if result.get('load_result', {}).get('file_path'):
            print(f"Output File: {result.get('load_result')['file_path']}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")
        print("Check the logs for more details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

