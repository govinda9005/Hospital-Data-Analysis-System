"""
DB Loader Module for Hospital Data Analysis Project.

Sample .env.example:
# --- .env.example ---
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=hospital_db
# DB_USER=postgres
# DB_PASSWORD=secret_password
# --------------------
"""

import os
import logging
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_engine_connection() -> Optional[Engine]:
    """
    Reads database credentials from a .env file and establishes a SQLAlchemy engine connection.

    Returns:
        Optional[Engine]: SQLAlchemy Engine instance if successful, else None.
    """
    load_dotenv()
    
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "hospital_db")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    
    # We validate presence of basic required credentials (not checking all strictly for dev purposes)
    if not db_user or not db_password:
        logger.warning("Database credentials not fully set in .env. Using defaults/empties.")
        
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Successfully established PostgreSQL connection.")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return None

def create_schema(engine: Engine) -> None:
    """
    Executes raw SQL to create two normalized tables (patients, billing) if they don't exist.
    Also creates necessary indexes.

    ER Diagram:
      +--------------------+       +------------------------+
      |      patients      |       |        billing         |
      +--------------------+       +------------------------+
      | patient_id (PK)    |<------| patient_id (FK)        |
      | name               |       | bill_id (PK)           |
      | age                |       | treatment_cost         |
      | gender             |       | medication_cost        |
      | diagnosis          |       | total_amount           |
      | admission_date     |       | payment_status         |
      | discharge_date     |       | insurance_provider     |
      | ward               |       +------------------------+
      +--------------------+
      
    Args:
        engine (Engine): SQLAlchemy engine connected to PostgreSQL.
    """
    create_patients_sql = """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(255),
        age INTEGER,
        gender VARCHAR(50),
        diagnosis VARCHAR(255),
        admission_date DATE,
        discharge_date DATE,
        ward VARCHAR(100)
    );
    """
    
    create_billing_sql = """
    CREATE TABLE IF NOT EXISTS billing (
        bill_id VARCHAR(50) PRIMARY KEY,
        patient_id VARCHAR(50),
        treatment_cost NUMERIC(10, 2),
        medication_cost NUMERIC(10, 2),
        total_amount NUMERIC(10, 2),
        payment_status VARCHAR(100),
        insurance_provider VARCHAR(255),
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
    );
    """
    
    # Notice that we ensure creating tables first safely. Note: index creation uses IF NOT EXISTS in PG 9.5+.
    indexes_sql = """
    CREATE INDEX IF NOT EXISTS idx_patient_id ON patients(patient_id);
    CREATE INDEX IF NOT EXISTS idx_admission_date ON patients(admission_date);
    CREATE INDEX IF NOT EXISTS idx_diagnosis ON patients(diagnosis);
    """
    
    logger.info("Creating schema and tables if they don't exist (patients, billing).")
    try:
        with engine.begin() as conn:
            conn.execute(text(create_patients_sql))
            conn.execute(text(create_billing_sql))
            # Some versions of sqlalchemy/postgres handle multi-statement texts differently. 
            # We break them up just to be safe.
            for stmt in indexes_sql.strip().split(';'):
                if stmt.strip():
                    conn.execute(text(stmt.strip() + ";"))
                    
        logger.info("Schema and indexes successfully created/verified.")
    except SQLAlchemyError as err:
        logger.error(f"Error creating schema: {err}")
        raise

def load_dataframe(df: pd.DataFrame, table_name: str, engine: Engine, if_exists: str = 'replace') -> None:
    """
    Loads a Pandas DataFrame into the specified PostgreSQL table.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        table_name (str): The target table name in the DB.
        engine (Engine): SQLAlchemy engine.
        if_exists (str): Action to take if table exists ('fail', 'replace', 'append'). Defaults to 'replace'.
                         Note: 'replace' will drop the table and recreate it without the FK constraints if you're not careful.
                         In our case, using 'append' is safer for keeping our explicitly defined schema active.
    """
    # Override 'replace' internally to 'append' with a truncate if needed,
    # or just trust the user request. But 'to_sql' replace drops the schema and loses the FK.
    # To keep the assignment robust, we will truncate the tables manually if if_exists == 'replace'.
    
    try:
        if if_exists == 'replace':
            with engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))
            actual_if_exists = 'append'
            logger.info(f"Truncated table {table_name} due to if_exists='replace'.")
        else:
            actual_if_exists = if_exists
            
        df.to_sql(table_name, con=engine, if_exists=actual_if_exists, index=False)
        logger.info(f"Successfully loaded {len(df)} rows into table '{table_name}'.")
    except Exception as e:
        logger.error(f"Error loading DataFrame into '{table_name}': {e}")
        raise

def verify_load(engine: Engine) -> None:
    """
    Runs SELECT COUNT(*) on both patients and billing tables to verify the loaded data.

    Args:
        engine (Engine): SQLAlchemy engine.
    """
    try:
        with engine.connect() as conn:
            patients_count = conn.execute(text("SELECT COUNT(*) FROM patients;")).scalar()
            billing_count = conn.execute(text("SELECT COUNT(*) FROM billing;")).scalar()
            
        logger.info(f"Verification successful - Patients Table Rows: {patients_count}")
        logger.info(f"Verification successful - Billing Table Rows: {billing_count}")
    except Exception as e:
        logger.error(f"Error during verification: {e}")

def run_loader(patients_csv_path: str, billing_csv_path: str) -> None:
    """
    Runs the full database loading pipeline.

    Args:
        patients_csv_path (str): Path to the cleaned patients CSV.
        billing_csv_path (str): Path to the cleaned billing CSV.
    """
    logger.info("=== Starting PostgreSQL Import Pipeline ===")
    
    engine = create_engine_connection()
    if not engine:
        logger.error("Pipeline stopped due to connection failure.")
        return
        
    try:
        create_schema(engine)
        
        # Load CSVs
        logger.info(f"Loading data from {patients_csv_path} and {billing_csv_path}")
        patients_df = pd.read_csv(patients_csv_path)
        billing_df = pd.read_csv(billing_csv_path)
        
        # Since billing has an FK to patients, we MUST load patients first.
        load_dataframe(patients_df, "patients", engine, if_exists="replace")
        load_dataframe(billing_df, "billing", engine, if_exists="replace")
        
        verify_load(engine)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        engine.dispose()
        
    logger.info("=== Pipeline Finished ===")

if __name__ == "__main__":
    if os.path.exists("data/cleaned/patients_cleaned.csv") and os.path.exists("data/cleaned/billing_cleaned.csv"):
        run_loader("data/cleaned/patients_cleaned.csv", "data/cleaned/billing_cleaned.csv")
    else:
        logger.error("Cleaned data not found. Please ensure data/cleaned/ has patients_cleaned.csv and billing_cleaned.csv")
